"""Measure real local-model performance on this machine.

Unlike the deterministic test suite, this script talks to the running Ollama
instance and reads actual GPU/RAM values. It reports:

- hard numbers from Ollama responses (load time, prompt tokens/s, output
  tokens/s, embedding throughput),
- sampled nvidia-smi utilization/VRAM and host RAM,
- clearly separated derived estimates (never mixed into measurements).

Run from ``backend/``:

    .\\.venv\\Scripts\\python.exe -m scripts.benchmark_local
    .\\.venv\\Scripts\\python.exe -m scripts.benchmark_local --json report.json
    .\\.venv\\Scripts\\python.exe -m scripts.benchmark_local --skip-cold-load

The script unloads the app models before and after the run so the GPU is left
clean. It never calls a cloud service.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

try:
    from app.core.config import settings
except Exception:  # pragma: no cover - allows running as a plain script
    settings = None


DEFAULT_HOST = getattr(settings, "OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = getattr(settings, "OLLAMA_MODEL", "qwen3:4b")
DEFAULT_EMBED_MODEL = getattr(settings, "OLLAMA_EMBED_MODEL", "nomic-embed-text")
DEFAULT_CONTEXT = getattr(settings, "MODEL_CONTEXT_LENGTH", 4096)
DEFAULT_NUM_BATCH = getattr(settings, "OLLAMA_NUM_BATCH", 512)


def gpu_query() -> Optional[Dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    command = [
        executable,
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,"
        "temperature.gpu,driver_version,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=3, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    fields = [field.strip() for field in completed.stdout.strip().splitlines()[0].split(",")]
    if len(fields) < 8:
        return None

    def number(value: str) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    return {
        "name": fields[0],
        "vram_total_mb": number(fields[1]),
        "vram_used_mb": number(fields[2]),
        "vram_free_mb": number(fields[3]),
        "utilization_percent": number(fields[4]),
        "temperature_c": number(fields[5]),
        "driver_version": fields[6],
        "power_w": number(fields[7]),
    }


class Sampler(threading.Thread):
    """Background GPU/RAM sampler running while requests execute."""

    def __init__(self, interval: float = 0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_event = threading.Event()
        self.gpu_samples: List[Dict[str, Any]] = []
        self.ram_samples: List[float] = []

    def run(self) -> None:
        while not self.stop_event.is_set():
            sample = gpu_query()
            if sample:
                self.gpu_samples.append(sample)
            if psutil is not None:
                self.ram_samples.append(
                    float(psutil.virtual_memory().available) / (1024 * 1024)
                )
            self.stop_event.wait(self.interval)

    def start(self) -> "Sampler":
        super().start()
        return self

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=3)

    def summary(self) -> Dict[str, Any]:
        def stats(values: List[float]) -> Optional[Dict[str, float]]:
            if not values:
                return None
            return {
                "avg": round(statistics.fmean(values), 1),
                "max": round(max(values), 1),
                "min": round(min(values), 1),
                "samples": len(values),
            }

        gpu: Optional[Dict[str, Any]] = None
        if self.gpu_samples:
            gpu = {
                "name": self.gpu_samples[0].get("name"),
                "utilization_percent": stats([
                    s["utilization_percent"] for s in self.gpu_samples
                    if s.get("utilization_percent") is not None
                ]),
                "vram_used_mb": stats([
                    s["vram_used_mb"] for s in self.gpu_samples
                    if s.get("vram_used_mb") is not None
                ]),
                "temperature_c": stats([
                    s["temperature_c"] for s in self.gpu_samples
                    if s.get("temperature_c") is not None
                ]),
            }
        return {
            "gpu": gpu,
            "system_available_mb": stats(self.ram_samples),
        }


class OllamaProbe:
    def __init__(self, host: str, timeout: float = 300.0):
        self.host = host.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self.client.close()

    def tags(self) -> List[str]:
        response = self.client.get(f"{self.host}/api/tags")
        response.raise_for_status()
        return [item.get("name", "") for item in response.json().get("models", [])]

    def ps(self) -> List[Dict[str, Any]]:
        response = self.client.get(f"{self.host}/api/ps")
        response.raise_for_status()
        return response.json().get("models", [])

    def unload(self, model: str) -> None:
        try:
            self.client.post(
                f"{self.host}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": 0},
            )
        except Exception:
            pass

    def unload_app_models(self, models: List[str]) -> None:
        loaded = {item.get("name") or item.get("model") for item in self.ps()}
        for model in models:
            if model in loaded or any(
                str(name).startswith(model.split(":")[0]) for name in loaded if name
            ):
                self.unload(model)

    def chat(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        context: int,
        num_batch: int,
        keep_alive: str,
        think: bool = False,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        response = self.client.post(
            f"{self.host}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": think,
                "keep_alive": keep_alive,
                "options": {
                    "temperature": 0.1,
                    "num_predict": max_tokens,
                    "num_ctx": context,
                    "num_batch": num_batch,
                },
            },
        )
        wall_seconds = time.perf_counter() - started
        response.raise_for_status()
        data = response.json()

        def seconds(value: Any) -> float:
            try:
                return float(value) / 1_000_000_000.0
            except (TypeError, ValueError):
                return 0.0

        eval_count = int(data.get("eval_count") or 0)
        eval_seconds = seconds(data.get("eval_duration"))
        prompt_count = int(data.get("prompt_eval_count") or 0)
        prompt_seconds = seconds(data.get("prompt_eval_duration"))
        load_seconds = seconds(data.get("load_duration"))

        return {
            "wall_seconds": round(wall_seconds, 3),
            "load_seconds": round(load_seconds, 3),
            "prompt_tokens": prompt_count,
            "output_tokens": eval_count,
            "prompt_tokens_per_second": (
                round(prompt_count / prompt_seconds, 1) if prompt_seconds > 0 else None
            ),
            "output_tokens_per_second": (
                round(eval_count / eval_seconds, 1) if eval_seconds > 0 else None
            ),
            "total_seconds_reported": round(
                seconds(data.get("total_duration")), 3
            ),
            "response_chars": len(
                (data.get("message") or {}).get("content", "")
            ),
        }

    def embed(
        self,
        model: str,
        texts: List[str],
        keep_alive: str,
        dimensions: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "input": texts,
            "keep_alive": keep_alive,
        }
        if dimensions:
            payload["dimensions"] = dimensions
        started = time.perf_counter()
        response = self.client.post(f"{self.host}/api/embed", json=payload)
        wall_seconds = time.perf_counter() - started
        response.raise_for_status()
        data = response.json()
        vectors = data.get("embeddings") or []
        return {
            "batch_size": len(texts),
            "wall_seconds": round(wall_seconds, 3),
            "dimension": len(vectors[0]) if vectors else None,
            "texts_per_second": (
                round(len(texts) / wall_seconds, 2) if wall_seconds > 0 else None
            ),
        }


def build_embedding_texts(count: int) -> List[str]:
    sentence = (
        "KnowledgeGraph AI extracts entities, relationships and page-level "
        "evidence from local PDF documents using a small local language model. "
    )
    return [f"Chunk {index}. {sentence * 2}" for index in range(count)]


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    probe = OllamaProbe(args.host, timeout=args.timeout)
    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "measurement_note": (
            "All values under measurements come from this machine/Ollama. "
            "Estimates are derived arithmetically and labeled separately."
        ),
        "host": {},
        "config": {
            "model": args.model,
            "embed_model": args.embed_model,
            "context_length": args.context,
            "num_batch": args.num_batch,
            "keep_alive": args.keep_alive,
            "think": args.think,
        },
        "measurements": {},
        "estimates": {},
    }

    try:
        models = probe.tags()
        report["host"]["ollama_models"] = models
        for required in (args.model, args.embed_model):
            if not any(required in name for name in models):
                raise RuntimeError(
                    f"Required Ollama model not found: {required}. "
                    f"Run: ollama pull {required}"
                )

        if psutil is not None:
            memory = psutil.virtual_memory()
            report["host"]["system_ram_total_mb"] = int(memory.total / (1024 * 1024))
            report["host"]["system_ram_available_mb"] = int(memory.available / (1024 * 1024))
        gpu = gpu_query()
        report["host"]["gpu"] = gpu
        report["host"]["platform"] = {
            "python": __import__("sys").version.split()[0],
            "executable": __import__("sys").executable,
        }

        loaded_before = probe.ps()
        report["measurements"]["models_loaded_before"] = [
            {
                "name": item.get("name"),
                "size_mb": int((item.get("size") or 0) / (1024 * 1024)),
                "vram_mb": int((item.get("size_vram") or 0) / (1024 * 1024)),
            }
            for item in loaded_before
        ]

        if not args.skip_cold_load:
            probe.unload(args.model)
            time.sleep(1.0)
            sampler = Sampler().start()
            cold = probe.chat(
                args.model,
                "Reply with the single word: ready",
                max_tokens=8,
                context=args.context,
                num_batch=args.num_batch,
                keep_alive=args.keep_alive,
                think=False,
            )
            sampler.stop()
            report["measurements"]["cold_load"] = {
                **cold,
                "samples": sampler.summary(),
            }

        # Warm generation. The first call after cold load is reported too, but
        # the median of several calls is the stable number.
        generation_runs = []
        prompts = [
            "In one sentence, what is a knowledge graph?",
            "List three differences between vector search and graph traversal.",
            "Explain why citations matter in a document question-answering system.",
            "Summarize the role of embeddings in retrieval in two sentences.",
        ]
        sampler = Sampler().start()
        for prompt in prompts[: args.generation_runs]:
            generation_runs.append(
                probe.chat(
                    args.model,
                    prompt,
                    max_tokens=args.max_tokens,
                    context=args.context,
                    num_batch=args.num_batch,
                    keep_alive=args.keep_alive,
                    think=args.think,
                )
            )
        sampler.stop()
        rates = [
            run["output_tokens_per_second"]
            for run in generation_runs
            if run.get("output_tokens_per_second")
        ]
        report["measurements"]["generation"] = {
            "runs": generation_runs,
            "median_output_tokens_per_second": (
                round(statistics.median(rates), 1) if rates else None
            ),
            "samples": sampler.summary(),
        }

        # Embedding throughput at several batch sizes. Small batches are what
        # the indexer actually uses on a 6 GB card.
        embedding_runs = []
        for batch_size in args.embed_batch_sizes:
            texts = build_embedding_texts(batch_size)
            # Warm the embedding model once per size, then report the second run.
            probe.embed(
                args.embed_model, texts, keep_alive=args.keep_alive,
                dimensions=args.embed_dimensions,
            )
            sampler = Sampler().start()
            result = probe.embed(
                args.embed_model, texts, keep_alive=args.keep_alive,
                dimensions=args.embed_dimensions,
            )
            sampler.stop()
            result["samples"] = sampler.summary()
            embedding_runs.append(result)
        report["measurements"]["embeddings"] = {
            "runs": embedding_runs,
            "dimension_expected": (
                args.embed_dimensions
                or getattr(settings, "EMBEDDING_DIMENSION", None)
            ),
        }

        loaded_after = probe.ps()
        report["measurements"]["models_loaded_after"] = [
            {
                "name": item.get("name"),
                "size_mb": int((item.get("size") or 0) / (1024 * 1024)),
                "vram_mb": int((item.get("size_vram") or 0) / (1024 * 1024)),
            }
            for item in loaded_after
        ]

        # Derived estimates use only measured rates; they are not measurements.
        best_embed = max(
            (
                run["texts_per_second"]
                for run in embedding_runs
                if run.get("texts_per_second")
            ),
            default=None,
        )
        median_generation = report["measurements"]["generation"][
            "median_output_tokens_per_second"
        ]
        if best_embed:
            report["estimates"]["chunks_per_minute_at_best_embed_batch"] = round(
                best_embed * 60, 1
            )
        if median_generation:
            # Graph extraction asks for JSON and is not the same workload; this
            # is only a rough upper-bound style estimate.
            report["estimates"]["output_tokens_per_minute"] = round(
                median_generation * 60, 1
            )
        return report
    finally:
        probe.unload_app_models([args.model, args.embed_model])
        probe.close()


def format_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    host = report.get("host", {})
    gpu = host.get("gpu") or {}
    lines.append("KnowledgeGraph AI - local benchmark")
    lines.append("=" * 40)
    lines.append(f"Generated: {report.get('generated_at')}")
    lines.append(
        f"RAM: {host.get('system_ram_total_mb')} MB total, "
        f"{host.get('system_ram_available_mb')} MB free before the run"
    )
    if gpu:
        lines.append(
            f"GPU: {gpu.get('name')} | {gpu.get('vram_total_mb')} MB VRAM | "
            f"driver {gpu.get('driver_version')}"
        )
    else:
        lines.append("GPU: nvidia-smi unavailable (Ollama may use CPU)")

    measurements = report.get("measurements", {})
    cold = measurements.get("cold_load")
    if cold:
        lines.append("")
        lines.append("MEASURED - cold load")
        lines.append(
            f"  load {cold.get('load_seconds')}s | wall {cold.get('wall_seconds')}s | "
            f"{cold.get('output_tokens_per_second')} output tok/s"
        )
        gpu_summary = cold.get("samples", {}).get("gpu") or {}
        util = gpu_summary.get("utilization_percent") or {}
        vram = gpu_summary.get("vram_used_mb") or {}
        if util:
            lines.append(
                f"  GPU util avg {util.get('avg')}% max {util.get('max')}% | "
                f"VRAM used max {vram.get('max')} MB"
            )

    generation = measurements.get("generation", {})
    runs = generation.get("runs", [])
    if runs:
        lines.append("")
        lines.append("MEASURED - generation")
        for index, run in enumerate(runs, start=1):
            lines.append(
                f"  run {index}: wall {run.get('wall_seconds')}s | "
                f"prompt {run.get('prompt_tokens')} tok @ "
                f"{run.get('prompt_tokens_per_second')} tok/s | "
                f"output {run.get('output_tokens')} tok @ "
                f"{run.get('output_tokens_per_second')} tok/s"
            )
        lines.append(
            f"  median output rate: "
            f"{generation.get('median_output_tokens_per_second')} tok/s"
        )
        gpu_summary = generation.get("samples", {}).get("gpu") or {}
        util = gpu_summary.get("utilization_percent") or {}
        vram = gpu_summary.get("vram_used_mb") or {}
        if util:
            lines.append(
                f"  GPU util avg {util.get('avg')}% max {util.get('max')}% | "
                f"VRAM used max {vram.get('max')} MB"
            )

    embeddings = measurements.get("embeddings", {})
    embed_runs = embeddings.get("runs", [])
    if embed_runs:
        lines.append("")
        lines.append("MEASURED - embeddings")
        for run in embed_runs:
            lines.append(
                f"  batch {run.get('batch_size')}: wall {run.get('wall_seconds')}s | "
                f"{run.get('texts_per_second')} texts/s | dim {run.get('dimension')}"
            )

    estimates = report.get("estimates", {})
    if estimates:
        lines.append("")
        lines.append("ESTIMATED (derived from measured rates, not measured)")
        if "chunks_per_minute_at_best_embed_batch" in estimates:
            lines.append(
                "  embedding ceiling: "
                f"{estimates['chunks_per_minute_at_best_embed_batch']} chunks/min"
            )
        if "output_tokens_per_minute" in estimates:
            lines.append(
                "  generation ceiling: "
                f"{estimates['output_tokens_per_minute']} output tokens/min"
            )

    lines.append("")
    lines.append("Models were unloaded after the run.")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--num-batch", type=int, default=DEFAULT_NUM_BATCH)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--keep-alive", default="5m")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--generation-runs", type=int, default=2)
    parser.add_argument(
        "--embed-batch-sizes", type=int, nargs="+", default=[1, 8, 16]
    )
    parser.add_argument(
        "--embed-dimensions",
        type=int,
        default=0,
        help="Optional output dimension; 0 lets the model use its native size",
    )
    parser.add_argument("--skip-cold-load", action="store_true")
    parser.add_argument("--json", default=None, help="Write the full report to this file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_benchmark(args)
    except Exception as exc:
        print(f"Benchmark failed: {type(exc).__name__}: {exc}")
        return 1

    print(format_report(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nJSON report: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
