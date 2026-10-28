export interface Document {
  document_id: string;
  original_filename: string;
  file_size: number;
  page_count: number;
  status: string;
  uploaded_at: string;
  processed_at?: string;
  error_message?: string;
}

export interface DocumentDetail extends Document {
  pages?: Array<{
    page_number: number;
    text: string;
    char_count: number;
  }>;
}

export interface ChatRequest {
  question: string;
  document_id?: string;
}

export interface Citation {
  chunk_id: string;
  document_id: string;
  document_name?: string;
  page_number: number;
  snippet: string;
}

export interface ChatResponse {
  question: string;
  answer: string;
  citations: Citation[];
  evidence: Array<Record<string, unknown>>;
  has_evidence: boolean;
  retrieval_metadata: Record<string, unknown>;
  message?: string;
}

export interface HealthResponse {
  status: string;
  services: Record<string, unknown>;
}

export interface ModelStatus {
  ollama_host: string;
  llm: { model: string; available: boolean; error?: string };
  embeddings: { model: string; available: boolean; error?: string };
  all_available: boolean;
}

export interface GraphNode {
  id: string;
  label: string;
  type: string;
  aliases?: string[];
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label: string;
  confidence?: number;
}

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  total_nodes: number;
  total_edges: number;
}
