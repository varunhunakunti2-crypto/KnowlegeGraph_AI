"""Synthetic corpus, gold annotations, and question set for GraphRAG evaluation.

The corpus is deliberately small, entirely synthetic-to-the-project, and safe to
commit: three short PDFs (8 pages / 8 chunks at the default 512-character chunk
size). Gold entities, relationships, and answers are declared next to the page
text so every metric has an explicit reference.

The content is public-domain historical/ factual prose written for this suite;
no private documents are used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class GoldEntity:
    """An entity that should be extracted from one page."""

    name: str
    type: str


@dataclass(frozen=True)
class GoldRelationship:
    """A subject-predicate-object triple with its source evidence sentence."""

    subject: str
    predicate: str
    object: str
    evidence: str


@dataclass(frozen=True)
class SamplePage:
    """One synthetic PDF page and its gold annotations."""

    page_number: int
    text: str
    entities: Sequence[GoldEntity] = field(default_factory=tuple)
    relationships: Sequence[GoldRelationship] = field(default_factory=tuple)
    # Deliberate model noise used to prove that sanitisation and metrics work.
    noise_entities: Sequence[GoldEntity] = field(default_factory=tuple)
    noise_relationships: Sequence[GoldRelationship] = field(default_factory=tuple)


@dataclass(frozen=True)
class SampleDocument:
    """One synthetic PDF."""

    key: str
    filename: str
    title: str
    pages: Sequence[SamplePage]

    def page(self, page_number: int) -> SamplePage:
        for page in self.pages:
            if page.page_number == page_number:
                return page
        raise KeyError(page_number)


@dataclass(frozen=True)
class SourceRef:
    """A page-level source reference used by retrieval/citation metrics."""

    document_key: str
    page_number: int


@dataclass(frozen=True)
class EvalQuestion:
    """A question with a known answer and expected supporting sources."""

    key: str
    question: str
    answer_keywords: Sequence[str]
    expected_sources: Sequence[SourceRef]
    requires_graph: bool = False
    document_scope: str | None = None
    expect_evidence: bool = True
    forbidden_keywords: Sequence[str] = field(default_factory=tuple)
    notes: str = ""


ADA = SampleDocument(
    key="ada_lovelace",
    filename="ada_lovelace.pdf",
    title="Ada Lovelace and the Analytical Engine",
    pages=(
        SamplePage(
            page_number=1,
            text=(
                "Charles Babbage designed the Analytical Engine in 1837. "
                "Babbage was a mathematician who lived in London. "
                "The Difference Engine was an earlier machine designed by Babbage."
            ),
            entities=(
                GoldEntity("Charles Babbage", "Person"),
                GoldEntity("Analytical Engine", "Technology"),
                GoldEntity("London", "Location"),
                GoldEntity("Difference Engine", "Technology"),
            ),
            relationships=(
                GoldRelationship(
                    "Charles Babbage",
                    "CREATED",
                    "Analytical Engine",
                    "Charles Babbage designed the Analytical Engine in 1837.",
                ),
                GoldRelationship(
                    "Charles Babbage",
                    "LOCATED_IN",
                    "London",
                    "Babbage was a mathematician who lived in London.",
                ),
                GoldRelationship(
                    "Charles Babbage",
                    "CREATED",
                    "Difference Engine",
                    "The Difference Engine was an earlier machine designed by Babbage.",
                ),
            ),
            noise_entities=(
                # Invalid entity type: the sanitiser must drop it.
                GoldEntity("Mathematician", "Scientist"),
            ),
            noise_relationships=(
                # Invalid predicate: it is preserved as RELATED_TO, never as
                # the model-invented label.
                GoldRelationship(
                    "Charles Babbage",
                    "INVENTED",
                    "Analytical Engine",
                    "Charles Babbage designed the Analytical Engine in 1837.",
                ),
            ),
        ),
        SamplePage(
            page_number=2,
            text=(
                "Ada Lovelace worked with Charles Babbage on the Analytical Engine. "
                "In 1843, Ada published a translation of Menabrea's notes with "
                "extensive annotations. Her notes explained how the engine could "
                "compute Bernoulli numbers."
            ),
            entities=(
                GoldEntity("Ada Lovelace", "Person"),
                GoldEntity("Charles Babbage", "Person"),
                GoldEntity("Analytical Engine", "Technology"),
                GoldEntity("Bernoulli numbers", "Concept"),
                GoldEntity("Menabrea", "Person"),
            ),
            relationships=(
                GoldRelationship(
                    "Ada Lovelace",
                    "RELATED_TO",
                    "Charles Babbage",
                    "Ada Lovelace worked with Charles Babbage on the Analytical Engine.",
                ),
                GoldRelationship(
                    "Ada Lovelace",
                    "MENTIONS",
                    "Analytical Engine",
                    "Ada Lovelace worked with Charles Babbage on the Analytical Engine.",
                ),
                GoldRelationship(
                    "Ada Lovelace",
                    "MENTIONS",
                    "Bernoulli numbers",
                    "Her notes explained how the engine could compute Bernoulli numbers.",
                ),
                GoldRelationship(
                    "Ada Lovelace",
                    "MENTIONS",
                    "Menabrea",
                    "In 1843, Ada published a translation of Menabrea's notes with "
                    "extensive annotations.",
                ),
            ),
        ),
        SamplePage(
            page_number=3,
            text=(
                "Ada Lovelace is often called the first computer programmer. "
                "She described the Analytical Engine as a general-purpose computer. "
                "Her published notes are still studied today."
            ),
            entities=(
                GoldEntity("Ada Lovelace", "Person"),
                GoldEntity("Analytical Engine", "Technology"),
                GoldEntity("computer programmer", "Concept"),
            ),
            relationships=(
                GoldRelationship(
                    "Ada Lovelace",
                    "RELATED_TO",
                    "Analytical Engine",
                    "She described the Analytical Engine as a general-purpose computer.",
                ),
                GoldRelationship(
                    "Ada Lovelace",
                    "RELATED_TO",
                    "computer programmer",
                    "Ada Lovelace is often called the first computer programmer.",
                ),
            ),
        ),
    ),
)


APOLLO = SampleDocument(
    key="apollo11",
    filename="apollo11.pdf",
    title="Apollo 11 and the first Moon landing",
    pages=(
        SamplePage(
            page_number=1,
            text=(
                "Apollo 11 was launched by NASA in July 1969. "
                "The mission used the Saturn V rocket. "
                "NASA is an American space agency."
            ),
            entities=(
                GoldEntity("Apollo 11", "Event"),
                GoldEntity("NASA", "Organization"),
                GoldEntity("Saturn V", "Technology"),
            ),
            relationships=(
                GoldRelationship(
                    "NASA",
                    "CREATED",
                    "Apollo 11",
                    "Apollo 11 was launched by NASA in July 1969.",
                ),
                GoldRelationship(
                    "Apollo 11",
                    "USES",
                    "Saturn V",
                    "The mission used the Saturn V rocket.",
                ),
            ),
            noise_entities=(
                GoldEntity("Saturn V", "Rocket"),
            ),
            noise_relationships=(
                GoldRelationship(
                    "NASA",
                    "BUILT",
                    "Saturn V",
                    "The mission used the Saturn V rocket.",
                ),
            ),
        ),
        SamplePage(
            page_number=2,
            text=(
                "In July 1969, Neil Armstrong and Buzz Aldrin landed the lunar module "
                "Eagle on the Moon. Neil Armstrong was the first person to walk on the "
                "Moon. NASA is headquartered in Washington."
            ),
            entities=(
                GoldEntity("Neil Armstrong", "Person"),
                GoldEntity("Buzz Aldrin", "Person"),
                GoldEntity("Eagle", "Technology"),
                GoldEntity("Moon", "Location"),
                GoldEntity("NASA", "Organization"),
                GoldEntity("Washington", "Location"),
            ),
            relationships=(
                GoldRelationship(
                    "Buzz Aldrin",
                    "RELATED_TO",
                    "Moon",
                    "In July 1969, Neil Armstrong and Buzz Aldrin landed the lunar module "
                    "Eagle on the Moon.",
                ),
                GoldRelationship(
                    "Neil Armstrong",
                    "RELATED_TO",
                    "Moon",
                    "Neil Armstrong was the first person to walk on the Moon.",
                ),
                GoldRelationship(
                    "NASA",
                    "LOCATED_IN",
                    "Washington",
                    "NASA is headquartered in Washington.",
                ),
            ),
        ),
        SamplePage(
            page_number=3,
            text=(
                "Neil Armstrong's childhood hobby was the trumpet. "
                "He learned the trumpet in Wapakoneta, Ohio."
            ),
            entities=(
                GoldEntity("Neil Armstrong", "Person"),
                GoldEntity("trumpet", "Concept"),
                GoldEntity("Wapakoneta", "Location"),
                GoldEntity("Ohio", "Location"),
            ),
            relationships=(
                GoldRelationship(
                    "Neil Armstrong",
                    "RELATED_TO",
                    "trumpet",
                    "Neil Armstrong's childhood hobby was the trumpet.",
                ),
                GoldRelationship(
                    "Neil Armstrong",
                    "LOCATED_IN",
                    "Wapakoneta",
                    "He learned the trumpet in Wapakoneta, Ohio.",
                ),
            ),
            noise_entities=(
                GoldEntity("Ohio", "State"),
            ),
        ),
    ),
)


PHOTOSYNTHESIS = SampleDocument(
    key="photosynthesis",
    filename="photosynthesis.pdf",
    title="Photosynthesis",
    pages=(
        SamplePage(
            page_number=1,
            text=(
                "Photosynthesis converts light energy into chemical energy in plants. "
                "Chlorophyll absorbs light inside chloroplasts."
            ),
            entities=(
                GoldEntity("Photosynthesis", "Concept"),
                GoldEntity("light", "Concept"),
                GoldEntity("plants", "Concept"),
                GoldEntity("Chlorophyll", "Concept"),
                GoldEntity("chloroplasts", "Concept"),
            ),
            relationships=(
                GoldRelationship(
                    "Photosynthesis",
                    "RELATED_TO",
                    "light",
                    "Photosynthesis converts light energy into chemical energy in plants.",
                ),
                GoldRelationship(
                    "Chlorophyll",
                    "PART_OF",
                    "chloroplasts",
                    "Chlorophyll absorbs light inside chloroplasts.",
                ),
            ),
            noise_entities=(
                GoldEntity("Energy", "Force"),
            ),
            noise_relationships=(
                GoldRelationship(
                    "Chlorophyll",
                    "ABSORBS",
                    "light",
                    "Chlorophyll absorbs light inside chloroplasts.",
                ),
            ),
        ),
        SamplePage(
            page_number=2,
            text=(
                "Plants use carbon dioxide and water to produce glucose and oxygen. "
                "Photosynthesis occurs in chloroplasts."
            ),
            entities=(
                GoldEntity("Plants", "Concept"),
                GoldEntity("carbon dioxide", "Concept"),
                GoldEntity("water", "Concept"),
                GoldEntity("glucose", "Concept"),
                GoldEntity("oxygen", "Concept"),
                GoldEntity("Photosynthesis", "Concept"),
                GoldEntity("chloroplasts", "Concept"),
            ),
            relationships=(
                GoldRelationship(
                    "Plants",
                    "USES",
                    "carbon dioxide",
                    "Plants use carbon dioxide and water to produce glucose and oxygen.",
                ),
                GoldRelationship(
                    "Plants",
                    "USES",
                    "water",
                    "Plants use carbon dioxide and water to produce glucose and oxygen.",
                ),
                GoldRelationship(
                    "Plants",
                    "RELATED_TO",
                    "glucose",
                    "Plants use carbon dioxide and water to produce glucose and oxygen.",
                ),
                GoldRelationship(
                    "Photosynthesis",
                    "LOCATED_IN",
                    "chloroplasts",
                    "Photosynthesis occurs in chloroplasts.",
                ),
            ),
        ),
    ),
)


@dataclass(frozen=True)
class SyntheticCorpus:
    """The full evaluation corpus plus the question set."""

    documents: Sequence[SampleDocument]
    questions: Sequence[EvalQuestion]

    def document(self, key: str) -> SampleDocument:
        for document in self.documents:
            if document.key == key:
                return document
        raise KeyError(key)

    def question(self, key: str) -> EvalQuestion:
        for question in self.questions:
            if question.key == key:
                return question
        raise KeyError(key)

    def gold_relationships(self) -> list[GoldRelationship]:
        return [
            relationship
            for document in self.documents
            for page in document.pages
            for relationship in page.relationships
        ]

    def gold_entities(self) -> list[GoldEntity]:
        return [
            entity
            for document in self.documents
            for page in document.pages
            for entity in page.entities
        ]


CORPUS = SyntheticCorpus(
    documents=(ADA, APOLLO, PHOTOSYNTHESIS),
    questions=(
        EvalQuestion(
            key="q_babbage",
            question="Who designed the Analytical Engine?",
            answer_keywords=("charles babbage",),
            expected_sources=(SourceRef("ada_lovelace", 1),),
            notes="Single-hop vector question; answer is on page 1.",
        ),
        EvalQuestion(
            key="q_ada",
            question="Who worked with Charles Babbage on the Analytical Engine?",
            answer_keywords=("ada lovelace",),
            expected_sources=(SourceRef("ada_lovelace", 2),),
            notes="Relationship fact stated directly in one chunk.",
        ),
        EvalQuestion(
            key="q_1843",
            question="What did Ada Lovelace publish in 1843?",
            answer_keywords=("translation",),
            expected_sources=(SourceRef("ada_lovelace", 2),),
            notes="Same chunk as q_ada; useful for ranking checks.",
        ),
        EvalQuestion(
            key="q_apollo",
            question="Who was the first person to walk on the Moon?",
            answer_keywords=("neil armstrong",),
            expected_sources=(SourceRef("apollo11", 2),),
            notes="Single-hop vector question.",
        ),
        EvalQuestion(
            key="q_bridge",
            question="Which instrument did the first person to walk on the Moon play?",
            answer_keywords=("trumpet",),
            expected_sources=(SourceRef("apollo11", 3),),
            requires_graph=True,
            document_scope="apollo11",
            notes=(
                "Connected-fact question. The answer sentence shares almost no "
                "vocabulary with the question, so a tight vector window misses "
                "it; the graph reaches it through the Neil Armstrong entity."
            ),
        ),
        EvalQuestion(
            key="q_multi",
            question="Which organization launched the mission that Neil Armstrong flew on?",
            answer_keywords=("nasa",),
            expected_sources=(SourceRef("apollo11", 1), SourceRef("apollo11", 2)),
            requires_graph=True,
            document_scope="apollo11",
            notes="Two-page join: Apollo 11/NASA on page 1, Armstrong on page 2.",
        ),
        EvalQuestion(
            key="q_missing",
            question="What is the capital of France?",
            answer_keywords=(),
            expected_sources=(),
            expect_evidence=False,
            notes=(
                "Out-of-corpus question. The retriever has no relevance floor, so "
                "it returns low-similarity chunks; the generator must abstain and "
                "must not fabricate citations."
            ),
        ),
        EvalQuestion(
            key="q_scope",
            question="Who designed the Analytical Engine?",
            answer_keywords=(),
            expected_sources=(),
            document_scope="photosynthesis",
            forbidden_keywords=("charles babbage", "ada lovelace"),
            notes=(
                "The same question scoped to an unrelated document must not leak "
                "entities from the Ada Lovelace document."
            ),
        ),
    ),
)


def page_texts() -> list[str]:
    """All page texts, used to build the deterministic embedding vocabulary."""

    return [
        page.text
        for document in CORPUS.documents
        for page in document.pages
    ]
