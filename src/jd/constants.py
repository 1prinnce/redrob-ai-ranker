"""Job-description constants used for parsing, filtering, and scoring."""

MUST_HAVE_SKILLS: list[str] = [
    "embedding retrieval",
    "vector database",
    "python",
    "evaluation framework",
    "ndcg",
    "ranking system",
    "sentence transformers",
    "faiss",
    "pinecone",
]

NICE_TO_HAVE_SKILLS: list[str] = [
    "semantic search",
    "hybrid search",
    "reranking",
    "cross encoder",
    "bi encoder",
    "ann search",
    "hnsw",
    "bm25",
    "recall evaluation",
    "mrr",
    "information retrieval",
    "rag",
    "langchain",
    "llamaindex",
    "elasticsearch",
    "opensearch",
    "weaviate",
    "milvus",
    "qdrant",
]

DISQUALIFIER_KEYWORDS: list[str] = [
    "consulting-only",
    "staff augmentation",
    "body shopping",
    "outsourcing delivery",
    "implementation partner",
    "research-only",
    "publication-focused",
    "phd researcher",
    "postdoctoral researcher",
    "academic researcher",
    "faculty",
    "professor",
    "lecturer",
]

PENALIZED_PRIMARY_SKILLS: list[str] = [
    "computer vision",
    "speech recognition",
    "robotics",
    "image classification",
]

EXPERIENCE_MIN: int = 3
EXPERIENCE_MAX: int = 15

JD_TEXT: str = """
Placeholder job description for an AI retrieval and ranking systems role.
"""
