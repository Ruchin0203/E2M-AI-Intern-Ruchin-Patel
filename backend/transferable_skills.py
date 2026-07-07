"""
transferable_skills.py
Deterministic taxonomy mapping technologies to broader categories, used to
detect transferable-skill matches (e.g., Flask -> Backend Framework <- FastAPI).
"""

from typing import Dict, List, Optional
from utils import normalize_skill

# Category -> list of technologies belonging to it.
SKILL_CATEGORIES: Dict[str, List[str]] = {
    "Backend Framework": [
        "fastapi", "flask", "django", "express", "express.js", "spring",
        "spring boot", "nestjs", "koa", "asp.net", "gin", "laravel",
    ],
    "Frontend Framework": [
        "react", "react.js", "vue", "vue.js", "angular", "svelte",
        "next.js", "nextjs", "nuxt.js", "solidjs",
    ],
    "Cloud Platform": [
        "aws", "amazon web services", "azure", "gcp", "google cloud",
        "google cloud platform", "digitalocean", "heroku", "oracle cloud",
    ],
    "Deep Learning Framework": [
        "tensorflow", "pytorch", "keras", "jax", "mxnet", "caffe",
    ],
    "Relational Database": [
        "mysql", "postgresql", "postgres", "sqlite", "mssql", "oracle db",
        "mariadb",
    ],
    "NoSQL Database": [
        "mongodb", "cassandra", "dynamodb", "couchdb", "redis", "firebase firestore",
    ],
    "Vector Database": [
        "faiss", "pinecone", "chromadb", "chroma", "weaviate", "qdrant", "milvus",
    ],
    "Containerization": [
        "docker", "podman", "containerd",
    ],
    "Orchestration": [
        "kubernetes", "k8s", "docker swarm", "nomad",
    ],
    "CI/CD": [
        "github actions", "jenkins", "gitlab ci", "circleci", "travis ci", "argo cd",
    ],
    "LLM Orchestration": [
        "langchain", "langgraph", "llamaindex", "haystack", "semantic kernel",
    ],
    "Computer Vision": [
        "opencv", "yolo", "yolov8", "yolov5", "detectron2", "mediapipe",
    ],
    "API Style": [
        "rest api", "restful api", "graphql", "grpc", "soap",
    ],
    "Programming Language": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
        "kotlin", "swift", "php", "ruby",
    ],
    "Data Processing": [
        "pandas", "numpy", "spark", "pyspark", "dask", "polars",
    ],
    "Version Control": [
        "git", "github", "gitlab", "bitbucket", "svn",
    ],
    "Automation": [
        "n8n", "zapier", "make", "airflow", "prefect",
    ],
}

# Reverse index: normalized skill -> category
_SKILL_TO_CATEGORY: Dict[str, str] = {}
for category, skills in SKILL_CATEGORIES.items():
    for skill in skills:
        _SKILL_TO_CATEGORY[normalize_skill(skill)] = category


def get_category(skill: str) -> Optional[str]:
    """Return the taxonomy category for a given skill, if known."""
    return _SKILL_TO_CATEGORY.get(normalize_skill(skill))


def same_category(skill_a: str, skill_b: str) -> Optional[str]:
    """
    Return the shared category name if two skills belong to the same
    transferable-skill category, otherwise None.
    """
    cat_a = get_category(skill_a)
    cat_b = get_category(skill_b)
    if cat_a and cat_a == cat_b:
        return cat_a
    return None
