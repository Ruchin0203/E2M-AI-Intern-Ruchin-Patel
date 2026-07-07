"""
config.py
Centralized application configuration loaded from environment variables.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings."""

    APP_NAME: str = "Resume-JD Matching Tool"
    APP_VERSION: str = "1.0.0"

    # Gemini / LangChain
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    # If no Gemini key is present, the app falls back to a deterministic
    # rule/NLP based extractor so the project still runs end-to-end.
    USE_LLM: bool = bool(GOOGLE_API_KEY)

    # Matching engine
    FUZZY_MATCH_THRESHOLD: int = int(os.getenv("FUZZY_MATCH_THRESHOLD", "85"))
    SEMANTIC_MATCH_THRESHOLD: float = float(os.getenv("SEMANTIC_MATCH_THRESHOLD", "0.62"))
    SENTENCE_TRANSFORMER_MODEL: str = os.getenv(
        "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"
    )

    # Scoring weights (must sum to 100)
    WEIGHT_EXACT_SKILLS: float = 40
    WEIGHT_SEMANTIC: float = 20
    WEIGHT_TRANSFERABLE: float = 15
    WEIGHT_PROJECT: float = 15
    WEIGHT_EXPERIENCE: float = 5
    WEIGHT_EDUCATION: float = 5

    # CORS
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

    # Uploads
    MAX_UPLOAD_SIZE_MB: int = 10


settings = Settings()


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
