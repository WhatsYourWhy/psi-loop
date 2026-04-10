"""Embedder protocols and concrete backend implementations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from psi_loop.text import token_counts

SparseVector = dict[str, float]
DenseVector = tuple[float, ...]
Vector = SparseVector | DenseVector


class Embedder(Protocol):
    """Minimal text-to-vector interface for Psi0 scoring."""

    def embed(self, text: str) -> Vector:
        """Embed a text string into a vector space."""


class BowEmbedder:
    """Default embedder using L2-normalized bag-of-words vectors.

    Normalizing to unit length removes document-length bias from centroid
    calculations, so a 3-word chunk and a 300-word chunk contribute equally
    to the context centroid when computing surprise scores.
    """

    def embed(self, text: str) -> SparseVector:
        counts = token_counts(text)
        raw = {token: float(value) for token, value in counts.items()}
        norm = sum(v * v for v in raw.values()) ** 0.5
        if norm == 0.0:
            return raw
        return {token: value / norm for token, value in raw.items()}


class STEmbedder:
    """Optional dense embedder powered by sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "STEmbedder requires the optional dense dependencies. "
                    "Install them with `python -m pip install -e .[dense]`."
                ) from exc

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> DenseVector:
        model = self._load_model()
        vector = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return tuple(float(value) for value in vector.tolist())


def is_sparse_vector(vector: Vector) -> bool:
    return isinstance(vector, Mapping)


def cosine_similarity_vectors(left: Vector, right: Vector) -> float:
    """Cosine similarity for either sparse or dense vectors."""

    if is_sparse_vector(left) and is_sparse_vector(right):
        shared = set(left) & set(right)
        numerator = sum(left[token] * right[token] for token in shared)
        left_norm = sum(value * value for value in left.values()) ** 0.5
        right_norm = sum(value * value for value in right.values()) ** 0.5
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)

    if not is_sparse_vector(left) and not is_sparse_vector(right):
        if len(left) != len(right):
            raise ValueError("Dense vectors must have the same dimensionality.")
        numerator = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = sum(value * value for value in left) ** 0.5
        right_norm = sum(value * value for value in right) ** 0.5
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)

    raise TypeError("Vector types must match for similarity computation.")


def centroid(vectors: Sequence[Vector]) -> Vector:
    """Compute a centroid for sparse or dense vectors."""

    if not vectors:
        raise ValueError("At least one vector is required to compute a centroid.")

    first = vectors[0]
    if is_sparse_vector(first):
        total: dict[str, float] = {}
        for vector in vectors:
            if not is_sparse_vector(vector):
                raise TypeError("All vectors must have the same type.")
            for token, value in vector.items():
                total[token] = total.get(token, 0.0) + value
        count = float(len(vectors))
        return {token: value / count for token, value in total.items()}

    dense_vectors = []
    for vector in vectors:
        if is_sparse_vector(vector):
            raise TypeError("All vectors must have the same type.")
        dense_vectors.append(vector)

    width = len(dense_vectors[0])
    if any(len(vector) != width for vector in dense_vectors):
        raise ValueError("Dense vectors must have the same dimensionality.")

    return tuple(
        sum(vector[index] for vector in dense_vectors) / len(dense_vectors)
        for index in range(width)
    )
