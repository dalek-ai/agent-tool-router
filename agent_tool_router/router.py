"""Centroid-retrieval router.

A trained Router holds three artifacts:
  - vec      : a fitted sklearn TfidfVectorizer
  - centroids: ndarray [V, n_features], L2-normalized, one row per tool
  - vocab    : list[str], tool names aligned with centroids rows

Scoring is one sparse-times-dense matmul. No GPU, no torch, no LLM call —
this is the boring baseline that the rest of the project has to beat.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np

# Where bundled pretrained models live, relative to this file.
_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parent
_BUILTIN_MODELS_DIR = _REPO_ROOT / "models"


@dataclass
class RouteResult:
    tool: str
    score: float


class Router:
    def __init__(self, vec, centroids: np.ndarray, vocab: list[str]):
        self.vec = vec
        self.centroids = centroids
        self.vocab = list(vocab)
        self._name_to_idx = {n: i for i, n in enumerate(self.vocab)}

    @classmethod
    def from_pretrained(cls, name_or_path: str) -> "Router":
        """Load a model by name (looked up in <repo>/models/<name>/) or by
        absolute/relative directory path."""
        candidate = Path(name_or_path)
        if not candidate.is_absolute():
            built_in = _BUILTIN_MODELS_DIR / name_or_path
            if built_in.exists():
                candidate = built_in
            else:
                candidate = Path.cwd() / name_or_path
        if not candidate.exists():
            raise FileNotFoundError(
                f"No model at {candidate}. Train one with "
                f"`python -m agent_tool_router.train --out models/<name>`."
            )
        vec = joblib.load(candidate / "vectorizer.joblib")
        centroids = np.load(candidate / "centroids.npy")
        vocab = (candidate / "vocab.txt").read_text(encoding="utf-8").splitlines()
        return cls(vec=vec, centroids=centroids, vocab=vocab)

    def route(
        self,
        task: str | Iterable[str],
        k: int = 3,
        return_scores: bool = False,
    ) -> list[str] | list[RouteResult] | list[list[str]] | list[list[RouteResult]]:
        """Return top-k tool names for one task or a batch of tasks.

        - If `task` is a str: returns list[str] (or list[RouteResult] when
          return_scores=True), of length up to k.
        - If `task` is an iterable of str: returns list-of-list, same shape.
        """
        single = isinstance(task, str)
        tasks = [task] if single else list(task)
        if not tasks:
            return []
        X = self.vec.transform(tasks)
        # Cosine = (normalized X) @ centroids.T  (centroids already normalized).
        from sklearn.preprocessing import normalize

        Xn = normalize(X, axis=1)
        scores = np.asarray(Xn @ self.centroids.T)
        ranked = np.argsort(-scores, axis=1)
        out = []
        for row_i in range(scores.shape[0]):
            idxs = ranked[row_i, :k].tolist()
            if return_scores:
                out.append(
                    [
                        RouteResult(self.vocab[i], float(scores[row_i, i]))
                        for i in idxs
                    ]
                )
            else:
                out.append([self.vocab[i] for i in idxs])
        return out[0] if single else out

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vec, path / "vectorizer.joblib")
        np.save(path / "centroids.npy", self.centroids)
        (path / "vocab.txt").write_text("\n".join(self.vocab), encoding="utf-8")
        return path
