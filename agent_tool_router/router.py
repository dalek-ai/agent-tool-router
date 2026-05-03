"""Centroid-retrieval router.

A trained Router holds three artifacts:
  - vec      : a fitted sklearn TfidfVectorizer
  - centroids: ndarray [V, n_features], L2-normalized, one row per tool
  - vocab    : list[str], tool names aligned with centroids rows

Scoring is one sparse-times-dense matmul. No GPU, no torch, no LLM call —
this is the boring baseline that the rest of the project has to beat.

For the description-based constructor, an optional bi-encoder backend is
available behind the ``[encoder]`` extras: pass ``backend="encoder"`` to
score by sentence-transformer cosine, or ``backend="hybrid"`` for a linear
combination ``alpha * cos_tfidf + (1 - alpha) * cos_encoder``. This requires
``pip install agent-tool-router[encoder]`` (pulls in torch, ~250 MB).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np

# Where bundled pretrained models live, relative to this file.
_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parent
_BUILTIN_MODELS_DIR = _REPO_ROOT / "models"

DEFAULT_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class RouteResult:
    tool: str
    score: float


class Router:
    def __init__(
        self,
        vec=None,
        centroids: Optional[np.ndarray] = None,
        vocab: Optional[list[str]] = None,
        *,
        encoder_model=None,
        encoder_centroids: Optional[np.ndarray] = None,
        alpha: float = 1.0,
        backend: str = "tfidf",
    ):
        if vocab is None:
            raise ValueError("Router requires a vocab list.")
        if backend not in ("tfidf", "encoder", "hybrid"):
            raise ValueError(
                f"backend must be one of 'tfidf', 'encoder', 'hybrid'; got {backend!r}"
            )
        if backend in ("tfidf", "hybrid") and (vec is None or centroids is None):
            raise ValueError(f"backend={backend!r} requires vec and centroids.")
        if backend in ("encoder", "hybrid") and (
            encoder_model is None or encoder_centroids is None
        ):
            raise ValueError(
                f"backend={backend!r} requires encoder_model and encoder_centroids."
            )
        self.vec = vec
        self.centroids = centroids
        self.vocab = list(vocab)
        self._name_to_idx = {n: i for i, n in enumerate(self.vocab)}
        self.encoder_model = encoder_model
        self.encoder_centroids = encoder_centroids
        self.alpha = float(alpha)
        self.backend = backend

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

    @classmethod
    def from_examples(
        cls,
        examples: Iterable[tuple[str, Iterable[str]]],
        *,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
        sublinear_tf: bool = True,
        max_features: int = 50000,
    ) -> "Router":
        """Build a Router in memory from a list of (task, tools) pairs.

        Use this when you already know your tool set and have a handful of
        example tasks per tool. No persistence, no train/test split — just a
        same-shape centroid retriever as the pretrained models, fit on your
        own data.

        >>> r = Router.from_examples([
        ...     ("search the web for recent news on X", ["web_search"]),
        ...     ("compute 2+2", ["calculator"]),
        ... ])
        >>> r.route("look up tomorrow's weather online", k=1)
        ['web_search']
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        tasks: list[str] = []
        tool_lists: list[list[str]] = []
        for task, tools in examples:
            t = (task or "").strip()
            tl = [n for n in tools if isinstance(n, str) and n]
            if not t or not tl:
                continue
            tasks.append(t)
            tool_lists.append(tl)
        if not tasks:
            raise ValueError("from_examples: no usable (task, tools) pairs.")

        vocab = sorted({n for tl in tool_lists for n in tl})
        if not vocab:
            raise ValueError("from_examples: no tool names found.")
        name_to_idx = {n: i for i, n in enumerate(vocab)}

        # min_df clamps to corpus size to avoid silently dropping all features
        # on tiny example sets — this constructor is meant for tiny example sets.
        effective_min_df = max(1, min(min_df, len(tasks)))
        vec = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=effective_min_df,
            sublinear_tf=sublinear_tf,
            lowercase=True,
        )
        X = vec.fit_transform(tasks)
        n_feat = X.shape[1]

        V = len(vocab)
        centroids = np.zeros((V, n_feat), dtype=np.float64)
        counts = np.zeros(V, dtype=np.int64)
        for row_i, tl in enumerate(tool_lists):
            row = X[row_i].toarray().ravel()
            for tool in set(tl):
                tidx = name_to_idx[tool]
                centroids[tidx] += row
                counts[tidx] += 1
        nonzero = counts > 0
        centroids[nonzero] /= counts[nonzero, None]
        centroids = normalize(centroids, axis=1)

        return cls(vec=vec, centroids=centroids, vocab=vocab)

    @classmethod
    def from_descriptions(
        cls,
        descriptions: Iterable[tuple[str, str]],
        *,
        backend: str = "tfidf",
        alpha: float = 0.5,
        encoder_model_name: str = DEFAULT_ENCODER_MODEL,
        include_name: bool = True,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
        sublinear_tf: bool = True,
        max_features: int = 50000,
    ) -> "Router":
        """Build a Router in memory from (tool_name, description) pairs.

        Useful when you have OpenAI-style function specs but no example tasks.
        Each tool's "centroid" is the TF-IDF vector of its description (plus
        the name's subtokens by default), so routing is
        cosine(task, description). This is the same scoring rule that gave
        73.5% top-3 cross-source on Hermes held-out in our LOSO eval (see
        router/eval/baseline_loso_descriptions.py); for tools whose
        descriptions are domain-specific outliers, expect weaker results
        (tau-bench's 23 customer-service tools scored 1.5x random).

        Backends:
          - "tfidf"   : default, no extra dependency. Wins when task and tool
                        descriptions share lexical surface.
          - "encoder" : sentence-transformer cosine. Wins when descriptions
                        paraphrase the task semantically. Requires
                        ``pip install agent-tool-router[encoder]``.
          - "hybrid"  : linear combination
                        ``alpha * cos_tfidf + (1 - alpha) * cos_encoder``.
                        Pareto-dominates both backends with alpha=0.5 on 2/3
                        held-out sources in our LOSO eval. Same extras
                        requirement as ``encoder``.

        Practical caveat (tfidf only): TF-IDF needs a reasonable vocabulary to
        discriminate. With <50 tools and short descriptions the vocab is too
        thin and ranks will be noisy. For small tool sets prefer
        Router.from_examples() with a handful of (task, [tools]) pairs per
        tool, OR switch to backend="encoder" / "hybrid".
        """
        if backend not in ("tfidf", "encoder", "hybrid"):
            raise ValueError(
                f"backend must be one of 'tfidf', 'encoder', 'hybrid'; got {backend!r}"
            )

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        import re

        def _tool_doc(name: str, desc: str) -> str:
            desc = (desc or "").strip()
            if not include_name:
                return desc
            parts = re.split(r"[_\.\s\-]+", name or "")
            subs: list[str] = []
            for p in parts:
                subs.extend(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", p))
            name_text = " ".join(t.lower() for t in subs if len(t) >= 2)
            return f"{desc} {name_text}".strip()

        names: list[str] = []
        docs: list[str] = []
        seen: set[str] = set()
        for name, desc in descriptions:
            if not isinstance(name, str) or not name or name in seen:
                continue
            doc = _tool_doc(name, desc or "")
            if not doc:
                continue
            seen.add(name)
            names.append(name)
            docs.append(doc)
        if not names:
            raise ValueError(
                "from_descriptions: no usable (name, description) pairs."
            )

        vec = None
        centroids = None
        if backend in ("tfidf", "hybrid"):
            effective_min_df = max(1, min(min_df, len(docs)))
            vec = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=effective_min_df,
                sublinear_tf=sublinear_tf,
                lowercase=True,
            )
            X = vec.fit_transform(docs)
            centroids = normalize(X, axis=1).toarray().astype(np.float64)

        encoder_model = None
        encoder_centroids = None
        if backend in ("encoder", "hybrid"):
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    f"backend={backend!r} needs sentence-transformers. "
                    f"Install with: pip install agent-tool-router[encoder]"
                ) from e
            encoder_model = SentenceTransformer(encoder_model_name)
            encoder_centroids = encoder_model.encode(
                docs, batch_size=64, show_progress_bar=False,
                normalize_embeddings=True, convert_to_numpy=True,
            ).astype(np.float32, copy=False)

        return cls(
            vec=vec,
            centroids=centroids,
            vocab=names,
            encoder_model=encoder_model,
            encoder_centroids=encoder_centroids,
            alpha=alpha,
            backend=backend,
        )

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

        from sklearn.preprocessing import normalize

        scores_tfidf = None
        scores_enc = None
        if self.backend in ("tfidf", "hybrid"):
            X = self.vec.transform(tasks)
            Xn = normalize(X, axis=1)
            scores_tfidf = np.asarray(Xn @ self.centroids.T)
        if self.backend in ("encoder", "hybrid"):
            task_enc = self.encoder_model.encode(
                tasks, batch_size=64, show_progress_bar=False,
                normalize_embeddings=True, convert_to_numpy=True,
            ).astype(np.float32, copy=False)
            scores_enc = task_enc @ self.encoder_centroids.T

        if self.backend == "tfidf":
            scores = scores_tfidf
        elif self.backend == "encoder":
            scores = scores_enc
        else:
            scores = self.alpha * scores_tfidf + (1.0 - self.alpha) * scores_enc

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
