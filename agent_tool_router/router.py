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
import scipy.sparse as sp

# Where bundled pretrained models live, relative to this file.
_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parent
_BUILTIN_MODELS_DIR = _REPO_ROOT / "models"

# Default HuggingFace org for short-name pretrained lookups.
_HF_DEFAULT_ORG = "dalek-ai"

DEFAULT_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _hf_snapshot_download(repo_id: str) -> Path:
    """Download a pretrained model from HuggingFace Hub and return the
    local snapshot path. Raises a clear error if `huggingface_hub` is
    missing.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Pretrained model not found locally and `huggingface_hub` is "
            "not installed. Install it with `pip install huggingface_hub`, "
            "or train the model locally with "
            "`python -m agent_tool_router.train_descriptions --out models/<name>`."
        ) from exc
    return Path(snapshot_download(repo_id=repo_id))


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
        encoder_model_name: Optional[str] = None,
        alpha: float = 1.0,
        backend: str = "tfidf",
        markov_counts: Optional["sp.csr_matrix"] = None,
        markov_vocab: Optional[list[str]] = None,
        markov_alpha: float = 0.4,
        markov_rerank_n: int = 50,
    ):
        if vocab is None:
            raise ValueError("Router requires a vocab list.")
        if backend not in ("tfidf", "encoder", "hybrid"):
            raise ValueError(
                f"backend must be one of 'tfidf', 'encoder', 'hybrid'; got {backend!r}"
            )
        if backend in ("tfidf", "hybrid") and (vec is None or centroids is None):
            raise ValueError(f"backend={backend!r} requires vec and centroids.")
        if backend in ("encoder", "hybrid") and encoder_centroids is None:
            raise ValueError(
                f"backend={backend!r} requires encoder_centroids."
            )
        self.vec = vec
        self.centroids = centroids
        self.vocab = list(vocab)
        self._name_to_idx = {n: i for i, n in enumerate(self.vocab)}
        self.encoder_model = encoder_model
        self.encoder_centroids = encoder_centroids
        self.encoder_model_name = encoder_model_name
        self.alpha = float(alpha)
        self.backend = backend
        self.markov_counts = markov_counts
        self.markov_vocab = list(markov_vocab) if markov_vocab is not None else None
        self.markov_alpha = float(markov_alpha)
        self.markov_rerank_n = int(markov_rerank_n)
        if self.markov_counts is not None and self.markov_vocab is not None:
            self._markov_idx = {n: i for i, n in enumerate(self.markov_vocab)}
            # Pre-compute row sums for the smoothing denominator.
            self._markov_totals = np.asarray(
                self.markov_counts.sum(axis=1)
            ).ravel().astype(np.int64)
            self._markov_V = len(self.markov_vocab)
        else:
            self._markov_idx = None
            self._markov_totals = None
            self._markov_V = 0

    @classmethod
    def from_pretrained(cls, name_or_path: str) -> "Router":
        """Load a model by name or by absolute/relative directory path.

        Lookup order for a short name like ``"baseline-v1-desc-hybrid"``:

        1. ``<repo>/models/<name>/`` (when running from a clone that has
           trained models locally),
        2. ``<cwd>/<name>/``,
        3. ``huggingface.co/dalek-ai/<name>`` (downloaded and cached on
           first use, requires ``huggingface_hub``).

        For an explicit ``"user/repo"`` argument, an existing local path
        wins; otherwise the value is treated as a HuggingFace repo id.
        Absolute paths are always treated as local-only.

        Model directories may contain encoder centroids
        (``encoder_centroids.npy``) and a ``config.json`` setting the
        default backend; those are loaded automatically when present. The
        encoder model itself is lazy-loaded on the first route() call.
        """
        candidate = Path(name_or_path)
        if candidate.is_absolute():
            if not candidate.exists():
                raise FileNotFoundError(f"No model at {candidate}.")
        elif "/" in name_or_path:
            # Treat as path first; fall back to HF repo id.
            if candidate.exists():
                pass
            else:
                candidate = _hf_snapshot_download(name_or_path)
        else:
            built_in = _BUILTIN_MODELS_DIR / name_or_path
            cwd_local = Path.cwd() / name_or_path
            if built_in.exists():
                candidate = built_in
            elif cwd_local.exists():
                candidate = cwd_local
            else:
                candidate = _hf_snapshot_download(f"{_HF_DEFAULT_ORG}/{name_or_path}")

        config_path = candidate / "config.json"
        if config_path.exists():
            import json as _json
            cfg = _json.loads(config_path.read_text(encoding="utf-8"))
        else:
            cfg = {}
        backend = cfg.get("backend", "tfidf")
        alpha = float(cfg.get("alpha", 0.5))
        encoder_model_name = cfg.get("encoder_model_name")
        markov_alpha = float(cfg.get("markov_alpha", 0.4))
        markov_rerank_n = int(cfg.get("markov_rerank_n", 50))

        vec = None
        centroids = None
        vec_path = candidate / "vectorizer.joblib"
        if vec_path.exists():
            vec = joblib.load(vec_path)
        sparse_path = candidate / "centroids.npz"
        dense_path = candidate / "centroids.npy"
        if sparse_path.exists():
            centroids = sp.load_npz(sparse_path)
        elif dense_path.exists():
            centroids = np.load(dense_path)
        if backend in ("tfidf", "hybrid") and (vec is None or centroids is None):
            raise FileNotFoundError(
                f"backend={backend!r} model at {candidate} is missing "
                f"vectorizer.joblib or centroids.{{npz,npy}}."
            )

        encoder_centroids = None
        enc_path = candidate / "encoder_centroids.npy"
        if enc_path.exists():
            encoder_centroids = np.load(enc_path)
        if backend in ("encoder", "hybrid") and encoder_centroids is None:
            raise FileNotFoundError(
                f"backend={backend!r} model at {candidate} is missing "
                f"encoder_centroids.npy."
            )

        vocab = (candidate / "vocab.txt").read_text(encoding="utf-8").splitlines()

        markov_counts = None
        markov_vocab = None
        markov_path = candidate / "markov_counts.npz"
        markov_vocab_path = candidate / "markov_vocab.txt"
        if markov_path.exists() and markov_vocab_path.exists():
            markov_counts = sp.load_npz(markov_path).tocsr()
            markov_vocab = markov_vocab_path.read_text(
                encoding="utf-8"
            ).splitlines()

        return cls(
            vec=vec,
            centroids=centroids,
            vocab=vocab,
            encoder_centroids=encoder_centroids,
            encoder_model_name=encoder_model_name,
            alpha=alpha,
            backend=backend,
            markov_counts=markov_counts,
            markov_vocab=markov_vocab,
            markov_alpha=markov_alpha,
            markov_rerank_n=markov_rerank_n,
        )

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
            centroids = normalize(X, axis=1).tocsr()

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
            encoder_model_name=encoder_model_name if backend != "tfidf" else None,
            alpha=alpha,
            backend=backend,
        )

    def route(
        self,
        task: str | Iterable[str],
        k: int = 3,
        return_scores: bool = False,
        history: Optional[list[str] | Iterable[list[str]]] = None,
        markov_alpha: Optional[float] = None,
    ) -> list[str] | list[RouteResult] | list[list[str]] | list[list[RouteResult]]:
        """Return top-k tool names for one task or a batch of tasks.

        - If `task` is a str: returns list[str] (or list[RouteResult] when
          return_scores=True), of length up to k.
        - If `task` is an iterable of str: returns list-of-list, same shape.

        Optional ``history`` enables history-aware rerank: pass the list of
        tool names already called in the current trace (per task, if batch).
        When the model directory ships a Markov-1 transition table
        (baseline-v1-desc-hybrid does), the top-``markov_rerank_n``
        retrieval candidates are rescored with
        ``markov_alpha * retrieval_norm + (1 - markov_alpha) * markov_norm``.
        Pass ``markov_alpha`` to override the model's default
        (0.4 on baseline-v1-desc-hybrid, the sweep-best on n=2094 test).

        Measured lift on baseline-v1-desc-hybrid (n=2094 held-out triplets):
        top-1 13.8% → 34.6% (+20.8pp), top-3 32.7% → 48.0% (+15.3pp).
        Reproduce with router/eval/eval_next_tool_markov.py.
        """
        single = isinstance(task, str)
        tasks = [task] if single else list(task)
        if not tasks:
            return []

        if history is None:
            histories: list[list[str]] = [[] for _ in tasks]
        elif single:
            histories = [list(history) if history is not None else []]
        else:
            histories = [list(h) if h is not None else [] for h in history]
            if len(histories) != len(tasks):
                raise ValueError(
                    f"history must align with tasks: got {len(histories)} histories "
                    f"for {len(tasks)} tasks."
                )
        use_markov = (
            any(h for h in histories)
            and self.markov_counts is not None
            and self._markov_V > 0
        )
        alpha_mk = self.markov_alpha if markov_alpha is None else float(markov_alpha)

        from sklearn.preprocessing import normalize

        scores_tfidf = None
        scores_enc = None
        if self.backend in ("tfidf", "hybrid"):
            X = self.vec.transform(tasks)
            Xn = normalize(X, axis=1)
            product = Xn @ self.centroids.T
            scores_tfidf = (
                product.toarray() if sp.issparse(product) else np.asarray(product)
            )
        if self.backend in ("encoder", "hybrid"):
            if self.encoder_model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as e:
                    raise ImportError(
                        f"backend={self.backend!r} needs sentence-transformers. "
                        f"Install with: pip install agent-tool-router[encoder]"
                    ) from e
                model_name = self.encoder_model_name or DEFAULT_ENCODER_MODEL
                self.encoder_model = SentenceTransformer(model_name)
            task_enc = self.encoder_model.encode(
                tasks, batch_size=64, show_progress_bar=False,
                normalize_embeddings=True, convert_to_numpy=True,
            ).astype(np.float32, copy=False)
            # numpy 2.x emits spurious divide/overflow/invalid warnings from
            # this matmul on float32 inputs even when the result is finite.
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                scores_enc = task_enc @ self.encoder_centroids.T

        if self.backend == "tfidf":
            scores = scores_tfidf
        elif self.backend == "encoder":
            scores = scores_enc
        else:
            scores = self.alpha * scores_tfidf + (1.0 - self.alpha) * scores_enc

        out = []
        for row_i in range(scores.shape[0]):
            row_scores = scores[row_i]
            h = histories[row_i]
            if use_markov and h:
                # Retrieve top-N candidates, rerank with Markov-1 prior.
                n_cand = min(self.markov_rerank_n, row_scores.shape[0])
                cand_idx = np.argpartition(-row_scores, n_cand - 1)[:n_cand]
                cand_idx = cand_idx[np.argsort(-row_scores[cand_idx])]
                ret_scores = row_scores[cand_idx].astype(np.float64)
                ret_norm = (ret_scores - ret_scores.min()) / (
                    ret_scores.max() - ret_scores.min() + 1e-9
                )
                mk_scores = self._markov_probs(
                    h[-1], [self.vocab[i] for i in cand_idx]
                )
                mk_norm = (mk_scores - mk_scores.min()) / (
                    mk_scores.max() - mk_scores.min() + 1e-9
                )
                final = alpha_mk * ret_norm + (1.0 - alpha_mk) * mk_norm
                order = np.argsort(-final)
                idxs = [int(cand_idx[j]) for j in order[:k]]
                if return_scores:
                    out.append(
                        [
                            RouteResult(self.vocab[i], float(final[order[j]]))
                            for j, i in enumerate(idxs)
                        ]
                    )
                else:
                    out.append([self.vocab[i] for i in idxs])
            else:
                ranked = np.argsort(-row_scores)[:k]
                if return_scores:
                    out.append(
                        [
                            RouteResult(self.vocab[i], float(row_scores[i]))
                            for i in ranked.tolist()
                        ]
                    )
                else:
                    out.append([self.vocab[i] for i in ranked.tolist()])
        return out[0] if single else out

    def _markov_probs(self, prev: str, candidates: list[str]) -> np.ndarray:
        """Add-one-smoothed P(candidate | prev) over the Markov vocab."""
        prev_n = (prev or "").strip().lower()
        V = self._markov_V
        prev_idx = self._markov_idx.get(prev_n) if self._markov_idx else None
        if prev_idx is None:
            return np.full(len(candidates), 1.0 / max(V, 1), dtype=np.float64)
        row = self.markov_counts.getrow(prev_idx)
        total = float(self._markov_totals[prev_idx])
        denom = total + V
        # Pull only the columns we need from the sparse row.
        row_dense = row.toarray().ravel()
        probs = np.empty(len(candidates), dtype=np.float64)
        for i, cand in enumerate(candidates):
            cand_n = (cand or "").strip().lower()
            j = self._markov_idx.get(cand_n)
            count = float(row_dense[j]) if j is not None else 0.0
            probs[i] = (count + 1.0) / denom
        return probs

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.vec is not None and self.centroids is not None:
            joblib.dump(self.vec, path / "vectorizer.joblib")
            if sp.issparse(self.centroids):
                sp.save_npz(path / "centroids.npz", self.centroids.tocsr())
            else:
                np.save(path / "centroids.npy", self.centroids)
        if self.encoder_centroids is not None:
            np.save(
                path / "encoder_centroids.npy",
                np.asarray(self.encoder_centroids, dtype=np.float32),
            )
        (path / "vocab.txt").write_text("\n".join(self.vocab), encoding="utf-8")
        config = {
            "backend": self.backend,
            "alpha": self.alpha,
        }
        if self.encoder_model_name:
            config["encoder_model_name"] = self.encoder_model_name
        if self.markov_counts is not None and self.markov_vocab is not None:
            sp.save_npz(
                path / "markov_counts.npz", self.markov_counts.tocsr()
            )
            (path / "markov_vocab.txt").write_text(
                "\n".join(self.markov_vocab), encoding="utf-8",
            )
            config["markov_alpha"] = self.markov_alpha
            config["markov_rerank_n"] = self.markov_rerank_n
        import json as _json
        (path / "config.json").write_text(
            _json.dumps(config, indent=2) + "\n", encoding="utf-8",
        )
        return path
