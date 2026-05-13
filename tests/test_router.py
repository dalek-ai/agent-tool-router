"""Smoke tests for the Router class.

Run with: python -m unittest tests.test_router

Covers the three constructors (from_examples, from_descriptions, from_pretrained),
the save/load roundtrip for both dense (v0-style) and sparse (v1-desc-style)
centroids, route() shape contracts, and constructor validation.

Encoder/hybrid backend tests are skipped unless sentence-transformers is
installed, since they require ~250 MB of optional deps.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from agent_tool_router import Router
from agent_tool_router.router import RouteResult


def _has_sentence_transformers() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


class TestFromExamples(unittest.TestCase):
    def test_basic_routing(self):
        r = Router.from_examples([
            ("search the web for recent news", ["web_search"]),
            ("compute 12 plus 34", ["calculator"]),
            ("cancel order 12345", ["cancel_order"]),
            ("what's the weather tomorrow", ["web_search"]),
            ("multiply 7 by 8", ["calculator"]),
        ])
        self.assertEqual(set(r.vocab), {"web_search", "calculator", "cancel_order"})
        self.assertEqual(r.backend, "tfidf")
        self.assertIsInstance(r.centroids, np.ndarray)

        top1 = r.route("look up tomorrow's weather online", k=1)
        self.assertEqual(top1, ["web_search"])

        top3 = r.route("compute 5 times 9", k=3)
        self.assertEqual(top3[0], "calculator")

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            Router.from_examples([])
        with self.assertRaises(ValueError):
            Router.from_examples([("", []), ("ok", [])])


class TestFromDescriptions(unittest.TestCase):
    def test_tfidf_sparse_centroids(self):
        r = Router.from_descriptions([
            ("web_search", "Search the public web for a query and return links"),
            ("calculator", "Evaluate arithmetic expressions and return numeric results"),
            ("cancel_order", "Cancel a customer's pending order and refund the payment"),
        ])
        self.assertEqual(r.backend, "tfidf")
        self.assertTrue(sp.issparse(r.centroids))
        self.assertEqual(r.centroids.shape[0], 3)

        top1 = r.route("please cancel my order", k=1)
        self.assertEqual(top1, ["cancel_order"])

    def test_dedup_on_name(self):
        r = Router.from_descriptions([
            ("foo", "first description"),
            ("foo", "duplicate description"),
            ("bar", "another tool entirely"),
        ])
        self.assertEqual(r.vocab, ["foo", "bar"])

    def test_invalid_backend_raises(self):
        with self.assertRaises(ValueError):
            Router.from_descriptions(
                [("foo", "a tool")], backend="not_a_backend"
            )

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            Router.from_descriptions([])
        with self.assertRaises(ValueError):
            # With include_name=False, ("foo", "") yields an empty doc and
            # is dropped, so all-empty input has no usable rows.
            Router.from_descriptions(
                [("", ""), ("foo", "")], include_name=False
            )


class TestRoutingShapes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = Router.from_examples([
            ("search the web", ["web_search"]),
            ("compute math", ["calculator"]),
            ("cancel order", ["cancel_order"]),
        ])

    def test_str_input_returns_list_of_str(self):
        out = self.r.route("compute 1 plus 1", k=2)
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)
        self.assertTrue(all(isinstance(x, str) for x in out))

    def test_iterable_input_returns_list_of_lists(self):
        out = self.r.route(["search the web", "compute 2 plus 2"], k=2)
        self.assertEqual(len(out), 2)
        for sub in out:
            self.assertIsInstance(sub, list)
            self.assertEqual(len(sub), 2)

    def test_return_scores(self):
        out = self.r.route("compute 1 plus 1", k=2, return_scores=True)
        self.assertEqual(len(out), 2)
        for r in out:
            self.assertIsInstance(r, RouteResult)
            self.assertIsInstance(r.tool, str)
            self.assertIsInstance(r.score, float)
        self.assertGreaterEqual(out[0].score, out[1].score)


class TestSaveLoadRoundtrip(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="atr_test_"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_dense_centroids_roundtrip(self):
        r = Router.from_examples([
            ("search the web", ["web_search"]),
            ("compute math", ["calculator"]),
        ])
        r.save(self.tmpdir / "model")
        self.assertTrue((self.tmpdir / "model" / "centroids.npy").exists())
        r2 = Router.from_pretrained(str(self.tmpdir / "model"))
        self.assertEqual(r2.vocab, r.vocab)
        self.assertEqual(r2.route("search for python tutorials", k=1),
                         r.route("search for python tutorials", k=1))

    def test_sparse_centroids_roundtrip(self):
        r = Router.from_descriptions([
            ("web_search", "Search the public web for a query"),
            ("calculator", "Evaluate arithmetic expressions"),
            ("cancel_order", "Cancel a pending customer order"),
        ])
        r.save(self.tmpdir / "model")
        self.assertTrue((self.tmpdir / "model" / "centroids.npz").exists())
        self.assertFalse((self.tmpdir / "model" / "centroids.npy").exists())
        r2 = Router.from_pretrained(str(self.tmpdir / "model"))
        self.assertTrue(sp.issparse(r2.centroids))
        self.assertEqual(r2.vocab, r.vocab)
        self.assertEqual(r2.route("cancel my order please", k=1),
                         r.route("cancel my order please", k=1))

    def test_config_json_written_for_tfidf(self):
        r = Router.from_descriptions([
            ("web_search", "Search the web"),
            ("calculator", "Evaluate math"),
        ])
        r.save(self.tmpdir / "model")
        cfg_path = self.tmpdir / "model" / "config.json"
        self.assertTrue(cfg_path.exists())
        cfg = json.loads(cfg_path.read_text())
        self.assertEqual(cfg["backend"], "tfidf")
        self.assertNotIn("encoder_model_name", cfg)
        self.assertFalse(
            (self.tmpdir / "model" / "encoder_centroids.npy").exists()
        )

    @unittest.skipUnless(
        _has_sentence_transformers(),
        "sentence-transformers not installed",
    )
    def test_hybrid_roundtrip_with_lazy_load(self):
        r = Router.from_descriptions(
            [
                ("web_search", "Search the public web for a query"),
                ("calculator", "Evaluate arithmetic expressions"),
                ("cancel_order", "Cancel a pending customer order"),
            ],
            backend="hybrid",
            alpha=0.5,
        )
        r.save(self.tmpdir / "model")
        self.assertTrue(
            (self.tmpdir / "model" / "encoder_centroids.npy").exists()
        )
        cfg = json.loads(
            (self.tmpdir / "model" / "config.json").read_text()
        )
        self.assertEqual(cfg["backend"], "hybrid")
        self.assertEqual(cfg["alpha"], 0.5)
        self.assertIn("encoder_model_name", cfg)

        r2 = Router.from_pretrained(str(self.tmpdir / "model"))
        self.assertEqual(r2.backend, "hybrid")
        self.assertEqual(r2.vocab, r.vocab)
        # Lazy-load: encoder model isn't instantiated until first route().
        self.assertIsNone(r2.encoder_model)
        out = r2.route("cancel my order please", k=1)
        self.assertEqual(out, ["cancel_order"])
        self.assertIsNotNone(r2.encoder_model)

    @unittest.skipUnless(
        _has_sentence_transformers(),
        "sentence-transformers not installed",
    )
    def test_encoder_only_roundtrip(self):
        r = Router.from_descriptions(
            [
                ("web_search", "Search the public web for a query"),
                ("calculator", "Evaluate arithmetic expressions"),
                ("cancel_order", "Cancel a pending customer order"),
            ],
            backend="encoder",
        )
        r.save(self.tmpdir / "model")
        # Encoder-only model has no tfidf artifacts.
        self.assertFalse(
            (self.tmpdir / "model" / "centroids.npz").exists()
        )
        self.assertFalse(
            (self.tmpdir / "model" / "centroids.npy").exists()
        )
        self.assertFalse(
            (self.tmpdir / "model" / "vectorizer.joblib").exists()
        )
        self.assertTrue(
            (self.tmpdir / "model" / "encoder_centroids.npy").exists()
        )

        r2 = Router.from_pretrained(str(self.tmpdir / "model"))
        self.assertEqual(r2.backend, "encoder")
        out = r2.route("cancel my order please", k=1)
        self.assertEqual(out, ["cancel_order"])


class TestHistoryAwareRerank(unittest.TestCase):
    """Markov-1 rerank when history is passed and a table is loaded."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="atr_hist_"))
        self.r = Router.from_descriptions([
            ("lookup_order", "Look up a customer order by id"),
            ("cancel_order", "Cancel a pending customer order"),
            ("refund_order", "Issue a refund for a canceled order"),
            ("modify_order", "Change line items on a pending order"),
            ("search_flights", "Search for flights between airports"),
        ])
        # Synthetic Markov table: lookup_order is followed by cancel_order
        # 9 times, by modify_order once. Other priors are flat.
        names = ["cancel_order", "lookup_order", "modify_order",
                 "refund_order", "search_flights"]
        idx = {n: i for i, n in enumerate(names)}
        rows, cols, data = [], [], []
        for nxt, c in [("cancel_order", 9), ("modify_order", 1)]:
            rows.append(idx["lookup_order"])
            cols.append(idx[nxt])
            data.append(c)
        mat = sp.csr_matrix(
            (np.array(data, dtype=np.int32),
             (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
            shape=(len(names), len(names)),
        )
        self.r.markov_counts = mat
        self.r.markov_vocab = names
        self.r._markov_idx = idx
        self.r._markov_totals = np.asarray(mat.sum(axis=1)).ravel().astype(np.int64)
        self.r._markov_V = len(names)
        self.r.markov_alpha = 0.4
        self.r.markov_rerank_n = 5

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_no_history_matches_retrieval_only(self):
        # Without history, the rerank path is skipped.
        baseline = self.r.route("refund my canceled order", k=3)
        self.assertEqual(baseline, self.r.route(
            "refund my canceled order", k=3, history=None,
        ))

    def test_empty_history_matches_retrieval_only(self):
        baseline = self.r.route("refund my canceled order", k=3)
        self.assertEqual(baseline, self.r.route(
            "refund my canceled order", k=3, history=[],
        ))

    def test_history_reorders_candidates(self):
        # A query mentioning both "modify" and "cancel" leaves ambiguity in
        # the retrieval scores. With history=[lookup_order], the Markov
        # prior strongly favors cancel_order (9 obs vs 1 for modify_order),
        # so it should be promoted ahead of modify_order.
        with_history = self.r.route(
            "act on my pending order", k=5, history=["lookup_order"],
        )
        # cancel_order should rank ahead of modify_order under the prior.
        self.assertLess(
            with_history.index("cancel_order"),
            with_history.index("modify_order"),
        )

    def test_unknown_prev_falls_back_to_uniform(self):
        # An unseen prev tool yields uniform Markov scores, so the final
        # ranking is driven by the retrieval signal alone.
        baseline = self.r.route("cancel my order", k=3)
        with_unseen = self.r.route(
            "cancel my order", k=3, history=["never_seen_tool"],
        )
        self.assertEqual(baseline, with_unseen)

    def test_save_load_roundtrip_preserves_markov(self):
        self.r.save(self.tmpdir / "model")
        self.assertTrue((self.tmpdir / "model" / "markov_counts.npz").exists())
        self.assertTrue((self.tmpdir / "model" / "markov_vocab.txt").exists())
        r2 = Router.from_pretrained(str(self.tmpdir / "model"))
        self.assertIsNotNone(r2.markov_counts)
        self.assertEqual(r2.markov_vocab, self.r.markov_vocab)
        self.assertEqual(r2.markov_alpha, 0.4)
        out = r2.route("act on my pending order", k=5, history=["lookup_order"])
        self.assertLess(out.index("cancel_order"), out.index("modify_order"))


class TestMarkov2Backoff(unittest.TestCase):
    """Markov-2 history-bigram rerank with stupid backoff to Markov-1."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="atr_m2_"))
        self.r = Router.from_descriptions([
            ("lookup_order", "Look up a customer order by id"),
            ("cancel_order", "Cancel a pending customer order"),
            ("refund_order", "Issue a refund for a canceled order"),
            ("modify_order", "Change line items on a pending order"),
            ("exchange_item", "Exchange a delivered item for a different one"),
            ("search_flights", "Search for flights between airports"),
        ])
        names = ["cancel_order", "exchange_item", "lookup_order", "modify_order",
                 "refund_order", "search_flights"]
        idx = {n: i for i, n in enumerate(names)}
        # Markov-1: lookup_order -> modify_order(9), exchange_item(1).
        rows, cols, data = [], [], []
        for nxt, c in [("modify_order", 9), ("exchange_item", 1)]:
            rows.append(idx["lookup_order"])
            cols.append(idx[nxt])
            data.append(c)
        mat1 = sp.csr_matrix(
            (np.array(data, dtype=np.int32),
             (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
            shape=(len(names), len(names)),
        )
        # Markov-2: (cancel_order, lookup_order) -> exchange_item(8), modify(0).
        # When the user's last 2 tools are [cancel_order, lookup_order], the
        # bigram says exchange_item, even though the unigram says modify.
        bigram_keys = np.array(
            [[idx["cancel_order"], idx["lookup_order"]]], dtype=np.int32,
        )
        mat2 = sp.csr_matrix(
            (np.array([8], dtype=np.int32),
             (np.array([0], dtype=np.int32),
              np.array([idx["exchange_item"]], dtype=np.int32))),
            shape=(1, len(names)),
        )
        self.r.markov_counts = mat1
        self.r.markov_vocab = names
        self.r._markov_idx = idx
        self.r._markov_totals = np.asarray(mat1.sum(axis=1)).ravel().astype(np.int64)
        self.r._markov_V = len(names)
        # Keep the prior strong so the unit test exercises the bigram-vs-unigram
        # decision rather than retrieval blending (we cover blending in
        # TestHistoryAwareRerank).
        self.r.markov_alpha = 0.05
        self.r.markov_rerank_n = 6
        self.r.markov2_counts = mat2
        self.r.markov2_keys = bigram_keys
        self.r._markov2_keymap = {
            (int(bigram_keys[0, 0]), int(bigram_keys[0, 1])): 0,
        }
        self.r._markov2_totals = np.asarray(mat2.sum(axis=1)).ravel().astype(np.int64)
        self.r.markov2_lambda = 0.4

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_bigram_overrides_unigram(self):
        """The Markov-2 bigram favors exchange_item; Markov-1 alone would pick
        modify_order. With a 2-step history hitting the bigram, exchange_item
        must rank ahead of modify_order."""
        out = self.r.route(
            "act on my pending order", k=6,
            history=["cancel_order", "lookup_order"],
        )
        self.assertLess(out.index("exchange_item"), out.index("modify_order"))

    def test_unseen_bigram_backs_off_to_unigram(self):
        """A 2-step history whose (p2,p1) isn't in the bigram table must
        produce the same ranking as Markov-1 alone with the same p1."""
        with_m2 = self.r.route(
            "act on my pending order", k=6,
            history=["never_seen_tool", "lookup_order"],
        )
        with_m1 = self.r.route(
            "act on my pending order", k=6,
            history=["lookup_order"],
        )
        self.assertEqual(with_m2, with_m1)

    def test_short_history_uses_markov1(self):
        """A 1-step history must use the Markov-1 path even though Markov-2
        is loaded."""
        with_m1 = self.r.route(
            "act on my pending order", k=6,
            history=["lookup_order"],
        )
        # modify_order should win (9 obs vs 1 for exchange_item under M1).
        self.assertLess(with_m1.index("modify_order"), with_m1.index("exchange_item"))

    def test_save_load_roundtrip(self):
        self.r.save(self.tmpdir / "model")
        self.assertTrue((self.tmpdir / "model" / "markov2_counts.npz").exists())
        self.assertTrue((self.tmpdir / "model" / "markov2_keys.npy").exists())
        r2 = Router.from_pretrained(str(self.tmpdir / "model"))
        self.assertIsNotNone(r2.markov2_counts)
        self.assertEqual(r2.markov2_lambda, 0.4)
        self.assertEqual(r2.markov2_keys.shape, (1, 2))
        out = r2.route(
            "act on my pending order", k=6,
            history=["cancel_order", "lookup_order"],
        )
        self.assertLess(out.index("exchange_item"), out.index("modify_order"))


class TestConstructorValidation(unittest.TestCase):
    def test_missing_vocab_raises(self):
        with self.assertRaises(ValueError):
            Router()

    def test_invalid_backend_raises(self):
        with self.assertRaises(ValueError):
            Router(vocab=["foo"], backend="nonsense")

    def test_tfidf_requires_vec_and_centroids(self):
        with self.assertRaises(ValueError):
            Router(vocab=["foo"], backend="tfidf")

    def test_encoder_requires_encoder_artifacts(self):
        with self.assertRaises(ValueError):
            Router(vocab=["foo"], backend="encoder")


class TestEncoderBackend(unittest.TestCase):
    def test_encoder_clear_error_when_missing(self):
        try:
            import sentence_transformers  # noqa: F401
            self.skipTest("sentence-transformers is installed; "
                          "this test only covers the missing-dep error.")
        except ImportError:
            pass
        with self.assertRaises(ImportError) as ctx:
            Router.from_descriptions(
                [("foo", "a tool")], backend="encoder"
            )
        self.assertIn("sentence-transformers", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
