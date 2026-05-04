"""Smoke tests for the Router class.

Run with: python -m unittest tests.test_router

Covers the three constructors (from_examples, from_descriptions, from_pretrained),
the save/load roundtrip for both dense (v0-style) and sparse (v1-desc-style)
centroids, route() shape contracts, and constructor validation.

Encoder/hybrid backend tests are skipped unless sentence-transformers is
installed, since they require ~250 MB of optional deps.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from agent_tool_router import Router
from agent_tool_router.router import RouteResult


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
