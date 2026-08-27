"""Standalone unit tests for KacheDB SemanticCache without third-party test dependencies."""

import math
import unittest
from unittest.mock import MagicMock

from kachedb import KacheClient, SearchResult, SemanticCache
from kachedb.semantic.embedders import CallableAdapter, MockEmbedder


class TestMockEmbedder(unittest.TestCase):
    def test_dimension_and_normalization(self):
        embedder = MockEmbedder(dimension=128)
        vec = embedder.encode("How do I reset my password?")
        self.assertEqual(len(vec), 128)
        norm = math.sqrt(sum(x * x for x in vec))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_deterministic_vectors(self):
        embedder = MockEmbedder(dimension=64)
        v1 = embedder.encode("hello world")
        v2 = embedder.encode("hello world")
        self.assertEqual(v1, v2)

    def test_empty_string_vector(self):
        embedder = MockEmbedder(dimension=32)
        v = embedder.encode("")
        self.assertEqual(len(v), 32)
        norm = math.sqrt(sum(x * x for x in v))
        self.assertAlmostEqual(norm, 1.0, places=5)


class TestCallableAdapter(unittest.TestCase):
    def test_custom_callable(self):
        adapter = CallableAdapter(lambda s: [1.0, 0.0, 0.0])
        vec = adapter.encode("any text")
        self.assertEqual(vec, [1.0, 0.0, 0.0])


class TestSemanticCache(unittest.TestCase):
    def test_set_and_get_hit(self):
        mock_client = MagicMock(spec=KacheClient)
        mock_client.vadd.return_value = True
        mock_client.vsearch.return_value = [
            (b"How do I reset password?", 0.94, b"Go to Settings -> Reset Password")
        ]

        cache = SemanticCache(
            client=mock_client,
            index_name="faq",
            similarity_threshold=0.85,
            embedder=MockEmbedder(dimension=64),
        )

        stored = cache.set("How do I reset password?", "Go to Settings -> Reset Password")
        self.assertTrue(stored)
        self.assertTrue(mock_client.vadd.called)

        result = cache.get("Where to change my password?")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, SearchResult)
        self.assertEqual(result.value, "Go to Settings -> Reset Password")
        self.assertAlmostEqual(result.similarity, 0.94, places=2)
        self.assertEqual(str(result), "Go to Settings -> Reset Password")

    def test_get_miss(self):
        mock_client = MagicMock(spec=KacheClient)
        mock_client.vsearch.return_value = []

        cache = SemanticCache(
            client=mock_client,
            index_name="faq",
            similarity_threshold=0.85,
            embedder=MockEmbedder(dimension=64),
        )

        result = cache.get("Completely unrelated query about cooking recipes")
        self.assertIsNone(result)

    def test_delete_and_stats(self):
        mock_client = MagicMock(spec=KacheClient)
        mock_client.vdel.return_value = True
        mock_client.vstats.return_value = {"total_vectors": 10, "dimension": 64}

        cache = SemanticCache(
            client=mock_client,
            index_name="faq",
            embedder=MockEmbedder(dimension=64),
        )

        self.assertTrue(cache.delete("old prompt"))
        stats = cache.stats()
        self.assertEqual(stats["total_vectors"], 10)


if __name__ == "__main__":
    unittest.main()
