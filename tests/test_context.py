"""Importing context.py must not load the embedding model; the vector-store helpers load it on demand."""
import importlib
import sys
import types
import unittest
from unittest import mock

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class _FakeEmbeddings:
    """Stand-in for HuggingFaceEmbeddings that counts model loads instead of downloading one."""
    loads = 0

    def __init__(self, model_name: str):
        type(self).loads += 1
        self.model_name = model_name


def _fake_module(name: str, **attributes) -> types.ModuleType:
    module = types.ModuleType(name)
    vars(module).update(attributes)
    return module


class ContextLazyLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Swap in fakes for the heavy packages (no torch, no model download, no Chroma database),
        # then import a fresh context.py against them. sys.modules is restored after the class.
        cls._chroma = mock.MagicMock(name="Chroma")
        fakes = {
            "langchain_huggingface": _fake_module("langchain_huggingface", HuggingFaceEmbeddings=_FakeEmbeddings),
            "langchain_chroma": _fake_module("langchain_chroma", Chroma=cls._chroma),
        }
        modules = mock.patch.dict(sys.modules, fakes)
        modules.start()
        cls.addClassCleanup(modules.stop)
        sys.modules.pop("context", None)
        _FakeEmbeddings.loads = 0
        cls.context = importlib.import_module("context")
        cls.loads_during_import = _FakeEmbeddings.loads

    def setUp(self):
        _FakeEmbeddings.loads = 0
        self._chroma.reset_mock()

    def test_import_does_not_load_embedding_model(self):
        self.assertEqual(self.loads_during_import, 0)

    def test_get_vector_store_uses_default_model(self):
        self.context.get_vector_store()
        kwargs = self._chroma.call_args.kwargs
        self.assertEqual(kwargs["embedding_function"].model_name, DEFAULT_MODEL)
        self.assertEqual(kwargs["persist_directory"], "./chromadb")

    def test_create_vector_store_uses_default_model(self):
        self.context.create_vector_store([])
        kwargs = self._chroma.from_documents.call_args.kwargs
        self.assertEqual(kwargs["embedding"].model_name, DEFAULT_MODEL)
        self.assertEqual(kwargs["documents"], [])

    def test_explicit_model_is_used_without_loading_another(self):
        my_model = object()
        self.context.get_vector_store(embedding_model=my_model)
        self.assertIs(self._chroma.call_args.kwargs["embedding_function"], my_model)
        self.assertEqual(_FakeEmbeddings.loads, 0)


if __name__ == "__main__":
    unittest.main()
