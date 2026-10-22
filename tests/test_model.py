"""model.py must take the OpenRouter key from your shell or the .env next to it, never from the code."""
import importlib.util
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_DIR = Path(__file__).resolve().parent.parent
KEY = "OPENROUTER_API_KEY"


class ModelApiKeyTests(unittest.TestCase):
    def setUp(self):
        # Work on a copy of model.py in a temp folder, so each test controls which .env sits
        # next to it and your real .env is never read or touched.
        self._dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self._dir, ignore_errors=True)
        shutil.copy(PROJECT_DIR / "model.py", self._dir / "model.py")

        # Snapshot the environment, restore it after the test, and start with neither key set.
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(KEY, None)
        os.environ.pop("OPENAI_API_KEY", None)

    def _import_model(self):
        """Import the temp copy of model.py as a fresh module, running its import-time code."""
        name = f"model_under_test_{self._testMethodName}"
        spec = importlib.util.spec_from_file_location(name, self._dir / "model.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        self.addCleanup(sys.modules.pop, name, None)
        spec.loader.exec_module(module)
        return module

    def _write_env_file(self, value: str):
        (self._dir / ".env").write_text(f"{KEY}={value}\n")

    def test_key_from_shell_is_not_overwritten(self):
        os.environ[KEY] = "from-shell"
        self._write_env_file("from-dotenv")
        self._import_model()
        self.assertEqual(os.environ[KEY], "from-shell")

    def test_key_is_read_from_env_file_next_to_model(self):
        self._write_env_file("from-dotenv")
        self._import_model()
        self.assertEqual(os.environ.get(KEY), "from-dotenv")

    def test_missing_key_fails_fast_instead_of_sending_openai_key(self):
        os.environ["OPENAI_API_KEY"] = "sk-openai-must-not-reach-openrouter"
        model = self._import_model()  # no .env in the temp folder
        with self.assertRaisesRegex(ValueError, KEY):
            model.ChatModel(model_name="any/model")

    def test_explicit_key_is_sent_to_openrouter(self):
        model = self._import_model()
        chat = model.ChatModel(model_name="any/model", openai_api_key="sk-or-test")
        key = chat.openai_api_key
        self.assertEqual(getattr(key, "get_secret_value", lambda: key)(), "sk-or-test")
        self.assertEqual(chat.openai_api_base, "https://openrouter.ai/api/v1")


if __name__ == "__main__":
    unittest.main()
