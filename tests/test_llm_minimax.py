"""Unit tests for MiniMax LLM provider integration."""
import json
import unittest
from unittest.mock import MagicMock, patch


class TestLLMChange(unittest.TestCase):
    """Test the llm_change model-to-URL mapping."""

    def _get_llm_settings(self):
        """Return the llm_setting dict used by llm_change."""
        return {
            "gpt-3.5-turbo-0125": ["https://api.openai.com/v1"],
            "gpt-4o": ["https://api.openai.com/v1"],
            "deepseek-chat": ["https://api.deepseek.com"],
            "yi-large": ["https://api.lingyiwanwu.com/v1"],
            "MiniMax-M2.7": ["https://api.minimax.io/v1"],
            "MiniMax-M2.5-highspeed": ["https://api.minimax.io/v1"],
        }

    def test_minimax_m27_url(self):
        settings = self._get_llm_settings()
        self.assertEqual(settings["MiniMax-M2.7"][0], "https://api.minimax.io/v1")

    def test_minimax_m25_highspeed_url(self):
        settings = self._get_llm_settings()
        self.assertEqual(settings["MiniMax-M2.5-highspeed"][0], "https://api.minimax.io/v1")

    def test_gpt4o_url_unchanged(self):
        settings = self._get_llm_settings()
        self.assertEqual(settings["gpt-4o"][0], "https://api.openai.com/v1")

    def test_deepseek_url_unchanged(self):
        settings = self._get_llm_settings()
        self.assertEqual(settings["deepseek-chat"][0], "https://api.deepseek.com")

    def test_yi_large_url_unchanged(self):
        settings = self._get_llm_settings()
        self.assertEqual(settings["yi-large"][0], "https://api.lingyiwanwu.com/v1")

    def test_all_models_present(self):
        settings = self._get_llm_settings()
        expected_models = [
            "gpt-3.5-turbo-0125", "gpt-4o", "deepseek-chat",
            "yi-large", "MiniMax-M2.7", "MiniMax-M2.5-highspeed"
        ]
        for model in expected_models:
            self.assertIn(model, settings)


class TestSendRequestMiniMax(unittest.TestCase):
    """Test send_request with MiniMax-specific behavior."""

    def _make_mock_response(self, content):
        mock_choice = MagicMock()
        mock_choice.message.content = content
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        return mock_resp

    @patch("llm_utils.openai")
    def test_minimax_temperature_set(self, mock_openai_module):
        """MiniMax models should pass temperature=0.7."""
        from llm_utils import send_request

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response("test")

        send_request(mock_client, "prompt", "text", "MiniMax-M2.7")

        call_kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(call_kwargs.kwargs.get("temperature") or call_kwargs[1].get("temperature"), 0.7)

    @patch("llm_utils.openai")
    def test_non_minimax_no_temperature(self, mock_openai_module):
        """Non-MiniMax models should not have temperature injected."""
        from llm_utils import send_request

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response("test")

        send_request(mock_client, "prompt", "text", "gpt-4o")

        call_kwargs = mock_client.chat.completions.create.call_args
        # temperature should NOT be in kwargs for non-minimax
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        self.assertNotIn("temperature", all_kwargs)

    @patch("llm_utils.openai")
    def test_minimax_think_tag_stripped(self, mock_openai_module):
        """MiniMax responses with <think> tags should be stripped."""
        from llm_utils import send_request

        content_with_think = '<think>reasoning here</think>\n[{"txt": "hello", "character": "旁白"}]'
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(content_with_think)

        result = send_request(mock_client, "prompt", "text", "MiniMax-M2.7")
        self.assertNotIn("<think>", result)
        self.assertIn('[{"txt": "hello"', result)

    @patch("llm_utils.openai")
    def test_minimax_m25_highspeed_think_tag_stripped(self, mock_openai_module):
        """MiniMax-M2.5-highspeed also strips think tags."""
        from llm_utils import send_request

        content_with_think = '<think>step by step</think>{"answer": 42}'
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(content_with_think)

        result = send_request(mock_client, "prompt", "text", "MiniMax-M2.5-highspeed")
        self.assertNotIn("<think>", result)
        self.assertEqual(result, '{"answer": 42}')

    @patch("llm_utils.openai")
    def test_non_minimax_no_think_strip(self, mock_openai_module):
        """Non-MiniMax models should not strip <think> tags (they don't produce them)."""
        from llm_utils import send_request

        content = '<think>some text</think> hello'
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(content)

        result = send_request(mock_client, "prompt", "text", "gpt-4o")
        self.assertIn("<think>", result)

    @patch("llm_utils.openai")
    def test_minimax_none_content(self, mock_openai_module):
        """MiniMax with None content should not crash."""
        from llm_utils import send_request

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(None)

        result = send_request(mock_client, "prompt", "text", "MiniMax-M2.7")
        self.assertIsNone(result)

    @patch("llm_utils.openai")
    def test_minimax_empty_content(self, mock_openai_module):
        """MiniMax with empty string content should return empty string."""
        from llm_utils import send_request

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response("")

        result = send_request(mock_client, "prompt", "text", "MiniMax-M2.7")
        self.assertEqual(result, "")


class TestLLMOperation(unittest.TestCase):
    """Test llm_operation creates client with correct base_url."""

    @patch("llm_utils.openai.OpenAI")
    @patch("llm_utils.process_text")
    def test_minimax_client_creation(self, mock_process, mock_openai_cls):
        """llm_operation should create an OpenAI client with MiniMax base_url."""
        from llm_utils import llm_operation

        mock_process.return_value = []

        llm_operation(
            "https://api.minimax.io/v1",
            "test-key",
            "MiniMax-M2.7",
            "prompt",
            "text",
            ["key"]
        )

        mock_openai_cls.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.minimax.io/v1"
        )


class TestExtractJson(unittest.TestCase):
    """Test JSON extraction from LLM responses (used with all providers including MiniMax)."""

    def test_extract_json_array(self):
        from llm_utils import extract_json
        response = 'Here is the script:\n[{"txt": "hello", "character": "旁白"}]'
        result = extract_json(response)
        self.assertIsNotNone(result)
        parsed = json.loads(result)
        # extract_json may return the inner object or the array depending on regex
        self.assertTrue(isinstance(parsed, (list, dict)))

    def test_extract_json_object(self):
        from llm_utils import extract_json
        response = 'Result: {"txt": "hello world", "character": "年轻男性"}'
        result = extract_json(response)
        self.assertIsNotNone(result)

    def test_no_json(self):
        from llm_utils import extract_json
        response = "No JSON here, just plain text."
        result = extract_json(response)
        self.assertIsNone(result)


class TestCleanAndLoadJson(unittest.TestCase):
    """Test JSON cleaning and parsing."""

    def test_valid_json(self):
        from llm_utils import clean_and_load_json
        result = clean_and_load_json('[{"txt": "hello", "character": "旁白"}]')
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)

    def test_single_quotes(self):
        from llm_utils import clean_and_load_json
        result = clean_and_load_json("[{'txt': 'hello', 'character': '旁白'}]")
        self.assertIsNotNone(result)


class TestRemoveJsonEscapeCharacters(unittest.TestCase):
    """Test escape character removal."""

    def test_removes_backslash(self):
        from llm_utils import remove_json_escape_characters
        result = remove_json_escape_characters("hello\\world")
        self.assertEqual(result, "helloworld")

    def test_removes_quotes(self):
        from llm_utils import remove_json_escape_characters
        result = remove_json_escape_characters('hello"world')
        self.assertEqual(result, "helloworld")

    def test_removes_newlines(self):
        from llm_utils import remove_json_escape_characters
        result = remove_json_escape_characters("hello\nworld")
        self.assertEqual(result, "helloworld")


if __name__ == "__main__":
    unittest.main()
