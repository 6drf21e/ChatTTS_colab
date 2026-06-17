"""Integration tests for MiniMax LLM provider.

These tests call the actual MiniMax API and require MINIMAX_API_KEY env var.
Skip with: pytest -m "not integration"
"""
import json
import os
import sys
import unittest

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
MINIMAX_BASE_URL = "https://api.minimax.io/v1"


@pytest.mark.integration
@unittest.skipUnless(MINIMAX_API_KEY, "MINIMAX_API_KEY not set")
class TestMiniMaxIntegration(unittest.TestCase):
    """Integration tests that call the real MiniMax API."""

    def test_minimax_m27_script_generation(self):
        """Test that MiniMax-M2.7 can generate a script in JSON format."""
        from llm_utils import llm_operation
        from config import LLM_PROMPT

        short_text = "小明和小红在公园里散步，看到了一只可爱的小狗。"

        results = llm_operation(
            MINIMAX_BASE_URL,
            MINIMAX_API_KEY,
            "MiniMax-M2.7",
            LLM_PROMPT,
            short_text,
            required_keys=["txt", "character"],
        )

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        for item in results:
            self.assertIn("txt", item)
            self.assertIn("character", item)

    def test_minimax_m25_highspeed_basic_request(self):
        """Test basic chat completion with MiniMax-M2.5-highspeed."""
        from llm_utils import send_request
        import openai

        client = openai.OpenAI(api_key=MINIMAX_API_KEY, base_url=MINIMAX_BASE_URL)
        result = send_request(
            client,
            "请用一句话回答：",
            "天空是什么颜色的？",
            "MiniMax-M2.5-highspeed"
        )

        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        # Should not contain think tags after stripping
        self.assertNotIn("<think>", result)

    def test_minimax_m27_json_output(self):
        """Test that MiniMax-M2.7 can produce valid JSON output."""
        import openai
        from llm_utils import send_request, extract_json, clean_and_load_json

        client = openai.OpenAI(api_key=MINIMAX_API_KEY, base_url=MINIMAX_BASE_URL)
        prompt = '请返回一个JSON数组，格式为 [{"txt": "内容", "character": "角色"}]。只包含一个元素。'
        result = send_request(client, prompt, "你好世界", "MiniMax-M2.7")

        self.assertIsNotNone(result)
        json_str = extract_json(result)
        self.assertIsNotNone(json_str, f"Could not extract JSON from: {result}")
        parsed = clean_and_load_json(json_str)
        self.assertIsNotNone(parsed)


if __name__ == "__main__":
    unittest.main()
