from __future__ import annotations

import unittest

from qwen_post_training.server import (
    ServerConfig,
    build_chat_completion,
    messages_to_prompt,
    parse_chat_request,
)


class Phase5ServerTests(unittest.TestCase):
    def test_messages_to_prompt_keeps_roles(self) -> None:
        prompt = messages_to_prompt(
            [
                {"role": "system", "content": "You help."},
                {"role": "user", "content": "Hello"},
            ]
        )

        self.assertIn("system: You help.", prompt)
        self.assertIn("user: Hello", prompt)
        self.assertTrue(prompt.endswith("assistant:"))

    def test_parse_chat_request_uses_body_and_config(self) -> None:
        request = parse_chat_request(
            {
                "model": "local-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 32,
                "temperature": 0.2,
                "seed": 9,
                "adapter_path": "adapters/sft/run",
            },
            ServerConfig(backend="mock"),
        )

        self.assertEqual(request.model, "local-model")
        self.assertEqual(request.max_tokens, 32)
        self.assertEqual(request.temperature, 0.2)
        self.assertEqual(request.seed, 9)
        self.assertEqual(request.adapter_path, "adapters/sft/run")
        self.assertEqual(request.backend, "mock")

    def test_mock_chat_completion_matches_openai_shape(self) -> None:
        response = build_chat_completion(
            {
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 8,
            },
            ServerConfig(backend="mock"),
        )

        self.assertEqual(response["object"], "chat.completion")
        self.assertEqual(response["choices"][0]["message"]["role"], "assistant")
        self.assertIn("Say hi", response["choices"][0]["message"]["content"])

    def test_chat_completion_rejects_missing_messages(self) -> None:
        with self.assertRaisesRegex(ValueError, "messages must be a list"):
            build_chat_completion({}, ServerConfig(backend="mock"))


if __name__ == "__main__":
    unittest.main()
