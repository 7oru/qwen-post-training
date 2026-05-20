"""Local OpenAI-compatible HTTP serving."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

from qwen_post_training.inference import DEFAULT_MODEL, GenerationRequest, generate


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    model: str = DEFAULT_MODEL
    adapter_path: Optional[str] = None
    backend: str = "mlx"
    max_tokens: int = 512
    temperature: float = 0.0
    seed: Optional[int] = None


def messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"messages[{index}].role must be system, user, or assistant")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"messages[{index}].content must be a non-empty string")
        parts.append(f"{role}: {content.strip()}")
    if not parts:
        raise ValueError("messages must be a non-empty list")
    parts.append("assistant:")
    return "\n".join(parts)


def parse_chat_request(body: dict[str, Any], config: ServerConfig) -> GenerationRequest:
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")

    model = body.get("model") or config.model
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")

    max_tokens = body.get("max_tokens", config.max_tokens)
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")

    temperature = body.get("temperature", config.temperature)
    if not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be numeric")

    seed = body.get("seed", config.seed)
    if seed is not None and not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    adapter_path = body.get("adapter_path", config.adapter_path)
    if adapter_path is not None and not isinstance(adapter_path, str):
        raise ValueError("adapter_path must be a string")

    return GenerationRequest(
        prompt=messages_to_prompt(messages),
        model=model,
        adapter_path=adapter_path,
        max_tokens=max_tokens,
        temperature=float(temperature),
        seed=seed,
        backend=config.backend,
    )


def build_chat_completion(body: dict[str, Any], config: ServerConfig) -> dict[str, Any]:
    request = parse_chat_request(body, config)
    content = generate(request)
    created = int(time.time())
    return {
        "id": f"chatcmpl-local-{created}",
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": None,
    }


def json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def make_handler(config: ServerConfig):
    class LocalQwenHandler(BaseHTTPRequestHandler):
        server_version = "qwenpt-local/0.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            data = json_bytes(payload)
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "model": config.model,
                        "adapter_path": config.adapter_path,
                        "backend": config.backend,
                    },
                )
                return
            if self.path == "/v1/models":
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": config.model,
                                "object": "model",
                                "owned_by": "local",
                            }
                        ],
                    },
                )
                return
            self.send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self.send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError("request body must be a JSON object")
                response = build_chat_completion(body, config)
            except json.JSONDecodeError as exc:
                self.send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": {"message": f"invalid JSON: {exc.msg}"}},
                )
                return
            except Exception as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(exc)}})
                return
            self.send_json(HTTPStatus.OK, response)

    return LocalQwenHandler


def run_server(config: ServerConfig) -> None:
    server = ThreadingHTTPServer((config.host, config.port), make_handler(config))
    try:
        print(f"Serving qwenpt on http://{config.host}:{config.port}")
        print("POST /v1/chat/completions")
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
