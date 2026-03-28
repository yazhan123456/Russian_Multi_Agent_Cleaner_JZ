from __future__ import annotations

import asyncio
import base64
import http.client
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class RequestMeta:
    attempts: int
    retries_used: int


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: int = 180,
    retries: int = 3,
    retry_delay: float = 1.5,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            **headers,
            "Content-Type": "application/json",
            "User-Agent": "Russian-Data-Cleaning-Agent/1.0",
        },
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw_bytes = response.read()
                if not raw_bytes:
                    raise RuntimeError("empty response body")
                return json.loads(raw_bytes.decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - networked path
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} calling {url}: {body[:800]}") from exc
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.IncompleteRead,
            http.client.HTTPException,
            ssl.SSLError,
            json.JSONDecodeError,
            OSError,
            RuntimeError,
        ) as exc:  # pragma: no cover - networked path
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_delay * attempt)
    raise RuntimeError(f"Network error calling {url}: {last_error}") from last_error


async def post_json_async(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: int = 180,
    retries: int = 3,
    retry_delay: float = 1.5,
) -> tuple[dict[str, Any], RequestMeta]:
    request_headers = {
        **headers,
        "Content-Type": "application/json",
        "User-Agent": "Russian-Data-Cleaning-Agent/1.0",
    }
    last_error: Exception | None = None
    async with httpx.AsyncClient(timeout=timeout, headers=request_headers) as client:
        for attempt in range(1, retries + 1):
            try:
                response = await client.post(url, json=payload)
                if response.status_code >= 400:
                    body = response.text
                    retryable = response.status_code in {429, 500, 503}
                    if retryable and attempt < retries:
                        last_error = RuntimeError(f"HTTP {response.status_code} calling {url}: {body[:800]}")
                        await asyncio.sleep(retry_delay * (2 ** (attempt - 1)))
                        continue
                    raise RuntimeError(f"HTTP {response.status_code} calling {url}: {body[:800]}")
                raw_bytes = response.content
                if not raw_bytes:
                    raise RuntimeError("empty response body")
                return (
                    json.loads(raw_bytes.decode("utf-8")),
                    RequestMeta(attempts=attempt, retries_used=attempt - 1),
                )
            except (httpx.TimeoutException, httpx.TransportError, json.JSONDecodeError, OSError, RuntimeError) as exc:
                last_error = exc
                if attempt >= retries:
                    break
                await asyncio.sleep(retry_delay * (2 ** (attempt - 1)))
    raise RuntimeError(f"Network error calling {url}: {last_error}") from last_error


def normalize_base_url(base_url: str) -> str:
    return base_url[:-1] if base_url.endswith("/") else base_url


def extract_openai_message_text(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    return str(content or "")


def openai_compatible_chat_completion(
    *,
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: int = 180,
) -> dict[str, Any]:
    url = f"{normalize_base_url(base_url)}/chat/completions"
    return post_json(url, payload, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)


def deepseek_chat_completion(
    *,
    api_key: str,
    payload: dict[str, Any],
    timeout: int = 180,
) -> dict[str, Any]:
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    return openai_compatible_chat_completion(base_url=base_url, api_key=api_key, payload=payload, timeout=timeout)


async def deepseek_chat_completion_async(
    *,
    api_key: str,
    payload: dict[str, Any],
    timeout: int = 180,
    retries: int = 3,
    retry_delay: float = 1.5,
) -> tuple[dict[str, Any], RequestMeta]:
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    url = f"{normalize_base_url(base_url)}/chat/completions"
    return await post_json_async(
        url,
        payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
    )


def qwen_compatible_chat_completion(
    *,
    api_key: str,
    payload: dict[str, Any],
    timeout: int = 180,
) -> dict[str, Any]:
    base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    return openai_compatible_chat_completion(base_url=base_url, api_key=api_key, payload=payload, timeout=timeout)


def qwen_ocr_via_dashscope(
    *,
    api_key: str,
    png_bytes: bytes,
    model: str = "qwen-vl-ocr-latest",
    task: str = "multi_lan",
    timeout: int = 180,
) -> str:
    endpoint = os.getenv(
        "DASHSCOPE_OCR_URL",
        "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
    )
    encoded = base64.b64encode(png_bytes).decode("ascii")
    payload = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "image": f"data:image/png;base64,{encoded}",
                            "min_pixels": 3072,
                            "max_pixels": 8388608,
                            "enable_rotate": False,
                        }
                    ],
                }
            ]
        },
        "parameters": {
            "ocr_options": {
                "task": task,
            }
        },
    }
    response = post_json(endpoint, payload, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
    output = response.get("output", {})
    choices = output.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", [])
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                return str(item["text"])
    return ""
