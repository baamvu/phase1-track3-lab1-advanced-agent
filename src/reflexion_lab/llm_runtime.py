from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer

load_dotenv()

_clients: list[OpenAI] = []
_lock = threading.Lock()
_index = 0


def _get_clients() -> list[OpenAI]:
    global _clients
    if not _clients:
        default_url = os.environ.get("MIMO_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1")
        keys_str = os.environ.get("MIMO_API_KEYS", os.environ.get("MIMO_API_KEY", ""))
        entries = [e.strip() for e in keys_str.split(",") if e.strip()]
        for entry in entries:
            if "@" in entry:
                key, url = entry.split("@", 1)
                _clients.append(OpenAI(base_url=url.strip(), api_key=key.strip()))
            else:
                _clients.append(OpenAI(base_url=default_url, api_key=entry))
    return _clients


def _get_client() -> OpenAI:
    global _index
    clients = _get_clients()
    with _lock:
        client = clients[_index % len(clients)]
        _index += 1
    return client


def _chat(system: str, user: str, temperature: float = 0.0, max_retries: int = 20) -> tuple[str, dict[str, int]]:
    import random
    from rich import print as rprint
    client = _get_client()
    model = os.environ.get("MIMO_MODEL", "mimo-v2-omni")
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = resp.choices[0].message.content or ""
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                "total_tokens": resp.usage.total_tokens if resp.usage else 0,
            }
            return text, usage
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = min(5 * (2 ** attempt), 60) + random.uniform(1, 5)
                rprint(f"[yellow]429 rate limit, retry {attempt+1}/{max_retries} in {wait:.1f}s[/yellow]")
                time.sleep(wait)
                continue
            raise


def _parse_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON found in LLM response: {text[:200]}")


def _build_context(example: QAExample) -> str:
    parts = []
    for i, chunk in enumerate(example.context, 1):
        parts.append(f"[{i}] {chunk.title}: {chunk.text}")
    return "\n\n".join(parts)


def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> tuple[str, int, int]:
    user_msg = f"Question: {example.question}\n\nContext:\n{_build_context(example)}"
    if reflection_memory:
        notes = "\n".join(f"- {m}" for m in reflection_memory)
        user_msg += f"\n\nReflection notes from previous attempts:\n{notes}"
    t0 = time.perf_counter()
    text, usage = _chat(ACTOR_SYSTEM, user_msg)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    match = re.search(r"ANSWER:\s*(.+)", text, re.IGNORECASE)
    answer = match.group(1).strip() if match else text.strip().split("\n")[-1].strip()
    return answer, usage["total_tokens"], latency_ms


def evaluator(
    example: QAExample,
    answer: str,
) -> tuple[JudgeResult, int, int]:
    user_msg = (
        f"Question: {example.question}\n"
        f"Gold answer: {example.gold_answer}\n"
        f"Predicted answer: {answer}"
    )
    t0 = time.perf_counter()
    text, usage = _chat(EVALUATOR_SYSTEM, user_msg)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    try:
        data = _parse_json(text)
        result = JudgeResult(
            score=int(data.get("score", 0)),
            reason=data.get("reason", ""),
            missing_evidence=data.get("missing_evidence", []),
            spurious_claims=data.get("spurious_claims", []),
        )
    except Exception:
        is_correct = normalize_answer(example.gold_answer) == normalize_answer(answer)
        result = JudgeResult(
            score=1 if is_correct else 0,
            reason="Fallback: matched by normalize_answer" if is_correct else "Fallback: LLM parse error, answer mismatch",
        )
    return result, usage["total_tokens"], latency_ms


def reflector(
    example: QAExample,
    attempt_id: int,
    judge: JudgeResult,
) -> tuple[ReflectionEntry, int, int]:
    user_msg = (
        f"Question: {example.question}\n"
        f"Gold answer: {example.gold_answer}\n"
        f"Agent's wrong answer: (hidden for brevity)\n"
        f"Evaluator feedback: {judge.reason}\n"
        f"Missing evidence: {', '.join(judge.missing_evidence) if judge.missing_evidence else 'N/A'}\n"
        f"Spurious claims: {', '.join(judge.spurious_claims) if judge.spurious_claims else 'N/A'}"
    )
    t0 = time.perf_counter()
    text, usage = _chat(REFLECTOR_SYSTEM, user_msg)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    try:
        data = _parse_json(text)
        entry = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=data.get("failure_reason", judge.reason),
            lesson=data.get("lesson", ""),
            next_strategy=data.get("next_strategy", ""),
        )
    except Exception:
        entry = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="Reflection parse failed; review the question carefully.",
            next_strategy="Re-read context passages and verify each hop before answering.",
        )
    return entry, usage["total_tokens"], latency_ms
