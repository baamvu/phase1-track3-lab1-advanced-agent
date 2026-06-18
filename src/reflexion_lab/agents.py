from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    mode: Literal["mock", "llm"] = "mock"

    def run(self, example: QAExample) -> RunRecord:
        if self.mode == "llm":
            return self._run_llm(example)
        return self._run_mock(example)

    def _run_mock(self, example: QAExample) -> RunRecord:
        from .mock_runtime import actor_answer, evaluator, reflector

        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        for attempt_id in range(1, self.max_attempts + 1):
            answer = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            judge = evaluator(example, answer)
            token_estimate = 320 + (attempt_id * 65) + (120 if self.agent_type == "reflexion" else 0)
            latency_ms = 160 + (attempt_id * 40) + (90 if self.agent_type == "reflexion" else 0)
            trace = AttemptTrace(attempt_id=attempt_id, answer=answer, score=judge.score, reason=judge.reason, token_estimate=token_estimate, latency_ms=latency_ms)
            final_answer = answer
            final_score = judge.score
            if judge.score == 1:
                traces.append(trace)
                break
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection = reflector(example, attempt_id, judge)
                reflections.append(reflection)
                reflection_memory.append(
                    f"[Attempt {attempt_id} failed] Reason: {reflection.failure_reason}. "
                    f"Lesson: {reflection.lesson}. Next strategy: {reflection.next_strategy}"
                )
            traces.append(trace)
        return self._build_record(example, final_answer, final_score, reflections, traces)

    def _run_llm(self, example: QAExample) -> RunRecord:
        from .llm_runtime import actor_answer, evaluator, reflector

        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        for attempt_id in range(1, self.max_attempts + 1):
            answer, actor_tokens, actor_latency = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            judge, eval_tokens, eval_latency = evaluator(example, answer)
            total_tokens = actor_tokens + eval_tokens
            total_latency = actor_latency + eval_latency
            trace = AttemptTrace(
                attempt_id=attempt_id, answer=answer, score=judge.score,
                reason=judge.reason, token_estimate=total_tokens, latency_ms=total_latency,
            )
            final_answer = answer
            final_score = judge.score
            if judge.score == 1:
                traces.append(trace)
                break
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection, ref_tokens, ref_latency = reflector(example, attempt_id, judge)
                reflections.append(reflection)
                trace.token_estimate += ref_tokens
                trace.latency_ms += ref_latency
                reflection_memory.append(
                    f"[Attempt {attempt_id} failed] Reason: {reflection.failure_reason}. "
                    f"Lesson: {reflection.lesson}. Next strategy: {reflection.next_strategy}"
                )
            traces.append(trace)
        return self._build_record(example, final_answer, final_score, reflections, traces)

    def _classify_failure(self, example: QAExample, answer: str, judge_reason: str) -> str:
        from .utils import normalize_answer
        norm_answer = normalize_answer(answer)
        norm_gold = normalize_answer(example.gold_answer)
        if not norm_answer or norm_answer in ("", "unknown", "n/a"):
            return "no_answer"
        if norm_answer in norm_gold or norm_gold in norm_answer:
            return "none"
        reason_lower = judge_reason.lower()
        if "contradict" in reason_lower or "opposite" in reason_lower:
            return "contradiction"
        if "vague" in reason_lower or "too general" in reason_lower or "not specific" in reason_lower:
            return "incomplete_answer"
        if "scope" in reason_lower or "broader" in reason_lower or "larger area" in reason_lower:
            return "wrong_scope"
        if "part of the question" in reason_lower or "only states" in reason_lower:
            return "incomplete_answer"
        return "wrong_final_answer"

    def _build_record(
        self,
        example: QAExample,
        final_answer: str,
        final_score: int,
        reflections: list[ReflectionEntry],
        traces: list[AttemptTrace],
    ) -> RunRecord:
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = "none" if final_score == 1 else self._classify_failure(example, final_answer, traces[-1].reason if traces else "")
        return RunRecord(
            qid=example.qid, question=example.question, gold_answer=example.gold_answer,
            agent_type=self.agent_type, predicted_answer=final_answer,
            is_correct=bool(final_score), attempts=len(traces),
            token_estimate=total_tokens, latency_ms=total_latency,
            failure_mode=failure_mode, reflections=reflections, traces=traces,
        )


class ReActAgent(BaseAgent):
    def __init__(self, mode: Literal["mock", "llm"] = "mock") -> None:
        super().__init__(agent_type="react", max_attempts=1, mode=mode)


class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3, mode: Literal["mock", "llm"] = "mock") -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts, mode=mode)
