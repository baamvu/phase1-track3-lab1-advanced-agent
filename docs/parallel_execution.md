# Parallel Execution

## Vấn đề: Chạy tuần tự quá chậm

Tuần tự (workers=1):

```
React 1/100   ████████████████████ 18s
React 2/100   ████████████████████ 18s
React 3/100   ████████████████████ 18s
...
React 100/100 ████████████████████ 18s
                                      = 100 × 18s = 30 phút (React)
                                      + 100 × 18s = 30 phút (Reflexion)
                                      = 60 phút tổng
```

## Giải pháp: Chạy song song với ThreadPoolExecutor

Song song (workers=8):

```
Worker 1: React hp1   ████████████████████ 18s
Worker 2: React hp2   ████████████████████ 18s
Worker 3: React hp3   ████████████████████ 18s
Worker 4: React hp4   ████████████████████ 18s
Worker 5: React hp5   ████████████████████ 18s
Worker 6: React hp6   ████████████████████ 18s
Worker 7: React hp7   ████████████████████ 18s
Worker 8: React hp8   ████████████████████ 18s
                      ↓ xong batch 1, chạy batch 2...
Worker 1: React hp9   ████████████████████ 18s
Worker 2: React hp10  ████████████████████ 18s
...
                                      = ⌈100/8⌉ × 18s ≈ 3.75 phút (React)
                                      + ⌈100/8⌉ × 18s ≈ 3.75 phút (Reflexion)
                                      = 7.5 phút tổng
```

## Code flow

```
run_benchmark.py
    │
    ├─ mode="llm" && workers > 1 ?
    │   │
    │   YES → _run_parallel(react, examples, "React", 8)
    │   │         │
    │   │         ├─ ThreadPoolExecutor(max_workers=8)
    │   │         │     │
    │   │         │     ├─ Worker 1: agent.run(hp1) → LLM call → RunRecord
    │   │         │     ├─ Worker 2: agent.run(hp2) → LLM call → RunRecord
    │   │         │     ├─ Worker 3: agent.run(hp3) → LLM call → RunRecord
    │   │         │     ├─ ...
    │   │         │     └─ Worker 8: agent.run(hp8) → LLM call → RunRecord
    │   │         │
    │   │         ├─ as_completed() → nhận kết quả khi xong
    │   │         └─ records[idx] = result  (giữ đúng thứ tự)
    │   │
    │   └─ _run_parallel(reflexion, examples, "Reflexion", 8)
    │         └─ (tương tự)
    │
    NO → chạy tuần tự như cũ
```

## Tại sao thread-safe?

```python
# Mỗi worker chạy agent.run(example) độc lập:
def run(self, example: QAExample) -> RunRecord:
    reflection_memory: list[str] = []   # ← local variable, mỗi thread riêng
    reflections: list[ReflectionEntry] = []  # ← local variable
    traces: list[AttemptTrace] = []          # ← local variable
    ...
```

- `agent` object chỉ đọc (không có state thay đổi)
- `example` là Pydantic model immutable
- Mỗi `run()` tạo local variables riêng → không race condition
- `openai.OpenAI` client thread-safe (theo docs chính thức)

## So sánh tốc độ

| Workers | 100 mẫu React | 100 mẫu Reflexion | Tổng    |
| ------- | ------------- | ----------------- | ------- |
| 1 (tuần tự) | 30 phút    | 30 phút           | 60 phút |
| 4       | 7.5 phút      | 7.5 phút          | 15 phút |
| 8       | 3.75 phút     | 3.75 phút         | 7.5 phút |
| 16      | 1.9 phút      | 1.9 phút          | 3.8 phút |

## Lệnh chạy

```bash
python run_benchmark.py --dataset data/hotpot_100.json --mode llm --workers 8 --out-dir outputs/llm_run
```

Rubric không bị ảnh hưởng — output vẫn là `RunRecord` + `ReportPayload` giống hệt, chỉ nhanh hơn.
