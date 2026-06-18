from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer
from rich import print
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.schemas import QAExample, RunRecord
from src.reflexion_lab.utils import load_dataset, save_jsonl

app = typer.Typer(add_completion=False)


def _run_parallel(agent, examples: list[QAExample], label: str, workers: int) -> list[RunRecord]:
    records: list[RunRecord] = [None] * len(examples)  # type: ignore[list-item]
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(f"[cyan]{label}[/cyan]", total=len(examples))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(agent.run, ex): i for i, ex in enumerate(examples)}
            for future in as_completed(futures):
                idx = futures[future]
                records[idx] = future.result()
                progress.advance(task)
    return records


@app.command()
def main(
    dataset: str = "data/hotpot_100.json",
    out_dir: str = "outputs/sample_run",
    reflexion_attempts: int = 3,
    mode: str = typer.Option("mock", help="Runtime mode: 'mock' or 'llm'"),
    workers: int = typer.Option(8, help="Parallel workers for LLM mode"),
) -> None:
    examples = load_dataset(dataset)
    react = ReActAgent(mode=mode)
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts, mode=mode)

    if mode == "llm" and workers > 1:
        react_records = _run_parallel(react, examples, "React", workers)
        reflexion_records = _run_parallel(reflexion, examples, "Reflexion", workers)
    else:
        react_records = []
        for i, example in enumerate(examples):
            print(f"[cyan]React[/cyan] {i+1}/{len(examples)}: {example.qid}")
            react_records.append(react.run(example))
        reflexion_records = []
        for i, example in enumerate(examples):
            print(f"[magenta]Reflexion[/magenta] {i+1}/{len(examples)}: {example.qid}")
            reflexion_records.append(reflexion.run(example))

    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)

    report = build_report(all_records, dataset_name=Path(dataset).name, mode=mode)
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))


if __name__ == "__main__":
    app()
