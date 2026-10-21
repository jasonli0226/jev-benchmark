"""Command-line entry point: `uv run jev-bench <command>`."""

import argparse
import json
import logging
import sys
import tarfile
from datetime import datetime

from dotenv import load_dotenv

from jev_bench.config import RESULTS_DIR, ROOT, TASKS

log = logging.getLogger("jev_bench")

DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_JEV_PROMPTS = {
    "sst2": ["sst2_v1"],
    "agnews": ["agnews_v2"],
    "adult": [
        "adult_raw_v1",
        "adult_natural_v1",
        "adult_contextual_v1",
        "adult_natural_fewshot_v1",
        "adult_contextual_fewshot_v1",
    ],
}


def cmd_prepare(args) -> None:
    from jev_bench.data.splits import freeze

    for task in args.task:
        entry = freeze(task, overwrite=args.overwrite)
        log.info("%s: %s", task, {k: v["rows"] for k, v in entry.items()})


def _summary(result: dict) -> str:
    m = result["metrics"]
    return (
        f"{result['split']}/{result['task']}/{result['run']} seed={result['seed']} n={result['n']}: "
        f"acc={m['accuracy']:.4f} f1={m['macro_f1']:.4f} ece={m['ece']:.4f} "
        f"p50={result['latency']['single_item'].get('p50_ms', float('nan')):.1f}ms "
        f"$/1k={result['cost']['usd_per_1k_predictions']:.2e}"
    )


def cmd_run(args) -> None:
    from jev_bench.runner import run

    result = run(
        args.model,
        args.task,
        split=args.split,
        seed=args.seed,
        prompt_version=args.prompt,
        serialization=args.serialization,
        limit=args.limit,
        concurrency=args.concurrency,
    )
    log.info(_summary(result))
    if result["model"] == "jev":
        info = result["info"]
        log.info(
            "jev: %d requests (%d cached), parse failures %d, spent $%.5f",
            info["requests"],
            info["cached_requests"],
            info["parse_failures"],
            result["cost"]["spent_this_run_usd"],
        )


def _plan(args) -> list[dict]:
    from jev_bench.models.registry import LOCAL_TEXT_MODELS, TABULAR_MODELS

    jobs = []
    for task in args.task:
        if args.which in ("local", "all"):
            models = (TABULAR_MODELS + LOCAL_TEXT_MODELS) if TASKS[task].track == "B" else LOCAL_TEXT_MODELS
            jobs += [{"model_name": m, "task_name": task, "seed": s} for m in models for s in args.seeds]
        if args.which in ("jev", "all"):
            jobs += [
                {"model_name": "jev", "task_name": task, "seed": s, "prompt_version": p}
                for p in DEFAULT_JEV_PROMPTS[task]
                for s in args.seeds
            ]
    return jobs


def cmd_suite(args) -> None:
    from jev_bench.runner import result_exists, run

    jobs = _plan(args)
    failed = []
    for i, job in enumerate(jobs, 1):
        label = f"[{i}/{len(jobs)}] {json.dumps(job)}"
        if not args.rerun and result_exists(split=args.split, limit=args.limit, **job):
            log.info("%s already done, skipping", label)
            continue
        log.info("%s running", label)
        try:
            log.info(_summary(run(split=args.split, limit=args.limit, concurrency=args.concurrency, **job)))
        except Exception as exc:  # keep the suite going; report every failure at the end
            log.exception("%s failed: %s", label, exc)
            failed.append((label, repr(exc)))
    if failed:
        log.error("%d job(s) failed:\n%s", len(failed), "\n".join(f"  {l}: {e}" for l, e in failed))
        sys.exit(1)


def cmd_jev_check(args) -> None:
    """One cheap request (~$0.00002) to confirm the key and the response shape."""
    from jev_bench.jev.client import JevClient

    body = {
        "model": args.model,
        "state": {"review": "a gorgeous, witty, seductive movie."},
        "questions": {
            "answer": {
                "type": "choice",
                "instructions": "What is the sentiment of the review?",
                "criteria": {"negative": "Negative opinion.", "positive": "Positive opinion."},
            }
        },
    }
    res = JevClient(concurrency=1, use_cache=False).decide(body)
    print(json.dumps(res.body, indent=2))
    print(f"latency {res.latency_s * 1000:.0f} ms")


def cmd_report(args) -> None:
    from jev_bench.report.build import build_report

    path = build_report(split=args.split)
    log.info("report written to %s", path)


def cmd_recalibrate(args) -> None:
    from jev_bench.report.recalibrate import build_recalibration

    path = build_recalibration({t: DEFAULT_JEV_PROMPTS[t] for t in args.task})
    log.info("recalibration written to %s", path)


def cmd_confusion(args) -> None:
    from jev_bench.report.confusion import build_confusion

    figure, data = build_confusion()
    log.info("confusion matrices written to %s and %s", figure, data)


def cmd_export(args) -> None:
    """Bundle results (runs, predictions, report; not the API cache) for sharing."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ROOT / f"jev-bench-results-{stamp}.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        for sub in ("runs", "preds", "report"):
            path = RESULTS_DIR / sub
            if path.exists():
                tar.add(path, arcname=f"results/{sub}")
        tar.add(ROOT / "data" / "manifest.json", arcname="data/manifest.json")
        tar.add(ROOT / "configs" / "prompts", arcname="configs/prompts")
    log.info("wrote %s", out)


def main(argv=None) -> None:
    # .env is the source of truth for the key; a stale exported shell value must not win.
    load_dotenv(ROOT / ".env", override=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(prog="jev-bench")
    sub = parser.add_subparsers(dest="command", required=True)
    tasks = sorted(TASKS)

    p = sub.add_parser("prepare", help="download data and freeze splits")
    p.add_argument("--task", nargs="+", default=tasks, choices=tasks)
    p.add_argument("--overwrite", action="store_true", help="re-freeze (invalidates existing results)")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("run", help="one model on one task")
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True, choices=tasks)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt", help="Jev prompt version, e.g. sst2_v1")
    p.add_argument("--serialization", choices=["raw", "natural", "contextual"], help="Track B text models")
    p.add_argument("--limit", type=int, help="stratified subsample of the split (val iteration)")
    p.add_argument("--concurrency", type=int, default=8)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("suite", help="the full benchmark plan")
    p.add_argument("which", choices=["local", "jev", "all"])
    p.add_argument("--task", nargs="+", default=tasks, choices=tasks)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--limit", type=int)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--rerun", action="store_true", help="ignore existing results")
    p.set_defaults(func=cmd_suite)

    p = sub.add_parser("jev-check", help="one live request to verify key and response shape")
    p.add_argument("--model", default="typesafe/jev-1.14")
    p.set_defaults(func=cmd_jev_check)

    p = sub.add_parser("report", help="aggregate results into tables and plots")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("recalibrate", help="fit Platt/isotonic on cached Jev val preds, score on test")
    p.add_argument("--task", nargs="+", default=tasks, choices=tasks)
    p.set_defaults(func=cmd_recalibrate)

    p = sub.add_parser("confusion", help="AG News confusion matrices, Jev vs DistilBERT (test, seed 0)")
    p.set_defaults(func=cmd_confusion)

    p = sub.add_parser("export", help="tar results for sharing")
    p.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
