# Jev Benchmark

Benchmark comparing local ML models against **Jev** (TypeSafe's "System One" decision model, accessed via OpenRouter) on classification tasks.

## Motivation

Jev is a text-native structured decision model: it receives context + typed questions and returns typed answers with probabilities (~70–500ms, no reasoning trace). Real operational use cases (e.g. service-counter load prediction) are **structured/tabular**, which requires serializing numbers to text before Jev can consume them.

This benchmark answers two questions:

1. **How strong is Jev on its home turf** (text classification) vs classical/neural local models?
2. **How much of that advantage survives serialization** when the task is structured/tabular?

## Design

### Two task tracks

| Track | Task | Input to Jev | Local models input |
|---|---|---|---|
| **A — Text classification** | SST-2 / AG News | Raw text | Raw text (LR, DistilBERT) / TF-IDF or embeddings (trees) |
| **B — Structured decision** | UCI Adult (income >50k) | **Serialized number-to-text** | Native features (LR, LightGBM); text models read the `raw` serialization |

Track B uses **UCI Adult** — the classic open tabular benchmark (mixed numeric + categorical features), chosen for credibility and comparability with published reference numbers. Serialization variants for Jev are the study variable.

### Serialization variants (Track B, Jev only)

Number-to-text has no canonical form; each variant is a separate eval run:

1. **Raw dump** — `age=39, capital_gain=2174, ..., education=Bachelors, ...`
2. **Natural language** — "A 39-year-old male (White) ... Works in the State gov sector as Adm clerical, 40 hours per week..."
3. **Contextual framing** — natural language plus a `background` field in Jev's state ("about 24% of adults earn more than $50,000; higher income is associated with...")
4. **Few-shot** — natural or contextual, plus 20 labeled rows from the train split in `state.labeled_examples` (`examples:` block in the prompt config)

Templates live in `src/jev_bench/data/serialize.py`; framing text and few-shot settings live in the versioned prompt configs.

Accuracy deltas between variants are themselves a result: they quantify the hidden cost of using text-native models on structured prediction.

### Models under test

| # | Model | Type | Track |
|---|---|---|---|
| 0 | Tabular LR / Tabular LightGBM (native features) | classical | B (reference) |
| 1 | TF-IDF + Logistic Regression | classical, linear | A, B |
| 2 | TF-IDF + LightGBM | classical, non-linear | A, B |
| 3 | Embedding (sentence-transformers) + LightGBM | hybrid | A, B |
| 4 | DistilBERT fine-tune | local neural | A, B |
| 5 | **Jev** (`typesafe/jev-1.14` / `~typesafe/jev-latest`) | API decision model | A, B (serialized) |

### Metrics (three axes, not accuracy alone)

- **Accuracy / F1** on the test set
- **Latency** — p50/p95 per prediction (API roundtrip vs local inference)
- **Cost per 1k predictions** — Jev API spend vs local compute amortization
- **Calibration (ECE)** — all models output probabilities; Jev's confidence quality is a first-class question

## Eval Integrity Rules (non-negotiable)

1. **Frozen splits.** Train / val / test split at project start; test set is never used for training, hyperparameter search, or prompt iteration. Splits frozen in `data/` with a manifest (row counts + hashes).
2. **Prompt iteration only on val.** Jev prompt phrasing materially affects accuracy. All prompt/serialization iteration happens on val; prompts are frozen (versioned in `configs/prompts/`) before any test-set run.
3. **Frozen harness.** Eval harness version is recorded alongside every result; results from different harness versions are never compared.
4. **Multi-seed.** Trainable local models run with ≥3 seeds; report mean ± std, not the best seed.
5. **Interface alignment.** All models return probabilities over the same label set; the harness normalizes API and local outputs to one interface before scoring.
6. **Results are JSON, not notebook prints.** One JSON per run: `{model, track, seed, harness_version, prompt_version, metrics, latency, cost}`.

### Splits

| Task | Train | Val | Test | Notes |
|---|---|---|---|---|
| SST-2 | 65,349 | 2,000 | 872 | GLUE test labels are hidden, so the GLUE validation set is our test set and val is carved from train. Train contains sub-sentence phrases, so val accuracy runs higher than test. |
| AG News | 115,000 | 5,000 | 7,600 | Official test split. |
| Adult | 41,842 | 2,000 | 5,000 | OpenML v2 (train+test combined), stratified. `fnlwgt` and `education-num` are dropped for every model. |

Sources, licenses, column schemas and preprocessing details: [`data/README.md`](data/README.md).

## Repo Layout

```
jev-benchmark/
├── data/                  # frozen splits (parquet) + manifest.json (rows, hashes)
├── src/jev_bench/
│   ├── data/              # loaders, split freezing, Adult serialization
│   ├── models/            # each model: one class, one predict_proba() interface
│   ├── jev/               # Decisions API client, prompt configs -> requests/answers
│   ├── eval/              # metrics, calibration, latency stats
│   ├── report/            # tables, bootstrap CIs, figures
│   ├── runner.py          # one run -> results/runs/<split>/<task>/<run>__seed<n>.json
│   └── cli.py
├── configs/prompts/       # versioned Jev prompts (frozen before test)
├── results/               # runs/ (JSON), preds/ (per-item probabilities), report/, logs/; cache/ is not committed
├── reports/               # written report (Markdown) + its figures
├── tests/
└── README.md
```

## Running

Setup (uv, Python ≥ 3.11, CUDA GPU recommended for DistilBERT and embeddings):

```bash
uv sync
cp .env.example .env              # then put your OpenRouter key in .env
uv run pytest -q                  # harness unit tests, no network
uv run jev-bench jev-check        # one live Jev call (~$0.00002): confirms the key works
```

Data is already frozen in `data/`. Re-run `uv run jev-bench prepare --overwrite` only if you mean to invalidate every result.

1. **Local baselines** (42 runs, ≈45 min on an RTX 4090, no API cost):
   ```bash
   uv run jev-bench suite local
   ```
2. **Jev prompt iteration on val only** (cheap subsample; edit the YAML, re-run, compare):
   ```bash
   uv run jev-bench run --model jev --task sst2  --prompt sst2_v1  --split val --limit 300
   uv run jev-bench run --model jev --task adult --prompt adult_raw_v1 --split val --limit 300
   ```
   To change a prompt after you've used it, copy it to a new version (`sst2_v2.yaml`) rather than editing it in place.
3. **Freeze prompts.** Set `frozen: true` in each prompt you'll report on. If you changed versions, update `DEFAULT_JEV_PROMPTS` in `src/jev_bench/cli.py`. The harness refuses to run an unfrozen prompt on test.
4. **Jev on test** (21 runs: 7 prompts × 3 repeats; ≈35 min and ≈$4):
   ```bash
   uv run jev-bench suite jev --concurrency 16
   ```
5. **Report and export:**
   ```bash
   uv run jev-bench report           # results/report/test/summary.md + figures
   uv run jev-bench export           # jev-bench-results-<time>.tar.gz
   ```

`suite` skips runs that already exist for the current harness version, so it can be resumed. Jev responses are cached on disk, so re-running a repeat costs nothing. A seed > 0 forces fresh calls, which is how variance is measured. The harness version is a hash of the data, Jev, eval and runner code; changing any of them makes existing results stale, and `report` refuses to mix versions.

## Results

Test accuracy, 2026-10-21 run (mean of 3 seeds or repeats; harness `fe5444fddffa`):

| Task | Best local model | Jev (best prompt) | Δ Jev − best local (95% CI) |
|---|---|---|---|
| SST-2 | DistilBERT 0.905 | **0.960** (`sst2_v1`) | **+5.7** (+3.8, +7.7) |
| AG News | **DistilBERT 0.946** | 0.896 (`agnews_v2`) | −4.9 (−5.5, −4.2) |
| Adult | **Tabular LightGBM 0.879** | 0.826 (`adult_contextual_v1`) | −5.5 (−6.6, −4.6) |

- Jev wins on sentiment with zero training and is the best-calibrated model there (ECE 0.018), but trails every local model on AG News and Adult.
- Half of Jev's AG News gap is one boundary: Business stories about tech, telecom and health companies filed as Sci/Tech. AG News labels these inconsistently, and the prompt's rule fixed the opposite error but caused this one.
- Serialization costs trained models almost nothing: TF-IDF+LR and DistilBERT on the raw Adult text dump match native LightGBM within 0.1 point. Jev's Adult gap comes from being zero-shot, not from the text format.
- For Jev, context helps more than formatting: raw → natural +0.4 points, + background paragraph +1.7, about 90% of it from the income correlates rather than the base rate. Twenty few-shot examples raise AUC but lower accuracy, at 5× the tokens.
- Jev: ~370 ms median latency, $0.015–0.096 per 1k predictions. Local models: 0.4–4 ms, under $0.00003 per 1k.
- Caveat: these are public datasets, and Jev's training data is undisclosed, so contamination can't be ruled out for the SST-2 result.

Full write-up: [`reports/jev-benchmark-report.md`](reports/jev-benchmark-report.md) (charts in `reports/figures/`). Machine-generated tables: `results/report/test/summary.md`.

## Budget

Jev pricing on OpenRouter: **$0.042 per 1M input tokens, $0 output**. Actual spend for the 2026-10-21 run: **$4.11** ($3.99 test, $0.12 val iteration) over 100,416 test items. The pre-study estimate below assumed ~400 tokens per call and no few-shot prompts; the two few-shot prompts (~2,200 tokens per call) account for most of the difference. The harness bills from each response's `usage.cost`. Requests send one item each by default (the documented pattern). `batch_size` in a prompt config packs several items into one request's `state.items`; check on val that it doesn't change accuracy before you use it.

| Scope | Calls | Est. tokens | Cost |
|---|---|---|---|
| Track A test sets (SST-2 872 + AG News 7,600) | 8,472 | ~3.4M | ~$0.14 |
| Prompt/serialization iteration (val) | ~2,500 | ~1.0M | ~$0.04 |
| Track B test 5,000 × 3 serialization variants | 15,000 | ~6.0M | ~$0.25 |
| Variance re-runs (2×) | ~48,000 | ~19.2M | ~$0.81 |
| **Total (unbatched)** | **~74,000** | **~29.6M** | **~$1.25** |
| With 20-questions-per-request batching | ~3,700 req | ~9M | **~$0.40** |

Sensitivity: even at 800 tokens/call, total stays under ~$2.50. Practical note: OpenRouter requires prepay (~$10 minimum), so actual upfront spend is the top-up, not the benchmark.

## Roadmap

1. ~~**M1 — Data + harness:** loaders, frozen splits, metric/cost/latency tracking~~ done
2. ~~**M2 — Local baselines:** LR, LightGBM ×2, DistilBERT on both tracks~~ done
3. ~~**M3 — Jev integration:** OpenRouter client, serialization variants, prompt versioning~~ done
4. ~~**M4 — Full comparison:** all models × both tracks × 3 seeds → comparison tables + calibration plots~~ done

Next: an unpublished sentiment set to test for contamination, a structured task with no public training data. Recalibration on val is done: isotonic brings Jev's AG News ECE from 0.059 to 0.014 (see the report).

## Notes

- OpenRouter model id: `typesafe/jev-1.14` (rolling alias `~typesafe/jev-latest`), called via `POST https://openrouter.ai/api/alpha/decisions`. All questions in one request are answered in parallel and cannot see each other's answers. SST-2/AG News use a `choice` question; Adult uses `noul` (probability of >50K).
- Jev is not deterministic (probabilities vary slightly run to run), so test runs use 3 repeats and the report measures repeat agreement.
- Local cost = wall time × rented-hardware rate (`GPU_USD_PER_HOUR`, `CPU_USD_PER_HOUR` in `config.py`). Local latency = single-item `predict_proba` calls on this machine; Jev latency = HTTP round trip.
- Jev returns no reasoning trace — latency/accuracy only, no CoT analysis.
- API costs are logged per run (`cost.spent_this_run_usd` in each result JSON).
