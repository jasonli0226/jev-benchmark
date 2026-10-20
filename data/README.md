# Datasets

This folder holds the frozen train / val / test splits used for every result in this repo. Each split is a Parquet file, and `manifest.json` records its row count, label counts and SHA-256 hash. The harness checks the hash on every load and refuses to run if a split has changed.

Splits were built once with `uv run jev-bench prepare` (split seed `20261021`, code in `src/jev_bench/data/`). Do not re-run `prepare --overwrite` unless you intend to invalidate every existing result.

## Sources and licenses

| Dataset | Downloaded from | Original source | License / terms | Cite |
|---|---|---|---|---|
| SST-2 | Hugging Face [`stanfordnlp/sst2`](https://huggingface.co/datasets/stanfordnlp/sst2) | [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/) | Not stated (the Hugging Face card lists "Unknown"; the original page gives no license) | Socher et al., [Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](https://aclanthology.org/D13-1170), EMNLP 2013 |
| AG News | Hugging Face [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) | [AG's corpus of news articles](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html) (A. Gulli) | Non-commercial use: the corpus page says "You are encouraged to download this corpus for any non-commercial use" and asks users to cite that page | Zhang, Zhao & LeCun, Character-level Convolutional Networks for Text Classification, NIPS 2015 |
| UCI Adult | OpenML [dataset 1590](https://www.openml.org/d/1590) (`adult`, version 2) | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (per UCI) | Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20 |

The copies here are redistributed for non-commercial research and benchmarking only, under the terms above. If you reuse them, follow each source's terms and cite the original authors.

## Files and columns

```
data/
├── manifest.json        # per split: rows, label_counts, sha256
├── sst2/{train,val,test}.parquet
├── agnews/{train,val,test}.parquet
└── adult/{train,val,test}.parquet
```

Every file has an `id` column: the row's position within its split, used to pair predictions across models. `label` is an integer index into the label list below; result files and Jev prompts use the same order.

| Task | Columns | Labels (index: name) |
|---|---|---|
| SST-2 | `id`, `text` (str), `label` (int) | 0: negative, 1: positive |
| AG News | `id`, `text` (str: title and description), `label` (int) | 0: world, 1: sports, 2: business, 3: sci_tech |
| Adult | `id`; numeric `age`, `capital_gain`, `capital_loss`, `hours_per_week` (int); categorical `workclass`, `education`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country` (string); `label` (int) | 0: <=50K, 1: >50K |

## Splits

| Task | Train | Val | Test | Positive / class balance (test) |
|---|---|---|---|---|
| SST-2 | 65,349 | 2,000 | 872 | 444 positive, 428 negative |
| AG News | 115,000 | 5,000 | 7,600 | 1,900 per class |
| Adult | 41,842 | 2,000 | 5,000 | 1,196 >50K (23.9%) |

How each split was made:

- **SST-2.** The GLUE test split has no public labels, so the GLUE **validation** split (872 sentences) is our test set. Val is a stratified sample of 2,000 rows from the GLUE train split; the remaining 65,349 rows are train. Text is stripped of leading and trailing whitespace.
- **AG News.** The official test split (7,600) is our test set. Val is a stratified sample of 5,000 rows from the official train split; the remaining 115,000 are train.
- **Adult.** OpenML version 2 combines the original UCI train and test files (48,842 rows). Test (5,000) and then val (2,000) are stratified samples from that pool; the remaining 41,842 are train. Two columns are dropped for every model, including Jev: `fnlwgt` (a census sampling weight, not a property of the person) and `education-num` (a numeric recoding of `education`). Missing categorical values are stored as the string `unknown` (in train: 2,409 for `workclass`, 2,416 for `occupation`, 744 for `native_country`). Column names use underscores instead of hyphens.

All stratified samples use `sklearn.model_selection.train_test_split` with `random_state=20261021`.

## Known issues

- **SST-2 val does not look like SST-2 test.** GLUE's SST-2 train split contains sub-sentence phrases from the treebank, so val (drawn from train) is mostly short phrases: median 7 words in train vs 19 in test. Val accuracy runs much higher than test (TF-IDF+LR: 0.942 val vs 0.825 test). Use val for model selection only; don't report SST-2 val numbers.
- **Possible pre-training contamination.** All three datasets are widely published. A model trained on web data, including Jev, may have seen them, so its test scores here may be optimistic.
- **Adult reflects 1994 US census data.** Income labels and correlations (by sex, race, country) reflect that population and time. They are a benchmark target, not a basis for decisions about people.

How the Adult rows are turned into text for Jev and the text models is described in the main [README](../README.md#serialization-variants-track-b-jev-only).
