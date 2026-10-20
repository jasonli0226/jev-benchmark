"""Model name -> class. Imports are lazy so CPU-only runs skip torch."""

from importlib import import_module

_MODELS = {
    "tfidf_lr": "jev_bench.models.classical:TfidfLR",
    "tfidf_lgbm": "jev_bench.models.classical:TfidfLGBM",
    "embed_lgbm": "jev_bench.models.embedding:EmbedLGBM",
    "distilbert": "jev_bench.models.distilbert:DistilBERT",
    "tabular_lr": "jev_bench.models.tabular:TabularLR",
    "tabular_lgbm": "jev_bench.models.tabular:TabularLGBM",
    "jev": "jev_bench.models.jev_model:Jev",
}
LOCAL_TEXT_MODELS = ("tfidf_lr", "tfidf_lgbm", "embed_lgbm", "distilbert")
TABULAR_MODELS = ("tabular_lr", "tabular_lgbm")
MODEL_NAMES = tuple(_MODELS)


def get_model_class(name: str):
    if name not in _MODELS:
        raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_NAMES}")
    module, attr = _MODELS[name].split(":")
    return getattr(import_module(module), attr)
