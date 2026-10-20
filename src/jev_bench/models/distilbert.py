"""DistilBERT fine-tune with a plain PyTorch loop; best epoch chosen on val."""

import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from jev_bench.models.base import Model

BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256
EPOCHS = 2
LEARNING_RATE = 5e-5
WARMUP_FRACTION = 0.06


class DistilBERT(Model):
    name = "distilbert"
    device = "gpu"

    def _device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _batches(self, texts: list[str], labels=None, shuffle=False, batch_size=EVAL_BATCH_SIZE):
        idx = list(range(len(texts)))
        generator = torch.Generator().manual_seed(self.seed)
        for chunk in DataLoader(idx, batch_size=batch_size, shuffle=shuffle, generator=generator):
            chunk = chunk.tolist()
            enc = self.tokenizer(
                [texts[i] for i in chunk], truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt"
            )
            enc = {k: v.to(self.dev) for k, v in enc.items()}
            if labels is not None:
                enc["labels"] = torch.tensor([labels[i] for i in chunk], device=self.dev)
            yield enc

    @torch.no_grad()
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        out = []
        for enc in self._batches(df["text"].tolist()):
            with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
                logits = self.model(**enc).logits
            out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.dev = self._device()
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=len(self.task.labels)
        ).to(self.dev)
        texts, labels = train["text"].tolist(), train["label"].tolist()
        steps = EPOCHS * -(-len(texts) // TRAIN_BATCH_SIZE)
        optim = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
        sched = get_linear_schedule_with_warmup(optim, int(WARMUP_FRACTION * steps), steps)

        best_acc, best_state, self.epoch_val_accuracy = -1.0, None, []
        for _ in range(EPOCHS):
            self.model.train()
            for enc in self._batches(texts, labels, shuffle=True, batch_size=TRAIN_BATCH_SIZE):
                with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
                    loss = self.model(**enc).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
            acc = float((self.predict_proba(val).argmax(1) == val["label"].to_numpy()).mean())
            self.epoch_val_accuracy.append(acc)
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(best_state)

    def run_info(self) -> dict:
        return {"base_model": BASE_MODEL, "epochs": EPOCHS, "epoch_val_accuracy": self.epoch_val_accuracy}
