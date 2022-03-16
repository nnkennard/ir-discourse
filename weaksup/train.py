import collections
import transformers
from transformers import BertTokenizer, BertModel

import pandas as pd
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import numpy as np
import tqdm

PRE_TRAINED_MODEL_NAME = "bert-base-uncased"

device = "cuda"


class WeakSupervisionDataset(Dataset):

  def __init__(self, texts, targets, tokenizer, max_len=512):
    self.texts = texts
    self.target_indices = targets
    self.targets = [np.eye(2, dtype=np.float64)[int(i)] for i in targets]
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, item):
    text = str(self.texts[item])
    target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    return {
        "reviews_text": text,
        "input_ids": encoding["input_ids"].flatten(),
        "attention_mask": encoding["attention_mask"].flatten(),
        "targets": torch.tensor(target, dtype=torch.float64),
        "target_indices": self.target_indices[item],
    }


class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    output = self.drop(bert_output["pooler_output"])
    # print(self.out(output))
    return self.out(output)


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
  model = model.train()

  n_examples = len(data_loader.dataset)
  losses = []
  correct_predictions = 0

  for d in tqdm.tqdm(data_loader):
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(
        outputs,
        targets,
    )

    correct_predictions += torch.sum(preds == d["target_indices"].to(device))
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device):
  model = model.eval()

  n_examples = len(data_loader.dataset)

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == d["target_indices"].to(device))
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)


def create_data_loader(subset, key, tokenizer, batch_size):

  with open(f"../data/weaksup/examples_{subset}_text.txt", "r") as f:
    texts = [l.strip() for l in f.readlines()]

  with open(f"../data/weaksup/examples_{subset}_{key}.txt", "r") as f:
    labels = [int(l.strip()) for l in f.readlines()]

  ds = WeakSupervisionDataset(
      texts,
      labels,
      tokenizer=tokenizer,
  )

  return DataLoader(ds, batch_size=batch_size, num_workers=4)


BATCH_SIZE = 64


def build_data_loaders(key, tokenizer):
  return (
      create_data_loader("train", key, tokenizer, BATCH_SIZE),
      create_data_loader("dev", key, tokenizer, BATCH_SIZE),
  )


def main():

  key = "bert_lemma|full"

  tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
  train_data_loader, val_data_loader = build_data_loaders(key, tokenizer)

  model = SentimentClassifier(2)
  model = model.to("cuda")

  EPOCHS = 10

  optimizer = AdamW(model.parameters(), lr=2e-5)
  total_steps = len(train_data_loader) * EPOCHS

  scheduler = transformers.get_linear_schedule_with_warmup(
      optimizer, num_warmup_steps=0, num_training_steps=total_steps)

  loss_fn = nn.BCEWithLogitsLoss().to("cuda")

  history = collections.defaultdict(list)
  best_accuracy = 0

  for epoch in range(EPOCHS):

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("-" * 10)

    train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn,
                                        optimizer, device, scheduler)

    train_acc, train_loss = eval_model(model, train_data_loader, loss_fn,
                                       device)
    print(f"Train loss {train_loss} accuracy {train_acc}")

    val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device)

    print(f"Val   loss {val_loss} accuracy {val_acc}")
    print()

    history["train_acc"].append(train_acc)
    history["train_loss"].append(train_loss)
    history["val_acc"].append(val_acc)
    history["val_loss"].append(val_loss)

    if val_acc > best_accuracy:
      torch.save(model.state_dict(), f"outputs/best_model_{key}.bin")
      best_accuracy = val_acc


if __name__ == "__main__":
  main()
