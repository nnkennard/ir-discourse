import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel

BATCH_SIZE = 64
PRE_TRAINED_MODEL_NAME = "bert-base-uncased"


class WeakSupervisionDataset(Dataset):

  def __init__(self, texts, targets, tokenizer, max_len=512):
    self.identifiers, self.texts = zip(*[x.split("\t",1 ) for x in texts])
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
        "identifier": self.identifiers[item],
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
    return self.out(output)


def create_data_loader(data_dir, subset, key, tokenizer, batch_size):

  with open(f"{data_dir}/examples_{subset}_text.txt", "r") as f:
    texts = [l.strip() for l in f.readlines()]

  with open(f"{data_dir}/examples_{subset}_{key}.txt", "r") as f:
    labels = [int(l.strip()) for l in f.readlines()]

  ds = WeakSupervisionDataset(
      texts,
      labels,
      tokenizer=tokenizer,
  )

  return DataLoader(ds, batch_size=batch_size, num_workers=8)


def build_data_loaders(data_dir, key, tokenizer):
  return (
      create_data_loader(
          data_dir,
          "train",
          key,
          tokenizer,
          BATCH_SIZE,
      ),
      create_data_loader(
          data_dir,
          "dev",
          key,
          tokenizer,
          BATCH_SIZE,
      ),
      create_data_loader(
          data_dir,
          "dev_mini",
          key,
          tokenizer,
          BATCH_SIZE,
      ),
  )
