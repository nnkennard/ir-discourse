import collections
import transformers
from transformers import BertTokenizer, BertModel

import pandas as pd
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import numpy as np

PRE_TRAINED_MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

device = 'cuda'

class WeakSupervisionDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len=512):
        #self.texts = ['a', 'a', 'a']
        #self.targets = [0,1,1]
        self.texts = texts
        self.targets = [int(i) for i in targets]
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
            "targets": torch.tensor(target, dtype=torch.long),
        }


with open("../data/weaksup/examples_train_text.txt", "r") as f:
    texts = [l.strip() for l in f.readlines()][:2000]

with open("../data/weaksup/examples_train_stanza|review.txt", "r") as f:
    labels = [int(l.strip()) for l in f.readlines()][:2000]

train_df = pd.DataFrame.from_dict({"text": text, "label":label} for text, label in zip(texts, labels))


for a, b in list(zip(texts, labels))[:10]:
  print(a, b)


def create_data_loader(df, tokenizer, batch_size):
    ds = WeakSupervisionDataset(
        df.text.to_numpy(),
        df.label.to_numpy(),
        tokenizer=tokenizer,
    )

    return DataLoader(ds, batch_size=batch_size, num_workers=4)


BATCH_SIZE = 16

train_data_loader = create_data_loader(train_df, tokenizer, BATCH_SIZE)
val_data_loader = create_data_loader(train_df, tokenizer, BATCH_SIZE)
#dev_data_loader = create_data_loader(texts, labels, tokenizer, BATCH_SIZE)


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        #_, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(bert_output["pooler_output"])
        return self.out(output)


model = SentimentClassifier(2)
model = model.to("cuda")


EPOCHS = 10

optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_data_loader) * EPOCHS

scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to('cuda')


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0

    print("BABABABA")
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

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

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


history = collections.defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("-" * 10)

    train_acc, train_loss = train_epoch(
        model, train_data_loader, loss_fn, optimizer, device, scheduler,
        len(texts)
    )

    print(f"Train loss {train_loss} accuracy {train_acc}")

    val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device,
    len(texts))

    print(f"Val   loss {val_loss} accuracy {val_acc}")
    print()

    history["train_acc"].append(train_acc)
    history["train_loss"].append(train_loss)
    history["val_acc"].append(val_acc)
    history["val_loss"].append(val_loss)

    if val_acc > best_accuracy:
        torch.save(model.state_dict(), "best_model_state.bin")
        best_accuracy = val_acc
