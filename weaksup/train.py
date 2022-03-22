import argparse
import collections
import json
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import tqdm
import transformers

from contextlib import nullcontext
from torch.optim import AdamW
from transformers import BertTokenizer
from rank_metrics import average_precision

import ws_lib

parser = argparse.ArgumentParser(description="Create examples for WS training")
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    help="path to data file containing score jsons",
)

parser.add_argument(
    "-t",
    "--tokenization",
    type=str,
    help="type of tokenization",
)
parser.add_argument(
    "-n",
    "--dataset_name",
    type=str,
    help="e.g. disapere or ape",
)
parser.add_argument(
    "-c",
    "--corpus_type",
    type=str,
    help="type of corpus for calculating scores (full dataset or review)",
)

DEVICE = "cuda"
EPOCHS = 100
PATIENCE = 20
LEARNING_RATE = 2e-5
TRAIN, EVAL = "train eval".split()


def get_ap_from_wins(wins, helper):
  scores = helper["bm25_scores"]
  relevant_indices = helper["actual_aligned_indices"]
  if not relevant_indices:
    return None
  win_count_to_indices = collections.OrderedDict()
  num_review_sentences = len(scores)

  missing_indices = set(range(num_review_sentences))
  for review_index, num_wins in wins.most_common():
    missing_indices.remove(int(review_index))
    if num_wins in win_count_to_indices:
      win_count_to_indices[num_wins].append(int(review_index))
    else:
      win_count_to_indices[num_wins] = [int(review_index)]
  win_count_to_indices[0] = list(sorted(missing_indices))

  is_ranked_sentence_relevant = [None] * num_review_sentences
  for win_count, indices in win_count_to_indices.items():
    ordered_indices = list(reversed(sorted([x, scores[x]] for x in indices)))
    for index, _ in ordered_indices:
      is_ranked_sentence_relevant[index] = int(index in relevant_indices)

  return average_precision(is_ranked_sentence_relevant)


def calculate_mrr(results, helper, key):
  identifiers = []
  preds = []
  by_review_id = collections.defaultdict(lambda: collections.defaultdict(dict))
  for batch in results:
    i_list, p_list = batch
    identifiers += i_list
    preds += p_list
  for i, p in zip(identifiers, preds):
    review_id, reb_i, rev_j, rev_k = i.split("|")
    by_review_id[review_id][reb_i][(rev_j, rev_k)] = p

  aps = []
  for review_id, review_preds in by_review_id.items():
    for rebuttal_index, this_reb_preds in review_preds.items():
      wins = collections.Counter()
      for (rev_j, rev_k), v in this_reb_preds.items():
        if v:
          wins[rev_k] += 1
        else:
          wins[rev_j] += 1
      maybe_ap = get_ap_from_wins(wins, helper[review_id][key][rebuttal_index])
      if maybe_ap is not None:
        aps.append(maybe_ap)

  print(aps)
  assert len(identifiers) == len(preds)
  return np.mean(aps)


def train_or_eval(
    mode,
    model,
    data_loader,
    loss_fn,
    device,
    return_preds=False,
    optimizer=None,
    scheduler=None,
):
  assert mode in [TRAIN, EVAL]
  is_train = mode == TRAIN
  if is_train:
    model = model.train()
    context = nullcontext()
    assert optimizer is not None
    assert scheduler is not None
  else:
    model = model.eval()
    context = torch.no_grad()

  results = []
  losses = []
  correct_predictions = 0
  n_examples = len(data_loader.dataset)

  with context:
    for d in tqdm.tqdm(data_loader):
      input_ids, attention_mask, targets, target_indices = [
          d[k].to(device)
          for k in "input_ids attention_mask targets target_indices".split()
      ]

      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      _, preds = torch.max(outputs, dim=1)
      if return_preds:
        results.append((d["identifier"], preds.cpu().numpy().tolist()))
      loss = loss_fn(outputs, targets)
      correct_predictions += torch.sum(preds == target_indices)
      losses.append(loss.item())
      if is_train:
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

  if return_preds:
    return results
  else:
    return correct_predictions.double().item() / n_examples, np.mean(losses)


def get_metric_helper(data_dir, subset_key):
  with open(f"{data_dir}/examples_{subset_key}_helper.json", "r") as f:
    return json.load(f)


def print_dict(dict_to_print):
  for k, v in dict_to_print.items():
    print(f"{k}\t{v}")

def main():

  args = parser.parse_args()

  hyperparams = {
      "tokenization": args.tokenization,
      "corpus": args.corpus_type,
      "epochs": EPOCHS,
      "patience": PATIENCE,
      "learning_rate": LEARNING_RATE,
      "batch_size": ws_lib.BATCH_SIZE,
      "bert_model": ws_lib.PRE_TRAINED_MODEL_NAME,
  }

  print_dict(hyperparams)

  key = args.tokenization + "|" + args.corpus_type

  tokenizer = BertTokenizer.from_pretrained(ws_lib.PRE_TRAINED_MODEL_NAME)
  (
      train_data_loader,
      val_data_loader,
      full_val_data_loader,
  ) = ws_lib.build_data_loaders(args.data_dir, key, tokenizer)
  val_metric_helper = get_metric_helper(args.data_dir, "dev_mini")

  model = ws_lib.SentimentClassifier(2).to(DEVICE)
  optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
  total_steps = len(train_data_loader) * EPOCHS

  scheduler = transformers.get_linear_schedule_with_warmup(
      optimizer, num_warmup_steps=0, num_training_steps=total_steps)

  loss_fn = nn.BCEWithLogitsLoss().to(DEVICE)

  history = []
  best_accuracy = 0
  best_accuracy_epoch = None

  for epoch in range(EPOCHS):

    if best_accuracy_epoch is not None and epoch - best_accuracy_epoch > PATIENCE:
      break
    mrr_accumulator = []

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("-" * 10)

    train_acc, train_loss = train_or_eval(
        "train",
        model,
        train_data_loader,
        loss_fn,
        DEVICE,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    val_acc, val_loss = train_or_eval("eval", model, val_data_loader, loss_fn,
                                      DEVICE)

    val_mrr = calculate_mrr(train_or_eval("eval", model, full_val_data_loader, loss_fn,
       DEVICE, return_preds=True), val_metric_helper, key)
    history.append(
        ws_lib.HistoryItem(epoch, train_acc, train_loss, val_acc, val_loss, val_mrr))
    print_dict(history[-1]._asdict())

    if val_acc > best_accuracy:
      torch.save(
          model.state_dict(),
          f"outputs/best_model_{args.dataset_name}_{key}.bin",
      )
      best_accuracy = val_acc
      best_accuracy_epoch = epoch

  with open(f"outputs/history_{args.dataset_name}_{key}.pkl", "wb") as f:
    pickle.dump((hyperparams, history), f)


if __name__ == "__main__":
  main()
