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
    "-m",
    "--mode",
    type=str,
    help="train or eval",
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
PATIENCE = 5
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
  rank_counter = 0
  for win_count, indices in win_count_to_indices.items():
    ordered_indices = list(reversed(sorted([x, scores[x]] for x in indices)))
    for index, _ in ordered_indices:
      is_ranked_sentence_relevant[rank_counter] = int(index in relevant_indices)
      rank_counter += 1

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
          wins[rev_j] += 1
        else:
          wins[rev_k] += 1
      maybe_ap = get_ap_from_wins(wins, helper[review_id][key][rebuttal_index])
      if maybe_ap is not None:
        aps.append(maybe_ap)

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
      print(collections.Counter(p.item() for p in preds))
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


def do_train(tokenizer, model, loss_fn, data_dir, dataset_name, key):
  hyperparams = {
      "key": key,
      "epochs": EPOCHS,
      "patience": PATIENCE,
      "learning_rate": LEARNING_RATE,
      "batch_size": ws_lib.BATCH_SIZE,
      "bert_model": ws_lib.PRE_TRAINED_MODEL_NAME,
  }

  print_dict(hyperparams)
  (
      train_data_loader,
      val_data_loader,
      full_val_data_loader,
  ) = ws_lib.build_data_loaders(data_dir, key, tokenizer)
  val_metric_helper = get_metric_helper(data_dir, "dev_mini")

  optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
  total_steps = len(train_data_loader) * EPOCHS

  scheduler = transformers.get_linear_schedule_with_warmup(
      optimizer, num_warmup_steps=0, num_training_steps=total_steps)

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

    val_mrr = calculate_mrr(
        train_or_eval("eval",
                      model,
                      full_val_data_loader,
                      loss_fn,
                      DEVICE,
                      return_preds=True),
        val_metric_helper,
        key,
    )
    history.append(
        ws_lib.HistoryItem(epoch, train_acc, train_loss, val_acc, val_loss,
                           val_mrr))
    print_dict(history[-1]._asdict())

    if val_acc > best_accuracy:
      torch.save(
          model.state_dict(),
          f"outputs/mrr_trend_best_model_{dataset_name}_{key}.bin",
      )
      best_accuracy = val_acc
      best_accuracy_epoch = epoch

  with open(f"outputs/mrr_trend_history_{dataset_name}_{key}.pkl", "wb") as f:
    pickle.dump((hyperparams, history), f)


def do_eval(tokenizer, model, loss_fn, data_dir, dataset_name, key):
  dataset_name = "mini_ape"
  data_dir = "../data/processed_data/mini_ape/weaksup"
  full_test_data_loader = ws_lib.create_data_loader(
      data_dir,
      "test_all",
      key,
      tokenizer,
      ws_lib.BATCH_SIZE,
  )

  test_metric_helper = get_metric_helper(data_dir, "test_all")

  dataset_name = "disapere"
  model.load_state_dict(
      torch.load(f"outputs/best_model_{dataset_name}_{key}.bin"))

  results = train_or_eval("eval",
                          model,
                          full_test_data_loader,
                          loss_fn,
                          DEVICE,
                          return_preds=True)

  identifiers = []
  labels = []
  for i, l in results:
    identifiers += i
    labels += l

  dataset_name = "mini_ape"

  with open(f"outputs/test_results_{dataset_name}_{key}.pkl", "wb") as f:
    pickle.dump({"results": list(zip(identifiers, labels))}, f)

  test_mrr = calculate_mrr(results, test_metric_helper, key)
  with open(f"outputs/test_results_{dataset_name}_{key}.pkl", "wb") as f:
    pickle.dump(
        {
            "test_map": test_mrr,
            "results": list(zip(identifiers, labels))
        }, f)


def main():

  args = parser.parse_args()

  assert args.mode in [TRAIN, EVAL]

  tokenizer = BertTokenizer.from_pretrained(ws_lib.PRE_TRAINED_MODEL_NAME)
  model = ws_lib.SentimentClassifier(2).to(DEVICE)
  loss_fn = nn.BCEWithLogitsLoss().to(DEVICE)

  key = args.tokenization + "|" + args.corpus_type

  if args.mode == TRAIN:
    do_train(tokenizer, model, loss_fn, args.data_dir, args.dataset_name, key)
  else:
    assert args.mode == EVAL
    do_eval(tokenizer, model, loss_fn, args.data_dir, args.dataset_name, key)


if __name__ == "__main__":
  main()
