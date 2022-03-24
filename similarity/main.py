from torch.utils.data import DataLoader
import collections
from sentence_transformers import losses, util
from sentence_transformers import SentenceTransformer, evaluation
from sentence_transformers.readers import InputExample
import os
import torch
from tqdm import tqdm
import json
import numpy as np
from rank_metrics import reciprocal_rank, average_precision

distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE

# Negative pairs should have a distance of at least 0.5
margin = 5


def build_dataset(data_dir, subset):
  input_examples = collections.defaultdict(list)
  discrete_maps = {}
  with open(f'{data_dir}/{subset}_raw.json', 'r') as f:
    examples = json.load(f)
    for example in examples:
      discrete_maps[example['identifier']] = example['discrete_mapping']
      for reb_i, reb_sentence in enumerate(example["rebuttal_lines"]):
        for rev_i, rev_sentence in enumerate(example["review_lines"]):
          label = example['discrete_mapping'][reb_i][rev_i]
          input_examples[example["identifier"]].append(
          InputExample(texts=[reb_sentence, rev_sentence],
                                label=label))
  return input_examples, discrete_maps

def build_data_loader(dataset):
  return DataLoader(sum(dataset.values(), []), shuffle=True, batch_size=TRAIN_BATCH_SIZE)

def build_binary_evaluator(data_loader):
  texts_1 = []
  texts_2 = []
  labels = []
  for example in data_loader.dataset:
    text_1, text_2 = example.texts
    texts_1.append(text_1)
    texts_2.append(text_2)
    labels.append(example.label)

  return evaluation.BinaryClassificationEvaluator(texts_1, texts_2, labels)

def get_correct_indices(discrete_map):
  correct_indices = {}
  for reb_i, reb_row in enumerate(discrete_map):
    correct_indices[reb_i] = [i for i, x in enumerate(reb_row) if x]
  return correct_indices

def convert_scores_to_is_relevant(scores, correct_indices):
  is_relevant_map = [None] * len(scores)
  ordered_scores = list(reversed(sorted((score, i) for i, score in
  enumerate(scores))))
  for rank, (score, i) in enumerate(ordered_scores):
    if i in correct_indices:
      is_relevant_map[rank] = 1
    else:
      is_relevant_map[rank] = 0
  assert set(is_relevant_map) == set([0,1])
  return is_relevant_map

def evaluate(model, test_file):
  with open(test_file, 'r') as f:
    examples = json.load(f)
  rrs = []
  aps = []
  for example in examples:
    review_embs = model.encode(example['review_lines'])
    rebuttal_embs = model.encode(example['rebuttal_lines'])
    correct_indices = get_correct_indices(example['discrete_mapping'])
    cosine_scores = util.pytorch_cos_sim(rebuttal_embs, review_embs)
    for reb_i, reb_scores in enumerate(cosine_scores):
      if not correct_indices[reb_i]:
        continue
      else:
        is_relevant_map = convert_scores_to_is_relevant(
        reb_scores, correct_indices[reb_i])
        aps.append(average_precision(is_relevant_map))
        rrs.append(reciprocal_rank(is_relevant_map))
  print(f"MAP: {np.mean(aps)}, MRR: {np.mean(rrs)}")

NUM_EPOCHS = 10
TRAIN_BATCH_SIZE = 128

# Train the model
def main():
  data_dir = "../data/processed_data/disapere/"
  subset = "train"

  datasets = {
  subset: build_dataset(data_dir, subset)
  for subset in "train dev test".split() }


  dev_data_loader = build_data_loader(datasets['dev'][0])


  model = SentenceTransformer('all-MiniLM-L6-v2')
  model_save_path = f'outputs/best_model'
  os.makedirs(model_save_path, exist_ok=True)

  evaluators = [build_binary_evaluator(dev_data_loader)]
  seq_evaluator = evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])
  train_loss = losses.OnlineContrastiveLoss(model=model, distance_metric=distance_metric, margin=margin)

  model.fit(train_objectives=[(build_data_loader(datasets['train'][0]), train_loss)],
            evaluator=seq_evaluator,
            epochs=NUM_EPOCHS,
            warmup_steps=1000,
            output_path=model_save_path
            )

  evaluate(model, f"{data_dir}/{subset}_raw.json")

if __name__ == "__main__":
  main()
