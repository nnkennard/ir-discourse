from torch.utils.data import DataLoader
import collections
from sentence_transformers import losses
from sentence_transformers import SentenceTransformer, evaluation
from sentence_transformers.readers import InputExample
import os
from tqdm import tqdm
import json

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

NUM_EPOCHS = 5
TRAIN_BATCH_SIZE = 128

# Train the model
def main():
  data_dir = "../data/processed_data/ape/"

  datasets = {
  subset: build_dataset(data_dir, subset)
  for subset in "train dev test".split() }


  dev_data_loader = build_data_loader(datasets['dev'][0])


  model = SentenceTransformer('all-MiniLM-L6-v2')
  model_save_path = f'outputs/best_ape_model'
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


if __name__ == "__main__":
  main()
