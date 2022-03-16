import argparse
import collections
import numpy as np
import pickle
import random
import tqdm

import ird_lib

random.seed(47)

parser = argparse.ArgumentParser(description="Create examples for WS training")
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    help="path to data file containing score jsons",
)


def build_text(rebuttal_sentence, review_sentence_1, review_sentence_2):
  return "[CLS] " + " [SEP] ".join(
      [rebuttal_sentence, review_sentence_1, review_sentence_2])


def sample_indices(num_review_sentences, num_rebuttal_sentences):
  pool = []
  for i in range(num_rebuttal_sentences):
    for j in range(num_review_sentences):
      for k in range(num_review_sentences):
        if j == k:
          continue
        pool.append((i, j, k))

  num_samples = num_review_sentences * num_rebuttal_sentences
  return random.sample(pool, num_samples)


def get_example_texts(review_sentences, rebuttal_sentences, sampled_indices):
  return [
      build_text(rebuttal_sentences[reb_i], review_sentences[rev_j],
                 review_sentences[rev_k])
      for reb_i, rev_j, rev_k in sampled_indices
  ]

def sample_and_label(subset_scores, raw_text_map):
  example_texts = []
  label_map = collections.defaultdict(list)
  for review_id, scores in tqdm.tqdm(subset_scores.items()):
    review_sentences = raw_text_map[review_id]["review_lines"]
    rebuttal_sentences = raw_text_map[review_id]["rebuttal_lines"]
    sampled_indices = sample_indices(len(review_sentences),
                                     len(rebuttal_sentences))

    review_example_texts = get_example_texts(review_sentences,
                                             rebuttal_sentences,
                                             sampled_indices)

    review_label_map = collections.defaultdict(list)
    for key, score_matrix in scores.items():
      if key == "discrete":
        continue
      for reb_i, rev_j, rev_k in sampled_indices:
        score_j = score_matrix[reb_i][rev_j]
        score_k = score_matrix[reb_i][rev_k]
        label = 1 if score_j > score_k else 0
        review_label_map[key].append(label)
    for key, labels in review_label_map.items():
      label_map[key] += labels
    example_texts += review_example_texts
  return example_texts, label_map

def main():
  args = parser.parse_args()
  with open(f"{args.data_dir}/scores.pkl", "rb") as f:
    scores = pickle.load(f)
  example_map = collections.defaultdict(lambda: collections.defaultdict(list))
  for subset, subset_scores in scores.items():
    print(f"Sampling examples from {subset}")
    raw_text_map = ird_lib.load_examples(args.data_dir, None, subset)
    example_texts, label_map = sample_and_label(subset_scores, raw_text_map)
    for key, labels in label_map.items():
      with open(f"{args.data_dir}/weaksup/examples_{subset}_{key}.txt", "w") as f:
        f.write("\n".join(str(i) for i in labels))
    with open(f"{args.data_dir}/weaksup/examples_{subset}_text.txt", "w") as f:
      f.write("\n".join(example_texts))


if __name__ == "__main__":
  main()
