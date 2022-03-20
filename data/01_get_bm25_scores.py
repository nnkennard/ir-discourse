import argparse
import collections
import numpy as np
import pickle
import tqdm

import ird_lib
import tokenization_lib as tlib
from rank_bm25 import BM25Okapi

parser = argparse.ArgumentParser(description="Score datasets for WS training")
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    help="path to data file containing score jsons",
)
parser.add_argument(
    "-n",
    "--name",
    type=str,
    help="dataset name",
)
# =============================================================================


def build_overall_corpus(tokenized_examples):
  corpora = collections.defaultdict(list)
  offset = 0
  offset_map = {}
  for subset, subset_examples in tokenized_examples.items():
    for review_id, example in subset_examples.items():
      review_sentences = example.tokenized_review_lines
      offset_map[review_id] = (offset, offset + len(review_sentences["raw"]))
      for preprocessor, tokenized in review_sentences.items():
        corpora[preprocessor] += tokenized
      offset += len(review_sentences)
  return corpora, offset_map


def batch_preprocess(data, preprocessor):
  preprocessed = []
  for i in range(0, len(data), 200):
    preprocessed += preprocessor(data[i:i + 200])
  return preprocessed


def tokenize_sentences(sentences):
  output = {
      preprocessor: batch_preprocess(sentences, tlib.PREP[preprocessor])
      for preprocessor in ird_lib.Preprocessors.ALL
  }
  output[ird_lib.RAW] = sentences
  return output


def tokenize(examples):
  tokenized_examples = {}
  for subset, subset_examples in examples.items():
    tokenized_subset_examples = {}
    print(f"Tokenizing examples in {subset}")
    for review_id, example in tqdm.tqdm(subset_examples.items()):
      tokenized_subset_examples[review_id] = ird_lib.TokenizedExample(
          tokenize_sentences(example["review_lines"]),
          tokenize_sentences(example["rebuttal_lines"]),
          example["discrete_mapping"],
          review_id,
      )
    tokenized_examples[subset] = tokenized_subset_examples
  return tokenized_examples


def score_examples(tokenized_examples, overall_models, offset_map):
  scores = collections.defaultdict(dict)
  for subset, subset_examples in tokenized_examples.items():
    print(f"Scoring examples in {subset}")
    for review_id, info in tqdm.tqdm(subset_examples.items()):
      review_sentences, rebuttal_sentences, alignment_map, _ = info
      offsets = offset_map[review_id]
      # print(review_sentences)
      # print("====")
      # print(len(review_sentences["raw"]), offsets[1] - offsets[0])
      this_review_scores = {"discrete": alignment_map}
      for preprocessor in tlib.Preprocessors.ALL:
        mini_model = BM25Okapi(review_sentences[preprocessor])
        big_scores = []
        small_scores = []
        for i, query in enumerate(rebuttal_sentences[preprocessor]):
          big_scores.append(overall_models[preprocessor].get_scores(query)
                            [offsets[0]:offsets[1]])
          small_scores.append(mini_model.get_scores(query))
        this_review_scores.update({
            "|".join([preprocessor, ird_lib.Corpus.REVIEW]):
                np.array(small_scores),
            "|".join([preprocessor, ird_lib.Corpus.FULL]):
                np.array(big_scores),
        })
      scores[subset][review_id] = this_review_scores
  return scores


def main():
  args = parser.parse_args()

  examples = {
      subset: ird_lib.load_examples(args.data_dir, args.name, subset)
      for subset in ird_lib.SUBSETS
  }
  tokenized_examples = tokenize(examples)

  overall_corpus, offset_map = build_overall_corpus(tokenized_examples)
  overall_models = {
      preprocessor: BM25Okapi(sentences)
      for preprocessor, sentences in overall_corpus.items()
  }
  with open(f"{args.data_dir}/scores.pkl", "wb") as f:
    pickle.dump(score_examples(tokenized_examples, overall_models, offset_map),
                f)


if __name__ == "__main__":
  main()
