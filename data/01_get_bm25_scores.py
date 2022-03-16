import argparse
import pickle
import ird_lib
import pickle

import tokenizer_lib as tlib

parser = argparse.ArgumentParser(description="prepare CSVs for ws training")
parser.add_argument(
    "-d",
    "--data_dir",
    default="../DISAPERE/DISAPERE/final_dataset/",
    type=str,
    help="path to data file containing score jsons",
)
parser.add_argument(
    "-s",
    "--score_file",
    default="data/bm25_scores.pkl",
    type=str,
    help="path to data file containing score jsons",
)
# =============================================================================

def build_overall_corpus(tokenzied_examples):
  corpora = collections.defaultdict(list)
  offset = 0
  offset_map = {}
  for subset, subset_examples in tokenized_examples.items():
    for review_id, example in subset_examples.items():
      review_sentences = example["review_lines"]
      offset_map[review_id] = (offset, offset + len(review_sentences))
      for preprocessor, tokenized in review_sentences.items():
        corpora[preprocessor] += tokenized
      offset += review_len
  return corpora, offset_map

def batch_preprocess(data, preprocessor):
  preprocessed = []
  for i in range(0, len(data), 200):
    preprocessed += preprocessor(data[i:i + 200])
  return preprocessed


def tokenize_sentences(sentences):
  output = {
      preprocessor: batch_preprocess(sentence_texts, ird_lib.PREP[preprocessor])
      for preprocessor in ird_lib.Preprocessors.ALL
  }
  output[ird_lib.RAW] = sentence_texts
  return output

def tokenize_example(examples):
  tokenized_examples = {}
  for subset, subset_examples in examples.items():
    tokenized_subset_examples = {}
    print(f"Tokenizing examples in {subset}")
    for review_id, example in tqdm.tqdm(subset_examples.items()):
      tokenized_subset_examples[review_id] = TokenizedExample(
        tokenize_sentences(example['review_lines']),
        tokenize_sentences(example['rebuttal_lines']),
        example['discrete_mapping'],
        review_id)
    tokenized_examples[subset] = tokenized_subset_examples
  return tokenized_examples

def main():
  args = parser.parse_args()

  examples = {subset: load_examples(data_dir, dataset_name, subset)
                        for subset in ird_lib.SUBSETS}
  tokenized_examples = tokenize(examples)

  overall_corpus, offset_map = build_overall_corpus(examples)

  with open(args.score_file, "wb") as f:
    pickle.dump(texts, f)


if __name__ == "__main__":
  main()
