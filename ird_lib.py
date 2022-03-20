import collections
import glob
import json
import tqdm
import numpy as np


class Preprocessors(object):
  STANZA = "stanza"
  BERT_STOP = "bert_stop"
  ALL = [
      STANZA,
      BERT_STOP,
  ]


RAW = "raw"

SUBSETS = "train dev test".split()

# == Data prep ===============================================

Example = collections.namedtuple(
    "Example",
    "review_lines rebuttal_lines discrete_mapping identifier".split())

TokenizedExample = collections.namedtuple(
    "TokenizedExample",
    "tokenized_review_lines tokenized_rebuttal_lines discrete_mapping identifier"
    .split(),
)


def dump_raw_text_to_file(examples, data_dir, subset):
  with open(f"{data_dir}/{subset}_raw.json", "w") as f:
    json.dump([x._asdict() for x in examples], f)


def load_examples(data_dir, dataset_name, subset):
  with open(f"{data_dir}/{subset}_raw.json", "r") as f:
    return {example["identifier"]: example for example in json.load(f)}


class Corpus(object):
  REVIEW = "review"
  FULL = "full"
  ALL = [REVIEW, FULL]


class Texts(object):

  def __init__(self, data_dir, dataset_name):

    for subset in ird_lib.SUBSETS:
      examples = load_examples(data_dir, dataset_name, subset)

    # Preprocess examples

    print("Building overall models")
    self.build_overall_corpus()
    overall_models = {
        preprocessor: BM25Okapi(sentences)
        for preprocessor, sentences in self.corpora.items()
    }
    print("Scoring")
    self.score(overall_models)

  def build_overall_corpus(self):
    self.corpora = collections.defaultdict(list)
    offset = 0
    self.offset_map = {}
    for subset, example_map in self.texts.items():
      for review_id in sorted(example_map.keys()):
        review_sentences, _, _, review_len = example_map[review_id]
        self.offset_map[review_id] = (offset, offset + review_len)
        for preprocessor, tokenized in review_sentences.items():
          if preprocessor == "raw":
            continue
          self.corpora[preprocessor] += tokenized
        offset += review_len
