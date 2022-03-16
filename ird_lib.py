import collections
import glob
import json
import tqdm
import numpy as np

from rank_bm25 import BM25Okapi

class Preprocessors(object):
  STANZA = "stanza"
  BERT_STOP = "bert_lemma"
  RAW = "raw"
  ALL = [
      STANZA,
      BERT_STOP,
      RAW
  ]

# == Data prep ===============================================

Example = collections.namedtuple(
    "Example", "review_lines rebuttal_lines discrete_mapping identifier".split())

def dump_raw_text_to_file(examples, data_path, dataset, subset):
  with open(f"{data_path}/{dataset}_{subset}.json", 'w') as f:
    json.dump([x._asdict() for x in examples], f)


class Corpus(object):
  REVIEW = "review"
  FULL = "full"
  ALL = [REVIEW, FULL]


SUBSETS = "train dev test".split()

TextInfo = collections.namedtuple(
    "TextInfo",
    "review_sentences rebuttal_sentences alignment_map review_len".split())


def batch_preprocess(data, preprocessor):
  preprocessed = []
  for i in range(0, len(data), 200):
    preprocessed += preprocessor(data[i:i + 200])
  return preprocessed

def load_examples(data_dir, dataset_name, subset):
  with open(f"{data_dir}/{dataset_name}_{subset}.json", 'r') as f:
    return json.load(f)

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

  def process_sentences(self, sentences):
    sentence_texts = [x["text"] for x in sentences]
    return {
        preprocessor: batch_preprocess(sentence_texts, tlib.PREP[preprocessor])
        for preprocessor in tlib.Preprocessors.ALL
    }, len(sentence_texts)

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

  def score(self, overall_models):
    self.scores = collections.defaultdict(dict)
    for subset, reviews in self.texts.items():
      print(subset)
      for review_id, info in tqdm.tqdm(reviews.items()):
        review_sentences, rebuttal_sentences, alignment_map, _ = info
        this_review_scores = {"discrete": alignment_map}
        for preprocessor in tlib.Preprocessors.ALL:
          if preprocessor == 'raw':
            continue
          mini_model = BM25Okapi(review_sentences[preprocessor])
          big_scores = []
          small_scores = []
          for i, query in enumerate(rebuttal_sentences[preprocessor]):
            offsets = self.offset_map[review_id]
            big_scores.append(overall_models[preprocessor].get_scores(query)
                              [offsets[0]:offsets[1]])
            small_scores.append(mini_model.get_scores(query))
          this_review_scores.update({
              "|".join([preprocessor, Corpus.REVIEW]): np.array(small_scores),
              "|".join([preprocessor, Corpus.FULL]): np.array(big_scores),
          })
        self.scores[subset][review_id] = this_review_scores
