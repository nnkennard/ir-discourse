import argparse
import collections
import glob
import json
import numpy as np
import pickle
import stanza
import tqdm

from rank_bm25 import BM25Okapi
from transformers import BertTokenizer

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
    default="bm25_scores.pkl",
    type=str,
    help="path to data file containing score jsons",
)


class Preprocessors(object):
  STANZA = "stanza"
  BERT = "bert"
  ALL = [STANZA, BERT]


class Corpus(object):
  REVIEW = "review"
  FULL = "full"
  ALL = [REVIEW, FULL]


SUBSETS = "train dev test".split()

# ===== Preprocessing =========================================================

with open("nltk_stopwords.json", "r") as f:
  STOPWORDS = json.load(f)

BERT_TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")


def bert_preprocess(sentences):
  return [[
      tok for tok in BERT_TOKENIZER.tokenize(sentence) if tok not in STOPWORDS
  ] for sentence in sentences]


STANZA_PIPELINE = stanza.Pipeline("en",
                                  processors="tokenize,lemma",
                                  tokenize_no_ssplit=True)


def stanza_preprocess(sentences):
  doc = STANZA_PIPELINE("\n\n".join(sentences))
  lemmatized = []
  for sentence in doc.sentences:
    sentence_lemmas = []
    for token in sentence.tokens:
      (token_dict,) = token.to_dict()
      maybe_lemma = token_dict["lemma"].lower()
      if maybe_lemma not in STOPWORDS:
        sentence_lemmas.append(maybe_lemma)
    lemmatized.append(sentence_lemmas)
  return lemmatized


PREP = {
    Preprocessors.STANZA: stanza_preprocess,
    Preprocessors.BERT: bert_preprocess
}

# =============================================================================

TextInfo = collections.namedtuple(
    "TextInfo",
    "review_sentences rebuttal_sentences alignment_map review_len".split())


def batch_preprocess(data, preprocessor):
  preprocessed = []
  for i in range(0, len(data), 200):
    preprocessed += preprocessor(data[i:i + 200])
  return preprocessed


class Texts(object):

  def __init__(self, data_dir):
    self.texts = collections.defaultdict(dict)
    print("Preprocesing data")
    for subset in SUBSETS:
      print(subset)
      for filename in tqdm.tqdm(sorted(
          glob.glob(f"{data_dir}/{subset}/*"))):
        with open(filename, "r") as f:
          obj = json.load(f)
        review_id = obj["metadata"]["review_id"]
        review_sentences, review_len = self.process_sentences(
            obj["review_sentences"])
        rebuttal_sentences, _ = self.process_sentences(
            obj["rebuttal_sentences"])
        alignment_map = self._get_alignment_map(obj["rebuttal_sentences"],
                                                review_len)
        self.texts[subset][review_id] = TextInfo(
            review_sentences,
            rebuttal_sentences,
            alignment_map,
            review_len,
        )
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
        preprocessor: batch_preprocess(sentence_texts, PREP[preprocessor])
        for preprocessor in Preprocessors.ALL
    }, len(sentence_texts)

  def _get_alignment_map(self, rebuttal_sentences, num_review_sentences):
    map_starter = np.zeros([len(rebuttal_sentences), num_review_sentences])
    for reb_i, rebuttal_sentence in enumerate(rebuttal_sentences):
      align_type, indices = rebuttal_sentence["alignment"]
      if indices is None:
        continue
      else:
        for rev_i in indices:
          map_starter[reb_i][rev_i] += 1
    return map_starter

  def build_overall_corpus(self):
    self.corpora = collections.defaultdict(list)
    offset = 0
    self.offset_map = {}
    for subset, example_map in self.texts.items():
      for review_id in sorted(example_map.keys()):
        review_sentences, _, _, review_len = example_map[review_id]
        self.offset_map[review_id] = (offset, offset + review_len)
        for preprocessor, tokenized in review_sentences.items():
          self.corpora[preprocessor] += tokenized
        offset += review_len

  def score(self, overall_models):
    self.scores = collections.defaultdict(dict)
    for subset, reviews in self.texts.items():
      print(subset)
      for review_id, info in tqdm.tqdm(reviews.items()):
        review_sentences, rebuttal_sentences, alignment_map, _ = info
        this_review_scores = {"discrete": alignment_map}
        for preprocessor in Preprocessors.ALL:
          mini_model = BM25Okapi(review_sentences[preprocessor])
          big_scores = []
          small_scores = []
          for i, query in enumerate(rebuttal_sentences[preprocessor]):
            offsets = self.offset_map[review_id]
            big_scores.append(overall_models[preprocessor].get_scores(query)
                              [offsets[0]:offsets[1]])
            small_scores.append(mini_model.get_scores(query))
          this_review_scores.update({
            "_".join([preprocessor, Corpus.REVIEW]): np.array(small_scores),
            "_".join([preprocessor, Corpus.FULL]): np.array(big_scores),
          })
        self.scores[subset][review_id] = this_review_scores



def main():
  args = parser.parse_args()
  texts = Texts(args.data_dir)
  with open(args.score_file, "wb") as f:
    pickle.dump(texts, f)


if __name__ == "__main__":
  main()
