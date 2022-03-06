import argparse
import collections
import glob
import json
import nltk
import numpy as np
import pickle
import tqdm

from rank_bm25 import BM25Okapi
from transformers import BertTokenizer

parser = argparse.ArgumentParser(description='prepare CSVs for ws training')
parser.add_argument('-d',
                    '--data_dir',
                    default="../DISAPERE/DISAPERE/final_dataset/",
                    type=str,
                    help='path to data file containing score jsons')
parser.add_argument('-s',
                    '--score_file',
                    default="bm25_scores.pkl",
                    type=str,
                    help='path to data file containing score jsons')

SUBSETS = "train dev test".split()

class Preprocessors(object):
  NLTK = "nltk"
  BERT = "bert"
  ALL = [NLTK, BERT]

class Corpus(object):
  REVIEW = "review"
  FULL = "full"
  ALL = [REVIEW, FULL]


BERT_TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
def bert_preprocess(sentence):
  return BERT_TOKENIZER.tokenize(sentence)

STEMMER = nltk.stem.porter.PorterStemmer()
STOPWORDS = nltk.corpus.stopwords.words('english')

def nltk_preprocess(sentence):
  return [
      STEMMER.stem(word).lower()
      for word in nltk.word_tokenize(sentence)
      if word.lower() not in STOPWORDS
  ]

PREP = {
  Preprocessors.NLTK: nltk_preprocess,
  Preprocessors.BERT: bert_preprocess
}

print(PREP)


class Data(object):

  def __init__(self, data_dir):
    self.reviews = collections.defaultdict(collections.OrderedDict)
    self.rebuttals = collections.defaultdict(collections.OrderedDict)
    self.labels = collections.defaultdict(collections.OrderedDict)

    for subset in SUBSETS:
      for filename in sorted(glob.glob(data_dir + "/" + subset + "/*")):
        with open(filename, 'r') as f:
          obj = json.load(f)
        review_id = obj["metadata"]["review_id"]
        self.reviews[subset][review_id] = [
            x["text"] for x in obj["review_sentences"]
        ]
        self.rebuttals[subset][review_id] = [
            x["text"] for x in obj["rebuttal_sentences"]
        ]
        self.labels[subset][review_id] = [
            x["alignment"][1] for x in obj["rebuttal_sentences"]
        ]
    self.overall_corpus, self.offset_maps = self.build_overall_corpus()

  def build_overall_corpus(self):
    offset_maps = collections.defaultdict(dict)
    corpus = []
    for subset in SUBSETS:
      for review_id, review_sentences in self.reviews[subset].items():
        offset_maps[subset][review_id] = (len(corpus), len(corpus) + len(review_sentences))
        corpus += review_sentences
    return corpus, offset_maps


def main():

  args = parser.parse_args()

  # Get reviews and rebuttals
  data = Data(args.data_dir)

  # Get threshold

  # Get scores
  
  print("Building corpora")
  overall_corpora = {
    preprocessor: [PREP[preprocessor](sent) for sent in data.overall_corpus] for preprocessor in Preprocessors.ALL
  }

  print("Building overall models")
  overall_models = {
    preprocessor: BM25Okapi(overall_corpora[preprocessor]) for preprocessor in Preprocessors.ALL
  }

  scores = {}
  for review_id, review_sentences in tqdm.tqdm(data.reviews['test'].items()):
    rebuttal_sentences = data.rebuttals['test'][review_id]
    score_maps = collections.defaultdict(dict)
    for preprocessor in Preprocessors.ALL:
      mini_model = BM25Okapi(
        [PREP[preprocessor](sent) for sent in review_sentences])
      big_scores = []
      small_scores = []
      for i, query in enumerate(data.rebuttals['test'][review_id]):
        offsets = data.offset_maps['test'][review_id]
        big_scores.append(overall_models[preprocessor].get_scores(
            query)[offsets[0]:offsets[1]])
        small_scores.append(mini_model.get_scores(query))
      score_maps[preprocessor][Corpus.REVIEW] = np.array(small_scores)
      score_maps[preprocessor][Corpus.FULL] = np.array(big_scores)
    scores[review_id] = score_maps, data.labels['test'][review_id]
    if len(scores) > 10:
      break
    

  with open(args.score_file, 'wb') as f:
    pickle.dump(scores, f)


if __name__ == "__main__":
  main()
