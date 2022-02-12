import argparse
import collections
import glob
import json
import nltk

parser = argparse.ArgumentParser(description='prepare CSVs for ws training')
parser.add_argument('-d',
                    '--data_dir',
                    default="../DISAPERE/DISAPERE/final_dataset/",
                    type=str,
                    help='path to data file containing score jsons')


class Preprocessors(object):
  NLTK = "nltk"
  BERT = "bert"
  ALL = [NLTK]


class BERTPreprocessor(object):

  def __init__(self):
    pass

  def preprocess(self, reviews, rebuttals):
    pass


class NLTKPreprocessor(object):

  def __init__(self):
    self.stemmer = nltk.stem.porter.PorterStemmer()
    self.stopwords = nltk.corpus.stopwords.words('english')

  def preprocess(self, reviews, rebuttals):
    return ([self._preprocess_sentence(sent) for sent in reviews],
            [self._preprocess_sentence(sent) for sent in rebuttals])

  def _preprocess_sentence(self, sentence):
    return [
        self.stemmer.stem(word).lower()
        for word in nltk.word_tokenize(sentence)
        if word.lower() not in self.stopwords
    ]


class Data(object):

  def __init__(self, data_dir, preprocessor):
    self.reviews = collections.defaultdict(dict)
    self.rebuttals = collections.defaultdict(dict)
    self.labels = collections.defaultdict(dict)

    for subset in "train dev test".split():
      for filename in glob.glob(data_dir + "/" + subset + "/*"):
        with open(filename, 'r') as f:
          obj = json.load(f)
        review_id = obj["metadata"]["review_id"]
        self.reviews[subset][review_id] = [
            preprocessor.preprocessor(x["text"]) for x in obj["review_sentences"]
        ]
        self.rebuttals[subset][review_id] = [
            preprocessor(x["text"]) for x in obj["rebuttal_sentences"]
        ]
        self.labels[subset][review_id] = [
            x["alignment"][1] for x in obj["rebuttal_sentences"]
        ]


def get_predictions(data, model_type):
  if 'all' in model_type:
    overall_corpus = sum([
        sum(sentence_map.values(), [])
        for sentence_map in data.reviews.values()
    ], [])
    overall_model = BM25Okapi(overall_corpus)
  for review_id, rebuttal_sentences in data.rebuttals[test].items():
    if 'all' in model_type:
      model = overall_model
    else:
      model = BM25Okapi(data.reviews[test][review_id])
    scores = overall_model(query)
    print(scores)


def main():

  args = parser.parse_args()

  # Get reviews and rebuttals
  data = Data(args.data_dir, NLTKPreprocessor())

  get_predictions(data, "bm25_all")

  # Build BM 25 model with

  pass


if __name__ == "__main__":
  main()
