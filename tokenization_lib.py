import json
import stanza
from transformers import BertTokenizer

from ird_lib import Preprocessors


# ===== Preprocessing =========================================================

with open("nltk_stopwords.json", "r") as f:
  STOPWORDS = json.load(f)

BERT_TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")


def bert_preprocess_stop(sentences):
  return [[tok
           for tok in BERT_TOKENIZER.tokenize(sentence)
           if tok not in STOPWORDS]
          for sentence in sentences]



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
    Preprocessors.STANZA:
        stanza_preprocess,
    Preprocessors.BERT_STOP:
        bert_preprocess_stop,
}
