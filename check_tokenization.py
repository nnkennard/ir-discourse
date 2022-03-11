import collections
import glob
import json
import stanza
import tqdm

from transformers import BertTokenizer

STANZA_PIPELINE = stanza.Pipeline('en',
                                  processors='tokenize,lemma',
                                  tokenize_no_ssplit=True)


def main():
  corpus = []
  for filename in tqdm.tqdm(
      glob.glob("../DISAPERE/DISAPERE/final_dataset/train/*")):
    with open(filename, 'r') as f:
      obj = json.load(f)
      for key_pre in ["review", "rebuttal"]:
        corpus += [x['text'] for x in obj[key_pre + "_sentences"]]

  bert_counter = collections.Counter()
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  print("BERT tokenization")
  for sentence in tqdm.tqdm(corpus):
    bert_counter.update(tokenizer.tokenize(sentence))

  stanza_counter = collections.Counter()
  print("Stanza tokenization")
  for sentence in tqdm.tqdm(corpus):
    tokenized = STANZA_PIPELINE(sentence)
    sentence, = tokenized.sentences
    stanza_counter.update(
        [token.to_dict()[0]['lemma'] for token in sentence.tokens])

  with open("vocabs.json", 'w') as f:
    json.dump({"stanza_vocab": stanza_counter, "bert_vocab": bert_counter}, f)


if __name__ == "__main__":
  main()
