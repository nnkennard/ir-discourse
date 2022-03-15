from ird_lib import Texts, TextInfo
import collections
import pickle

no_match_tokens = ["no", "_", "match"]
NO_MATCH = "NO_MATCH"

Example = collections.namedtuple(
    "Example", "rebuttal_sentence review_sentence label".split())


def main():
  with open("data/bm25_scores.pkl", "rb") as f:
    texts = pickle.load(f)

  example_map = {}
  for subset, subset_texts in texts.texts.items():
    examples = []
    for review_id, info in subset_texts.items():
      for reb_i, reb_sentence in enumerate(info.rebuttal_sentences["raw"]):
        for rev_i, rev_sentence in enumerate(info.review_sentences["raw"]):
          examples.append(
              Example(reb_sentence, rev_sentence,
                      info.alignment_map[reb_i][rev_i]))
        if sum(info.alignment_map[reb_i]):
          no_match_score = 1.0
        else:
          no_match_score = 0.0
        examples.append(Example(reb_sentence, NO_MATCH, no_match_score))
    example_map[subset] = [x._asdict() for x in examples]

  with open("data/similarity_scores.pkl", "wb") as f:
    pickle.dump(example_map, f)


if __name__ == "__main__":
  main()
