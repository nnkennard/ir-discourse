import argparse
import collections
import ird_lib
import pickle

parser = argparse.ArgumentParser(
    description="Create examples for similarity training")
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    help="path to data file containing score jsons",
)

NO_MATCH = "NO_MATCH"


def main():
  args = parser.parse_args()

  with open('processed_data/mini_ape/weaksup/mini_ape_review_ids.txt', 'r') as f:
    mini_ape_review_ids = [line.strip() for line in f]

  print(mini_ape_review_ids)

  with open(f"{args.data_dir}/scores.pkl", "rb") as f:
    scores = pickle.load(f)

  example_counter = collections.Counter()

  for subset, subset_scores in scores.items():
    raw_text_map = ird_lib.load_examples(args.data_dir, None, subset)
    examples = []
    for review_id, info in subset_scores.items():
      if review_id not in mini_ape_review_ids:
        continue
      else:
        example_counter[subset] += 1
      review_sentences = raw_text_map[review_id]["review_lines"]
      rebuttal_sentences = raw_text_map[review_id]["rebuttal_lines"]
      alignment_map = info["discrete"]
      for reb_i, reb_sentence in enumerate(rebuttal_sentences):
        for rev_i, rev_sentence in enumerate(review_sentences):
          examples.append(
              f"{reb_sentence}\t{rev_sentence}\t{alignment_map[reb_i][rev_i]}")
        if sum(alignment_map[reb_i]):
          no_match_score = 1.0
        else:
          no_match_score = 0.0
        examples.append(f"{reb_sentence}\t{NO_MATCH}\t{no_match_score}")
    #with open(f"{args.data_dir}/similarity/{subset}.txt", "w") as f:
    with open(f"processed_data/mini_ape/similarity/{subset}.txt", "w") as f:
      f.write("\n".join(examples))

  print(example_counter)

if __name__ == "__main__":
  main()
