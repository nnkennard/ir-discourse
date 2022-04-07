import argparse
import collections
import json
import pickle
import numpy as np

from rank_metrics import average_precision

parser = argparse.ArgumentParser(description="Create examples for WS training")
parser.add_argument(
    "-t",
    "--tokenization",
    type=str,
    help="type of tokenization",
)
parser.add_argument(
    "-c",
    "--corpus_type",
    type=str,
    help="type of corpus for calculating scores (full dataset or review)",
)


def get_top_rr(is_ranked_sentence_relevant):
  for i, is_relevant in enumerate(is_ranked_sentence_relevant):
    if is_relevant:
      return 1 / (1 + i)

def count_ties(win_map):
  max_wins = max(win_map.keys())
  top_tie = len(win_map[max_wins]) > 1
  tie_count = 0
  non_tie_count = 0
  for win_count, indices in win_map.items():
    if len(indices) == 1:
      non_tie_count += 1
    else:
      tie_count += len(indices)

  return top_tie, tie_count, non_tie_count

def get_ap_from_wins(wins, helper):
  scores = helper["bm25_scores"]
  relevant_indices = helper["actual_aligned_indices"]
  if not relevant_indices:
    return None
  win_count_to_indices = collections.OrderedDict()
  num_review_sentences = len(scores)

  missing_indices = set(range(num_review_sentences))
  for review_index, num_wins in wins.most_common():
    missing_indices.remove(int(review_index))
    if num_wins in win_count_to_indices:
      win_count_to_indices[num_wins].append(int(review_index))
    else:
      win_count_to_indices[num_wins] = [int(review_index)]
  win_count_to_indices[0] = list(sorted(missing_indices))

  top_tie, total_ties, total_non_ties = count_ties(win_count_to_indices)

  is_ranked_sentence_relevant = [None] * num_review_sentences
  rank_counter = 0
  for win_count, indices in win_count_to_indices.items():
    # print(win_count, indices)
    ordered_indices = list(reversed(sorted([scores[x], x] for x in indices)))
    for a, index in ordered_indices:
      is_ranked_sentence_relevant[rank_counter] = int(index in relevant_indices)
      rank_counter += 1

  # print("Is ranked sentence relevant", is_ranked_sentence_relevant)

  return {
            "ap":average_precision(is_ranked_sentence_relevant),
            "rr": get_top_rr(is_ranked_sentence_relevant),
            "top_tie": top_tie,
            "tie_count": total_ties,
            "non_tie_count": total_non_ties}


def get_metric_helper(data_dir, subset_key):
  with open(f"{data_dir}/examples_{subset_key}_helper.json", "r") as f:
    return json.load(f)


def calculate_mrr(results, helper, key):
  identifiers = []
  preds = []
  by_review_id = collections.defaultdict(lambda: collections.defaultdict(dict))
  for i, p in results:
    review_id, reb_i, rev_j, rev_k = i.split("|")
    by_review_id[review_id][reb_i][(rev_j, rev_k)] = p
  aps = []
  rrs = []
  tie_count = 0
  non_tie_count = 0
  top_ties = []
  for review_id, review_preds in by_review_id.items():
    for rebuttal_index, this_reb_preds in review_preds.items():
      wins = collections.Counter()
      for (rev_j, rev_k), v in this_reb_preds.items():
        if v:
          wins[rev_j] += 1
        else:
          wins[rev_k] += 1
      metrics = get_ap_from_wins(
          wins, helper[review_id][key][rebuttal_index])
      if metrics is not None:
        aps.append(metrics['ap'])
        rrs.append(metrics['rr'])
        top_ties.append(metrics['top_tie'])
        tie_count += metrics['tie_count']
        non_tie_count += metrics['non_tie_count']

  return {
    "MAP": np.mean(aps),
    "MRR": np.mean(rrs),
    "top_ties": np.mean(top_ties),
    "total_ties": tie_count / (tie_count + non_tie_count)}


def main():

  args = parser.parse_args()

  key = f"{args.tokenization}|{args.corpus_type}"

  with open(f"outputs/test_results_disapere_{key}.pkl", "rb") as f:
  #with open(f"outputs/test_results_mini_ape_{key}.pkl", "rb") as f:
    results = pickle.load(f)["results"]

  #helper = get_metric_helper("../data/processed_data/mini_ape/weaksup/",
  helper = get_metric_helper("../data/processed_data/disapere/weaksup/",
                             "test_all")

  for k,v in calculate_mrr(results, helper, key).items():
    print(f"WS\t{key}\t{k}\t{v}")


if __name__ == "__main__":
  main()
