import collections
import pickle
import ird_lib


def compare_two_score_lists(score_list_1, score_list_2):
  matches = []
  for (i1, j1, k1, v1), (i2, j2, k2, v2) in zip(score_list_1, score_list_2):
    assert i1 == i2 and j1 == j2 and k1 == k2
    matches.append(v1 == v2)
  return len([i for i in matches if i]), len(matches)


def main():
  with open("data/bm25_scores.pkl", "rb") as f:
    texts = pickle.load(f)

  numerator_diff_counter = collections.Counter()
  denominator_diff_counter = collections.Counter()

  for subset, subset_texts in texts.texts.items():
    if not subset == 'train':
      continue
    for review_id, info in subset_texts.items():
      scores = texts.scores[subset][review_id]
      score_diffs = collections.defaultdict(list)
      for corpus_type in "review full".split():
        for preprocessor, rebuttal_sentences in info.rebuttal_sentences.items():
          if preprocessor == 'raw':
            continue
          key = preprocessor + "_" + corpus_type
          score_matrix = scores[key]
          for reb_i in range(len(rebuttal_sentences)):
            for rev_j in range(info.review_len):
              for rev_k in range(info.review_len):
                score_j = score_matrix[reb_i][rev_j]
                score_k = score_matrix[reb_i][rev_k]
                score_diffs[key].append(
                    (reb_i, rev_j, rev_k, score_j > score_k))
      for key1, diffs1 in score_diffs.items():
        for key2, diffs2 in score_diffs.items():
          a, b = compare_two_score_lists(diffs1, diffs2)
          numerator_diff_counter[(key1, key2)] += a
          denominator_diff_counter[(key1, key2)] += b

  for k, v in numerator_diff_counter.items():
    print(k, v / denominator_diff_counter[k])


if __name__ == "__main__":
  main()
