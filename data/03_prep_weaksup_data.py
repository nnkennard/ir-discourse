from ird_lib import Texts, TextInfo
import collections
import pickle
import random
import tqdm

Example = collections.namedtuple("Example", "text label".split())


def build_text(rebuttal_sentence, review_sentence_1, review_sentence_2):
  return "[CLS] " + " [SEP] ".join([rebuttal_sentence, review_sentence_1,
  review_sentence_2])


def get_raw_review_text(info):
  return get_raw_text(info, "review")


def get_raw_rebuttal_text(info):
  return get_raw_text(info, "rebuttal")


def get_raw_text(info, text_type):
  return info._asdict()[text_type + "_sentences"]["raw"]


def sample_indices(num_review_sentences, num_rebuttal_sentences):
  pool = []
  for i in range(num_rebuttal_sentences):
    for j in range(num_review_sentences):
      for k in range(num_review_sentences):
        if j == k:
          continue
        pool.append((i, j, k))

  num_samples = num_review_sentences * num_rebuttal_sentences
  return random.sample(pool, num_samples)


def get_example_texts(review_sentences, rebuttal_sentences, sampled_indices):
  return [
      build_text(
          rebuttal_sentences[reb_i],
          review_sentences[rev_j],
          review_sentences[rev_k]) for reb_i, rev_j, rev_k in sampled_indices]


def main():
  with open("data/bm25_scores.pkl", "rb") as f:
    texts = pickle.load(f)

  example_map = collections.defaultdict(lambda: collections.defaultdict(list))
  for subset, subset_texts in texts.texts.items():
    label_map = collections.defaultdict(list)
    example_texts = []
    print(subset)
    for review_id, info in tqdm.tqdm(subset_texts.items()):
      scores = texts.scores[subset][review_id]

      review_sentences = get_raw_review_text(info)
      rebuttal_sentences = get_raw_rebuttal_text(info)

      sampled_indices = sample_indices(len(review_sentences),
                                       len(rebuttal_sentences))

      review_example_texts = get_example_texts(review_sentences, rebuttal_sentences,
                                        sampled_indices)

      review_label_map = collections.defaultdict(list)
      for key, score_matrix in scores.items():
        if key == 'discrete':
          continue
        for reb_i, rev_j, rev_k in sampled_indices:
          score_j = score_matrix[reb_i][rev_j]
          score_k = score_matrix[reb_i][rev_k]
          label = 1 if score_j > score_k else 0
          review_label_map[key].append(label)
      for key, labels in review_label_map.items():
        label_map[key] += labels
      example_texts += review_example_texts
    for key, labels in label_map.items():
      with open(f"data/weaksup/examples_{subset}_{key}.txt", 'w') as f:
        f.write("\n".join(str(i) for i in labels))
    with open(f"data/weaksup/examples_{subset}_text.txt", 'w') as f:
      f.write("\n".join(example_texts))


if __name__ == "__main__":
  main()
