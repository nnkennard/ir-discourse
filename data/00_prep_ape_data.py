import collections
import json
import numpy as np

import ird_lib

REVIEW, REPLY = "Review", "Reply"


def get_subset_lines(subset):
  split_by_paper = collections.defaultdict(list)
  with open(f"original_data/ape/{subset}.txt", "r") as f:
    for line in f:
      if not line.strip():
        continue
      paper_number = line.strip().split()[-1]
      split_by_paper[paper_number].append(line)
  return split_by_paper


def bio_tag_to_mapping_index(bio_tag):
  if bio_tag == "O":
    return None
  else:
    return bio_tag.split("-")[1]


def build_example(review_lines, reply_lines, identifier):
  review_index_map = collections.defaultdict(list)

  review_text_lines = []
  for i, line in enumerate(review_lines):
    text, _, bio, _, _ = line.strip().split("\t")
    review_text_lines.append(text)
    maybe_mapping_index = bio_tag_to_mapping_index(bio)
    if maybe_mapping_index is None:
      continue
    else:
      review_index_map[maybe_mapping_index].append(i)

  reply_to_review_map = np.zeros([len(reply_lines), len(review_lines)],
  dtype=np.int64)
  reply_text_lines = []
  for rep_i, line in enumerate(reply_lines):
    text, _, bio, _, _ = line.strip().split("\t")
    reply_text_lines.append(text)
    maybe_mapping_index = bio_tag_to_mapping_index(bio)
    if maybe_mapping_index is None:
      continue
    else:
      for rev_i in review_index_map[maybe_mapping_index]:
        reply_to_review_map[rep_i][rev_i] = 1

  return ird_lib.Example(review_text_lines, reply_text_lines,
            reply_to_review_map.tolist(), identifier)


def build_examples(lines, paper_number):
  status = REPLY
  index = -1
  texts = {
      REVIEW: collections.defaultdict(list),
      REPLY: collections.defaultdict(list),
  }
  for line in lines:
    text_type = line.split("\t")[-2]
    if not text_type == status:
      status = text_type
      if text_type == REVIEW:
        index += 1
    texts[status][index].append(line)

  assert len(texts[REVIEW]) == len(texts[REPLY])
  return [
      build_example(review_lines, texts[REPLY][index],
                    f"ape_{paper_number}_{index}")
      for index, review_lines in texts[REVIEW].items()
  ]


def process_papers(split_by_paper):
  examples = []
  for paper_number, lines in split_by_paper.items():
    examples += build_examples(lines, paper_number)
  return examples
def main():
  data_path = "original_data/ape/"
  for subset in "train dev test".split():
    split_by_paper = get_subset_lines(subset)
    examples = process_papers(split_by_paper)
    ird_lib.dump_raw_text_to_file(examples, data_path, "ape", subset)


if __name__ == "__main__":
  main()
