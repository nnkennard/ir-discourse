import glob
import json
import numpy as np
import tqdm

import ird_lib

def get_alignment_map(rebuttal_sentences, num_review_sentences):
    map_starter = np.zeros(
        [len(rebuttal_sentences), num_review_sentences], dtype=np.int32
    )
    for reb_i, rebuttal_sentence in enumerate(rebuttal_sentences):
        align_type, indices = rebuttal_sentence["alignment"]
        if indices is None:
            continue
        else:
            for rev_i in indices:
                map_starter[reb_i][rev_i] = 1.0
    return map_starter


def process_sentences(sentences):
    return [x["text"] for x in sentences]


def main():
    data_dir = "../../DISAPERE/DISAPERE/final_dataset/"
    print("Preprocesing data")
    for subset in ird_lib.SUBSETS:
        print(subset)
        examples = []
        for filename in tqdm.tqdm(sorted(glob.glob(f"{data_dir}/{subset}/*"))):
            with open(filename, "r") as f:
                obj = json.load(f)
            review_id = obj["metadata"]["review_id"]
            review_sentences = process_sentences(obj["review_sentences"])
            rebuttal_sentences = process_sentences(obj["rebuttal_sentences"])
            alignment_map = get_alignment_map(
                obj["rebuttal_sentences"], len(review_sentences)
            )
            examples.append(ird_lib.Example(
                review_sentences,
                rebuttal_sentences,
                alignment_map.tolist(),
                f"disapere_{review_id}",
            ))
        ird_lib.dump_raw_text_to_file(examples, "processed_data", "disapere",
        subset)


if __name__ == "__main__":
    main()
