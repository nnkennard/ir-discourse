import argparse
import pickle
import ird_lib
import pickle

parser = argparse.ArgumentParser(description="prepare CSVs for ws training")
parser.add_argument(
    "-d",
    "--data_dir",
    default="../DISAPERE/DISAPERE/final_dataset/",
    type=str,
    help="path to data file containing score jsons",
)
parser.add_argument(
    "-s",
    "--score_file",
    default="data/bm25_scores.pkl",
    type=str,
    help="path to data file containing score jsons",
)
# =============================================================================


def main():
  args = parser.parse_args()
  texts = ird_lib.Texts(args.data_dir)
  with open(args.score_file, "wb") as f:
    pickle.dump(texts, f)


if __name__ == "__main__":
  main()
