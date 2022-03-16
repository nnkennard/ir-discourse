# Download and prep DISAPERE data

python 00_prep_disapere_data.py

# Download APE data
curl -O https://raw.githubusercontent.com/LiyingCheng95/ArgumentPairExtraction/master/data/rr-passage/train.txt
curl -O https://raw.githubusercontent.com/LiyingCheng95/ArgumentPairExtraction/master/data/rr-passage/dev.txt
curl -O https://raw.githubusercontent.com/LiyingCheng95/ArgumentPairExtraction/master/data/rr-passage/test.txt
python 00_prep_ape_data.py

# Calculate weak supervision scores

# Prepare weak supervision data

# Prepare similarity data
