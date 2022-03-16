mkdir -p original_data/disapere
mkdir -p original_data/ape
mkdir -p processed_data/disapere
mkdir -p processed_data/ape

## Download and prep DISAPERE data

#curl -O https://raw.githubusercontent.com/nnkennard/DISAPERE/main/DISAPERE.tar.gz
#tar -zxf DISAPERE.tar.gz
#mv DISAPERE/final_dataset/* original_data/disapere
#rm -r DISAPERE
#rm DISAPERE.tar.gz

python 00_prep_disapere_data.py -d original_data/disapere -o processed_data/disapere

# Download APE data
#curl -O https://raw.githubusercontent.com/LiyingCheng95/ArgumentPairExtraction/master/data/rr-passage/train.txt
#curl -O https://raw.githubusercontent.com/LiyingCheng95/ArgumentPairExtraction/master/data/rr-passage/dev.txt
#curl -O https://raw.githubusercontent.com/LiyingCheng95/ArgumentPairExtraction/master/data/rr-passage/test.txt
#mv train.txt test.txt dev.txt original_data/ape
python 00_prep_ape_data.py -d original_data/ape -o processed_data/ape

# Calculate weak supervision scores
python 01_get_bm25_scores.py -d processed_data/disapere -n disapere
python 01_get_bm25_scores.py -d processed_data/ape -n ape

# Prepare weak supervision data
mkdir processed_data/disapere/weaksup
mkdir processed_data/ape/weaksup
python 03_prep_weaksup_data.py -d processed_data/disapere
python 03_prep_weaksup_data.py -d processed_data/ape


# Prepare similarity data
