#mkdir -p original_data/disapere
#mkdir -p original_data/ape
#mkdir -p processed_data/disapere
#mkdir -p processed_data/ape

## Download and prep DISAPERE data

#curl -O https://raw.githubusercontent.com/nnkennard/DISAPERE/main/DISAPERE.tar.gz
#tar -zxf DISAPERE.tar.gz
#mv DISAPERE/final_dataset/* original_data/disapere
#rm -r DISAPERE
#rm DISAPERE.tar.gz

#python 00_prep_disapere_data.py -d original_data/disapere -o processed_data/disapere

# Download APE data
curl -O https://raw.githubusercontent.com/LiyingCheng95/ArgumentPairExtraction/master/data/rr-passage/train.txt
curl -O https://raw.githubusercontent.com/LiyingCheng95/ArgumentPairExtraction/master/data/rr-passage/dev.txt
curl -O https://raw.githubusercontent.com/LiyingCheng95/ArgumentPairExtraction/master/data/rr-passage/test.txt
mv train.txt test.txt dev.txt original_data/ape
python 00_prep_ape_data.py -d original_data/ape -o processed_data/ape
exit

# Calculate weak supervision scores

# Prepare weak supervision data

# Prepare similarity data
