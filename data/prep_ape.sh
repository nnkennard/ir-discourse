#!/bin/bash
#SBATCH --job-name=ape_prep    # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=nnayak@cs.umass.edu     # Where to send mail	
#SBATCH --mem=75GB                     # Job memory request
#SBATCH --output=ape_prep_%j.log   # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=2080ti-long


#python 00_prep_ape_data.py -d original_data/ape -o processed_data/ape
#python 01_get_bm25_scores.py -d processed_data/ape -n ape
#mkdir processed_data/ape/weaksup
python 03_prep_weaksup_data.py -d processed_data/ape/
