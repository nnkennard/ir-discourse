#!/bin/bash
#SBATCH --job-name=serial_job_test    # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=nnayak@cs.umass.edu     # Where to send mail	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=47GB                     # Job memory request
#SBATCH --output=serial_test_%j.log   # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx8000-long

pwd; hostname; date

python  train.py -d ../data/processed_data/disapere/weaksup/ -c full -t bert_stop -n disapere

date
