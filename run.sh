#!/bin/bash
set -e
echo "CUDA_VISIBLE_DEVICES= $CUDA_VISIBLE_DEVICES"
nvidia-smi

# point it to your conda environment
source activate torch1.6

# TO DEFAULT OUTPUT DIRS, TINY (DEBUG) RUN
hypers="--tiny 1 --resume_result_json 0"
override_runname="" # runname: default  -> tb/default and output/default

git log --graph --full-history --all --oneline | head -n 15
git status

loadpath="" # empty to resume from local phase 1 VAE pretrain; set to resume another phase1.
/usr/bin/time -v python main.py $override_runname $loadpath $hypers --phase 1