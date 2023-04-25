#!/bin/bash 
#SBATCH -J whisper          # Your job name to be displayed by squeue
#SBATCH -o /data/sls/scratch/clai24/lexicon/exp/slurm_dump/whisper_%j.out   # path to write stdout, %j will be jobID 
#SBATCH -e /data/sls/scratch/clai24/lexicon/exp/slurm_dump/whisper_%j.err   # path to write stderr, %j will be jobID 
#SBATCH --qos=regular 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 
#SBATCH --partition=2080
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00 
#SBATCH --mem=50G

## Set the python environment you want to use for your code 
PYTHON_VIRTUAL_ENVIRONMENT=yung-sung
CONDA_ROOT=/data/sls/scratch/clai24/anaconda3
source ${CONDA_ROOT}/etc/profile.d/conda.sh 
conda activate $PYTHON_VIRTUAL_ENVIRONMENT 

SPLIT=$1
python data_utils/read_train_mined_audio.py \
    --split $SPLIT \
    --nprocess $2 \
    --process_id $3 \
    --lan_pair s2u_en-es \
    --data_root /data/sls/scratch/clai24/lexicon/exp/hubert_kmeans/ \
    --save_root /data/sls/temp/clai24/data/speech_matrix/speech_to_unit/mfa_s2u_manifests/
