# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.10.0
* Python version >= 3.8

```bash 
# install and activate conda 
conda create -n lexlearner python=3.9
conda activate lexlearner

# install torch (adjust the cuda toolkit version)
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch

# install fairseq 
cd fairseq-ust/
pip install --editable ./
pip install tensorboardX pandas datasets

# install additional requirements 
pip install -r requirements.txt
```

* Satori installation is tricker, and we need to make the following changes instead:

```bash 
# Satori version 
# install and activate conda 
conda create -n lexlearner python=3.8
conda activate lexlearner

# prepend Satori specific channeles
conda config --prepend channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/
conda config --prepend channels https://opence.mit.edu
conda config --set channel_priority true  

# check Open-CE has the highest channel priority 
conda config --show channels

# install Satori default MLDL packages 
conda install powerai # this is not required 

# install torch. For Satori, remove torchaudio as it is not supported. We also do not want to find packages from the defauly PyTorch channel. 
conda install pytorch torchvision 
pip install "pillow<7" 

# install fairseq. We need to make changes to `setup.py` as the following packages caused installation errors. 
# this also implied that we need to run some pre-processing that involves sklearn in the SLS server. 
cd fairseq-ust/
# "torchaudio>=0.8.0" --> ""
# "scikit-learn" --> ""
# "editdistance" --> "editdistance==0.5.2"
# "pyarrow" --> ""
# "sacrebleu @ git+https://github.com/mjpost/sacrebleu.git@master" --> "pip install sacrebleu==1.5.0"

pip install --editable ./
pip install tensorboardX pandas

# setup torchaudio. Necessary for fbank extraction
git clone https://github.com/pytorch/audio.git
# then add this to your ~/.bashrc 
export PYTHONPATH="${PYTHONPATH}:/nobackup/users/clai24/tools/audio"
```

# Download necessary pre-trained models 

```bash 
# Download Pre-trained HuBERTs
HUBERT_MODEL_DIR=/data/sls/temp/clai24/pretrained-models/mHuBERT
cd $HUBERT_MODEL_DIR
wget https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_roman_it3.pt
wget https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_slavic_it3.pt
wget https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_germanic_it3.pt
wget https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_uralic_it3.pt
wget https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3.pt

wget https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_it_it3_L11_km800.bin
wget https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_pt_it3_L11_km1000.bin
wget https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_ro_it3_L11_km1000.bin
wget https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_cs_it3_L11_km1000.bin
wget https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_hr_it3_L11_km1000.bin
wget https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_lt_it3_L11_km1000.bin
wget https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_pl_it3_L11_km1000.bin
wget https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_sk_it3_L11_km1000.bin
wget https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_sl_it3_L11_km1000.bin
wget https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_de_it3_L11_km1000.bin
wget https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_nl_it3_L11_km1000.bin
wget https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_et_it3_L11_km1000.bin
wget https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_fi_it3_L11_km1000.bin
wget https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_hu_it3_L11_km1000.bin
wget https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin

# Download Pre-trained Unit HiFi-GAN 
UNIT_VOCODER_DIR=/data/sls/temp/clai24/data/speech_matrix/unit_vocoder
cd $UNIT_VOCODER_DIR
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_de.pt
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_de.json
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_nl.pt
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_nl.json
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_fi.pt
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_fi.json
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_hu.pt
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_hu.json
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_et.pt
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_et.json
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_it.pt
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_it.json
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_pt.pt
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_pt.json
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_ro.pt
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_ro.json
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_cs.pt
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_cs.json
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_pl.pt
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_pl.json
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_hr.pt
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_hr.json
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_lt.pt
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_lt.json
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_sk.pt
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_sk.json
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_sl.pt
wget https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_sl.json
wget -O vocoder_en.pt https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000
wget -O config_en.json https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json
wget -O vocoder_es.pt https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10/g_00500000
wget -O config_es.json https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10/config.json
wget -O vocoder_fr.pt https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_fr_css10/g_00500000
wget -O config_fr.json https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_fr_css10/config.json
```

# Speech translation data setup with fairseq-ust

For quick start, checkout our prepared `es-en` data at `data/speech_matrix/speech_to_unit/s2u_manifests/es-en/`.

```bash 
cd fairseq-ust/examples/speech_matrix/
SAVE_ROOT=/data/sls/temp/clai24/data/speech_matrix/speech_to_unit
export PYTHONPATH="${PYTHONPATH}:${PWD}:${PWD}/../../"
conda activate lexlearner

# Run the following commands *sequentially*

# SpeechMatrix: Speech-to-Speech Alignments (it takes 1-3 days to download all datasets)
# Re-Running this to ensure aligned_speech are downlaoded correctly on Satori 
python mined_train_sets/download_mined_data.py --save-root ${SAVE_ROOT}

# download the prepared audios for VoxPopuli valid and test data.
wget --directory-prefix=${SAVE_ROOT}/audios https://dl.fbaipublicfiles.com/speech_matrix/audios/valid_test_vp_aud.zip 

# SpeechMatrix: Speech-to-Unit Data
python mined_train_sets/download_speech_to_unit.py --save-root ${SAVE_ROOT}

# SpeechMatrix: Reproduce Bilingual Train Data
python3 speech_to_speech/prep_bilingual_textless_manifest.py --save-root ${SAVE_ROOT}

# EuroParl-ST: download data 
EPST_DIR=/data/sls/temp/clai24/data/speech_matrix/eval_data/europarl_st
wget --directory-prefix=${EPST_DIR} https://www.mllp.upv.es/europarl-st/v1.1.tar.gz
tar -xvf ${EPST_DIR}/v1.1.tar.gz --directory ${EPST_DIR}

PROC_EPST_DIR=${EPST_DIR}/fairseq_processed
HUBERT_MODEL_DIR=/data/sls/temp/clai24/pretrained-models/mHuBERT
export PYTHONPATH="${PYTHONPATH}:${PWD}:${PWD}/../../:${PWD}/../hubert/simple_kmeans/"

# EuroParl-ST: setup fairseq test manifest without src unit extraction (doesn't require GPU)
python3 valid_test_sets/prep_epst_test_data.py --epst-dir ${EPST_DIR}/v1.1 --proc-epst-dir ${PROC_EPST_DIR} --save-root ${SAVE_ROOT}

# EuroParl-ST: setup fairseq test manifest with src unit extraction (requires GPU)
CUDA_VISIBLE_DEVICES=0 python3 valid_test_sets/prep_epst_test_data_customized.py --epst-dir ${EPST_DIR}/v1.1 --proc-epst-dir ${PROC_EPST_DIR} --save-root ${SAVE_ROOT} --hubert-model-dir ${HUBERT_MODEL_DIR}

# FLEURS: setup FLORES 
DATA_ROOT=/data/sls/temp/clai24/data
wget --directory-prefix=${DATA_ROOT} --trust-server-names https://tinyurl.com/flores200dataset
FLORES_ROOT=/data/sls/temp/clai24/data/flores200_dataset
tar -xvf ${DATA_ROOT}/flores200_dataset.tar.gz --directory ${DATA_ROOT}
FLEURS_ROOT=/data/sls/temp/clai24/data/speech_matrix/eval_data/fleurs

# FLEURS: download via HF via python 
from datasets import load_dataset
load_dataset("google/fleurs", "all", cache_dir=${FLEURS_ROOT})

# FLEURS: preprocess
PROC_FLEURS_DIR=/data/sls/temp/clai24/data/speech_matrix/eval_data/fairseq_processed_fleurs
export PYTHONPATH="${PYTHONPATH}:${PWD}:${PWD}/../../:${PWD}/../hubert/simple_kmeans/"
python3 valid_test_sets/preproc_fleurs_data.py --proc-fleurs-dir ${PROC_FLEURS_DIR} --cache-dir ${FLEURS_ROOT}

# FLEURS: speech alignment 
MANIFEST_ROOT=/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests
python3 valid_test_sets/align_fleurs_data.py --flores-root ${FLORES_ROOT} --proc-fleurs-dir ${PROC_FLEURS_DIR} --save-root ${MANIFEST_ROOT} 

# FLEURS: setup fairseq valid manifest with src unit extraction (requires GPU) 
HUBERT_MODEL_DIR=/data/sls/temp/clai24/pretrained-models/mHuBERT
CUDA_VISIBLE_DEVICES=0 python3 valid_test_sets/prep_fleurs_valid_data.py --proc-fleurs-dir ${PROC_FLEURS_DIR} --save-root ${SAVE_ROOT} --hubert-model-dir ${HUBERT_MODEL_DIR}

# FLEURS: setup fairseq test manifest without src unit extraction (doesn't require GPU) 
python3 valid_test_sets/prep_fleurs_test_data.py --proc-fleurs-dir ${PROC_FLEURS_DIR} --save-root ${SAVE_ROOT}

# FLEURS: setup fairseq test manifest with src unit extraction (requires GPU)
CUDA_VISIBLE_DEVICES=0 python3 valid_test_sets/prep_fleurs_test_data_customized.py --proc-fleurs-dir ${PROC_FLEURS_DIR} --save-root ${SAVE_ROOT} --hubert-model-dir ${HUBERT_MODEL_DIR} 
```

# Create model development train set 

```bash 
# create filtered train / valid set for es-en 
MANIFEST_ROOT=/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests

# e.g. filtered by target utt frame <= 100 frames 
python data_utils/filter_manifest.py \ 
    --lan_pair es-en --data_filter_threshold 1.09 --frame_threshold 100 \
    --data_root ${MANIFEST_ROOT}

# e.g. filtered by target utt frame <= 120 frames 
python data_utils/filter_manifest.py \ 
    --lan_pair es-en --data_filter_threshold 1.09 --frame_threshold 120 \
    --data_root ${MANIFEST_ROOT}
```

# Bilingual S2ST model training 

First, update the `audio_root` in `config.yaml` and `data` in `config_multitask.yaml`. Do this for the target language pair, e.g. 
`data/speech_matrix/speech_to_unit/s2u_manifests/es-en/{config.yaml,config_multitask.yaml}`.

```bash 
cd fairseq-ust/examples/speech_to_speech/
conda activate lexlearner

# training command for single node 1 GPU 
CUDA_VISIBLE_DEVICES=0 ./s2ut-training_es-en.sh es en false 400

# training command for single node 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 ./s2ut-training_es-en.sh es en true 400
```

# Bilingual S2ST model evaluation 

Requires setting up pre-trained unit-vocoders, see above. 

```bash 
cd fairseq-ust/examples/speech_to_speech/
conda activate lexlearner

# inference, synthesis, AST-BLEU. Requires a GPU
CUDA_VISIBLE_DEVICES=0 ./s2ut-inference_es-en.sh es en 400
```

# Satori Instruction for Ekin

```bash 
# activate Jeff conda env 
conda activate /nobackup/users/clai24/tools/anaconda3/envs/lexlearner2

# add this to your ~/.bashrc 
export PYTHONPATH="${PYTHONPATH}:/nobackup/users/clai24/tools/audio"

# train a fairseq S2ST model. Remember to ensure the conda env is activated for either. 
# scripts below are for your reference. 
cd /nobackup/users/clai24/lexicon/fairseq-ust/examples/speech_to_speech

## train locally by first requesting a node 
srun --gres=gpu:4 -N 1 --exclusive --qos=sched_level_2 --mem=400G --time 24:00:00 --cpus-per-task=4 --ntasks-per-node=4 --pty /bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./satori-s2ut-training_es-en.sh es en true 400

## train via slurm 
sbatch satori-s2ut-training_es-en_L400.slurm

# evaluation -- Satori not supported. Need to transfer trained model to SLS for eval. 
```

# Create Unit-to-Unit fariseq tsv files 

```bash 
MANIFEST_ROOT=/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests

# e.g. create unit-to-unit tsv for utterances filtered by target utt frame <= 100 frames 
python data_utils/create_src_unit_tsv.py \ 
    --lan_pair es-en --data_filter_threshold 1.09 --frame_threshold 100 \
    --data_root ${MANIFEST_ROOT}
```

# Unit-to-Unit S2ST experiments 

```bash 
cd fairseq-ust/examples/speech_to_speech/
conda activate lexlearner

# training command for single node 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 ./u2ut-training_es-en.sh es en true 400

# evaluation command 
CUDA_VISIBLE_DEVICES=0 ./u2ut-inference_es-en.sh es en 400
```

# Create Unit-to-Unit Lexicon w/ IBM Model 2

```bash 
cd lexlstm/lexicon
# extract forward and reverse unit intersection counts, and convert to prob distributions
./speech_lexicon_extractor.sh
```

# Create oracle phone / word alignments 

```bash 
# Step 1: transcribe with Whisper (requires GPU)
pip install -U openai-whisper
python data_utils/transcribe_via_whisper.py
# do sbatch data_utils/transcribe_via_whisper.sh for slurm jobs 

# Alternative Step 1: if audio is in .zip (fairseq) format, do the following,
# it first reads the .zip file and stores the wav in a standalone file, then run whisper
# also ran in parallel. Specify # process and process ID in command line arg
python data_utils/read_train_mined_audio.py --split en-train_mined_t1.09 --nprocess 20 --process_id 0
# do sbatch data_utils/read_train_mined_audio.sh en-train_mined_t1.09 20 0

# Step 2: text post-processing (no GPU)
python data_utils/text_norm.py

# Step 3: run MFA for alignment (no GPU) 
# if the # of (*.wav, *.txt) exceeds 10k. Run aligner for 10k pairs at once. Organize the directory with 
# python distribute.py
./data_utils/aligner.sh
# if run in parallel, do ./data_utils/aligner.sh 23 (23 is the process_id)

# Step 4: process MFA textgrid files (no GPU)
pip install textgrid 
python data_utils/process_textgrid.py --seg_type phones
python data_utils/process_textgrid.py --seg_type words
```

# Training HubertInfoAlign Model 

```bash 
cd /data/sls/scratch/clai24/lexicon/fairseq/examples/hubert
sbatch scripts/run_pretrain_v03.sh
```

# Decoding HubertInfoAlign Model

```bash 
cd /data/sls/scratch/clai24/lexicon/fairseq/examples/hubert
# Step 1: run and store parsing results (requires GPU for model inference)
python decode_scripts/hubert_info_align_phn_decode.py --split en-valid_vp --min-merge-pmi -5 --parse-alg top_down 
# to run on slurm: sbatch decode_scripts/hubert_info_align_phn_decode.sh -5 top_down 

# Step 2: visualization 
run ipython notebook: parsing_phn_viz.ipynb

# Step 3: word seg F1 
python analysis_scripts/cal_word_seg_f1.py --split en-valid_vp --min-pmi 0 --parse-alg top_down --top-down-recursive-search
```
