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
pip install tensorboardX pandas

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
```

# Speech translation data setup with fairseq-ust

```bash 
cd fairseq-ust/examples/speech_matrix/
SAVE_ROOT=/data/sls/temp/clai24/data/speech_matrix/speech_to_unit
export PYTHONPATH="${PYTHONPATH}:${PWD}:${PWD}/../../"
conda activate lexlearner

# Run the following commands *sequentially*

# SpeechMatrix: Speech-to-Speech Alignments (it takes 1-3 days to download all datasets)
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

# EuroParl-ST: setup fairseq test manifest 
python3 valid_test_sets/prep_epst_test_data.py --epst-dir ${EPST_DIR}/v1.1 --proc-epst-dir ${PROC_EPST_DIR} --save-root ${SAVE_ROOT}

# EuroParl-ST: setup fairseq test manifest with src unit extraction (requires GPU)
# RUNNING on SLS
python3 valid_test_sets/prep_epst_test_data_customized.py --epst-dir ${EPST_DIR}/v1.1 --proc-epst-dir ${PROC_EPST_DIR} --save-root ${SAVE_ROOT} --hubert-model-dir ${HUBERT_MODEL_DIR}

# FLEURS: setup FLORES 
DATA_ROOT=/data/sls/temp/clai24/data
wget --directory-prefix=${DATA_ROOT} --trust-server-names https://tinyurl.com/flores200dataset
FLORES_ROOT=/data/sls/temp/clai24/data/flores200_dataset
tar -xvf ${DATA_ROOT}/flores200_dataset.tar.gz --directory ${DATA_ROOT}

# FLEURS: download via HF
```python
from datasets import load_dataset
fleurs_retrieval = load_dataset("google/fleurs", "all")
```
python3 valid_test_sets/preproc_fleurs_data.py --proc-fleurs-dir ${PROC_FLEURS_DIR}

PROC_FLEURS_DIR=/data/sls/temp/clai24/data/speech_matrix/eval_data/fairseq_processed_fleurs
HUBERT_MODEL_DIR=/data/sls/temp/clai24/pretrained-models/mHuBERT
export PYTHONPATH="${PYTHONPATH}:${PWD}:${PWD}/../../:${PWD}/../hubert/simple_kmeans/"

# FLEURS: setup fairset test manifest with src unit extraction (requires GPU)
# RUNNING on SLS 
python3 valid_test_sets/prep_fleurs_test_data_customized.py --proc-fleurs-dir ${PROC_FLEURS_DIR} --save-root ${SAVE_ROOT} --hubert-model-dir ${HUBERT_MODEL_DIR} 
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
CUDA_VISIBLE_DEVICES=0 ./s2ut-training_es-en.sh es en false

# training command for single node 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 ./s2ut-training_es-en.sh es en true
```

# Bilingual S2ST model evaluation 

Fisrt download pre-trained unit-vocoders.

```bash 
cd fairseq-ust/examples/speech_to_speech/
conda activate lexlearner

# inference, synthesis, AST-BLEU. Requires a GPU
CUDA_VISIBLE_DEVICES=0 ./inference_es-en.sh
```
