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
pip install tensorboardX

# install additional requirements 
pip install -r requirements.txt
```

* Satori installation is tricker, and we need to make the following changes instead:

```bash 
# Satori version 
# install and activate conda 
conda create -n lexlearner2 python=3.8
conda activate lexlearner2

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
pip install tensorboardX
```

# Speech translation data setup with fairseq-ust

```bash 
cd fairseq-ust/examples/speech_matrix/
SAVE_ROOT=/data/sls/temp/clai24/data/speech_matrix/speech_to_unit
export PYTHONPATH="${PYTHONPATH}:${PWD}"
conda activate lexlearner2

# Run the following commands *sequentially*

# Speech-to-Speech Alignments
python mined_train_sets/download_mined_data.py --save-root ${SAVE_ROOT}

# download additional VP valid and test zip file 
wget --directory-prefix=${SAVE_ROOT}/audios https://dl.fbaipublicfiles.com/speech_matrix/audios/valid_test_vp_aud.zip 

# Speech-to-Unit Data
python mined_train_sets/download_speech_to_unit.py --save-root ${SAVE_ROOT}

# Reproduce Bilingual Train Data
python3 speech_to_speech/prep_bilingual_textless_manifest.py --save-root ${SAVE_ROOT}
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

