# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.10.0
* Python version >= 3.8

``` bash 
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
conda create -n lexlearner3 python=3.7
conda activate lexlearner3

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
