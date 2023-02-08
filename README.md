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

# install additional requirements 
pip install -r requirements.txt
```
