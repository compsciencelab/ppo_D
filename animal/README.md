# Install

``` 
    conda create -n animal python=3.6
    git clone https://github.com/compsciencelab/AnimalAI-Olympics.git
    #git clone git@github.com:compsciencelab/AnimalAI-Olympics.git 
    cd AnimalAI-Olympics/animalai
    pip install -e .
    cd ..
    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
    cd env
    wget https://www.doc.ic.ac.uk/~bb1010/animalAI/env_linux_v1.0.0.zip
    unzip env_linux_v1.0.0.zip
    pip install git+git://github.com/compsciencelab/baselines.git
    pip install "git+git://github.com/compsciencelab/AnimalAI-Olympics.git#subdirectory=animalai"
    conda install pandas

```
