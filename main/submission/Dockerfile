#FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
#FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04
FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04

RUN apt-get clean && apt-get update && apt-get install -y locales
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV SHELL /bin/bash

RUN apt-get update && \
    apt-get install -y curl bzip2 xvfb ffmpeg git libxrender1

WORKDIR /aaio

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda clean -ya && \
     /opt/conda/bin/conda create -n python36 python=3.6 numpy

ENV PATH /opt/conda/envs/python36/bin:/opt/conda/envs/bin:$PATH

#RUN pip install animalai

COPY agent.py /aaio/agent.py
COPY data /aaio/data
RUN mkdir /aaio/ppo
COPY ppo/distributions.py /aaio/ppo/
COPY ppo/envs.py /aaio/ppo/
COPY ppo/model.py /aaio/ppo/
COPY ppo/utils.py /aaio/ppo/

ENV HTTP_PROXY ""
ENV HTTPS_PROXY ""
ENV http_proxy ""
ENV https_proxy ""

########################################################################################################################
# YOUR COMMANDS GO HERE

# For example, if your agent requires the animalai-train library
# you can add the following (remove if not needed):
# RUN pip install animalai-train
RUN apt-get install -y vim
RUN pip install git+git://github.com/compsciencelab/baselines.git
RUN pip install "git+git://github.com/compsciencelab/AnimalAI-Olympics.git#subdirectory=animalai"
# RUN pip install git+git://github.com/compsciencelab/pytorch-a2c-ppo-acktr-gail.git
RUN /opt/conda/bin/conda install -n python36 pytorch torchvision cudatoolkit=10.0 -c pytorch
RUN /opt/conda/bin/conda clean -ya 


########################################################################################################################

CMD ["/bin/bash"]

