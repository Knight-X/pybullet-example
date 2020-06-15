FROM ubuntu:16.04

RUN apt-get update \
    && apt-get install -qy git vim wget bzip2 gcc g++ \
        cmake libopenmpi-dev python3-dev zlib1g-dev curl libsm6 libglib2.0-0 libxrender1 libxext-dev \
	&& apt-get purge

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py
RUN python3 -m pip install numpy torch torchvision

RUN pip install jupyter matplotlib pybullet tqdm stable-baselines papermill nbdime tensorflow==1.14.0 \
	cloudpickle==1.2.0 bleach==1.5.0


COPY . /example

WORKDIR /example

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

