#docker build -t clip_cuda11_1:1.0 --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .

FROM nvcr.io/nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV cwd="/home/"
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"
WORKDIR $cwd

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#RUN apt-key del 7fa2af80
#RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
#RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC F60F4B3D7FA2AF80

RUN apt-get -y update
# RUN apt-get -y upgrade
RUN apt -y update

ARG DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata python3-tk

RUN apt-get install -y \
    software-properties-common \
    build-essential \
    checkinstall \
    cmake \
    pkg-config \
    yasm \
    git \
    vim \
    curl \
    wget \
    gfortran \
    libjpeg8-dev \
    libpng-dev \
    libtiff5-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libdc1394-22-dev \
    libxine2-dev \
    sudo \
    apt-transport-https \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    dbus-x11 \
    vlc \
    iputils-ping \
    python3-dev \
    python3-pip

RUN apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove

RUN rm -rf /var/cache/apt/archives/

### APT END ###

RUN python3 -m pip install --no-cache-dir --upgrade pip

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# INSTALL PYCOCOTOOLS
RUN pip3 install --no-cache-dir 'git+https://github.com/openai/CLIP.git'

#RUN pip3 install clearml

# minimal Dockerfile which expects to receive build-time arguments, and creates a new user called “user” (put at end of Dockerfile)
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user
