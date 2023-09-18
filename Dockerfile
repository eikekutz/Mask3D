# FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu18.04
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04
ENV VERSION_NUMBER 0.0.2

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /

# remove current python version

RUN apt-get remove python3.* -y && \
    apt-get remove --auto-remove python3.* -y && \
    apt-get purge python3.* -y && \
    apt-get purge --auto-remove python3.* -y && \
    rm -rf /usr/bin/python3 && \
    rm -rf /usr/bin/python
# Install Python3.10.9 from source
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get install ffmpeg libsm6 libxext6  -y && \
    apt -y install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev liblzma-dev git && \
    wget https://www.python.org/ftp/python/3.10.9/Python-3.10.9.tgz && \
    tar -xf Python-3.10.9.tgz && \
    cd Python-3.10.9 && \
    ./configure --enable-optimizations && \
    make -j 8 && \
    make install && \
    cd .. && \
    rm -rf Python-3.10.9 && \
    rm -rf Python-3.10.9.tgz && \
    python3 -m pip install --upgrade pip && \
    apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

RUN mkdir /Upcycling/ && \
    cd /Upcycling/ && \
    git clone https://github.com/eikekutz/Mask3D.git/ && \
    # git clone https://github.com/JonasSchult/Mask3D.git && \
    cd Mask3D

# ENV TORCH_CUDA_ARCH_LIST="3.7 5.0 6.0 7.0 8.0 8.6+PTX"

# set version number
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
ENV PYTORCH_NVFUSER_DISABLE=fallback

RUN conda env create -f /Upcycling/Mask3D/environment.yml

ENV PATH /opt/conda/envs/mask3d_cuda113/bin:$PATH
RUN echo "source activate mask3d_cuda113" > ~/.bashrc && \
    pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 && \
    pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html && \
    pip3 install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps && \
    conda install openblas-devel -c anaconda && \
    cd /Upcycling/Mask3D/third_party && \
    git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine" && \
    cd /Upcycling/Mask3D/third_party/MinkowskiEngine && \
    git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 && \
    python setup.py install --force_cuda --blas_include_dirs=/opt/conda/envs/mask3d_cuda113/include --blas=openblas && \
    cd /Upcycling/Mask3D/third_party && \
    git clone https://github.com/ScanNet/ScanNet.git && \
    cd ScanNet/Segmentator && \
    git checkout 3e5726500896748521a6ceb81271b0f5b2c0e7d2 && \
    make && \
    cd /Upcycling/Mask3D/third_party/pointnet2 && \
    python setup.py install && \
    pip3 install pytorch-lightning==1.7.2 && \
    pip install scikit-video
WORKDIR /Upcycling/Mask3D/

# ENTRYPOINT [ "cd /Upcycling/", "python setup.py install --force_cuda --blas=openblas" ]


