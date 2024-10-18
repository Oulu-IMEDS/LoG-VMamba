FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

SHELL ["/bin/bash", "--login", "-c"]

# RUN echo "export PATH=/opt/conda/bin:${PATH}" >> ~/.bashrc

# Looks like conda will activate the env ONLY if we run the whole thing with bash

# Setting up the system (as 1 layer for compactness)
RUN apt-get update && apt-get install -y --no-install-recommends wget

# Getting conda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && rm Miniconda3-latest-Linux-x86_64.sh

RUN echo ". \"/opt/miniconda/etc/profile.d/conda.sh\"" >> ~/.bashrc
ENV PATH=/opt/miniconda/bin:${PATH}

# Now we can create the environment
# RUN conda create -n mlpipeline_segmentation python=3.8 numpy scipy pandas matplotlib
SHELL ["conda", "run", "-n", "base", "/bin/bash", "--login", "-c"]

RUN conda install -y python=3.8.5
RUN pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip install numpy scipy matplotlib einops
RUN pip install tqdm omegaconf==2.0.5 opencv-python-headless==4.5.1.48 scikit-learn solt tensorboard medmnist
RUN pip install scikit-image==0.18.1 segmentation-models-pytorch==0.3.3 efficientnet-pytorch==0.7.1
RUN pip install pillow click ml-collections
RUN pip install pandas==1.3.5 monai==1.0.1 natsort yacs imutils nibabel mamba-ssm dynamic_network_architectures
RUN pip install nibabel GeodisTK SimpleITK
RUN pip install torchinfo

# RUN echo "conda activate mlpipeline_segmentation" >> /etc/bash.bashrc

# Copying the files for the initial env init.
COPY mlpipeline/ /opt/package/mlpipeline/
COPY setup.py /opt/package/
RUN pip install -e /opt/package

# This folder needs to be mounted
RUN mkdir /opt/workdir/
RUN mkdir /opt/inference_results/
RUN mkdir /opt/visuals/
WORKDIR /opt/workdir/
# ENTRYPOINT ["conda", "run", "-n", "mlpipeline_segmentation", "/bin/bash", "-l", "-c"]
