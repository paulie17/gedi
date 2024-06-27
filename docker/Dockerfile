FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

COPY . /gedi/

WORKDIR /gedi

RUN apt update
RUN apt install -y python3-pip

RUN pip install --upgrade pip
RUN pip install torch-1.8.1-cp38-cp38-linux_x86_64.whl
RUN pip install open3d==0.15.2 torchgeometry==0.1.2 gdown tensorboard protobuf==3.20
RUN pip install setuptools==69.5.1

WORKDIR /gedi/backbones
RUN pip install ./pointnet2_ops_lib/

WORKDIR /gedi
CMD ["/bin/bash"]

