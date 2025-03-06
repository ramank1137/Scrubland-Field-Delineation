# Scrubland Field Delineation
This repo is enabled to run inside a docker container. It requires `Docker Engine` with `nvidia-container-toolkit` installed as prerequisite to be installed inorder to build the container or pull it from docker hub. Please follow the following instructions to install the prerequisite.

### Prerequisites
1. Install `Docker Engine` from [here](https://docs.docker.com/engine/install/)
2. Install `nvidia-container-toolkit` from from [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Building Docker Image
After this, the docker image can be built by executing following command inside this github repo folder `Scrubland-Field-Delineation`

`sudo docker build --progress=plain -t <image-name> .`

Use the following if you don't want to see the progress

`sudo docker build -t <image-name> .`

### Starting Docker container
To start docker container please run the following command.

`sudo docker run --shm-size=60gb --gpus all --init -it -v $(pwd):/app <image-name> bash`
