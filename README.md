# Scrubland Field Delineation
This repository is designed to run inside a Docker container. It requires `Docker Engine` and `nvidia-container-toolkit` as prerequisites for building the container or pulling it from Docker Hub. Follow the instructions below to install the required dependencies.

### Prerequisites
1. Install `Docker Engine` from [here](https://docs.docker.com/engine/install/)
2. Install `nvidia-container-toolkit` from [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Building Docker Image
After this, the docker image can be built by executing following command inside this github repo folder `Scrubland-Field-Delineation`

`sudo docker build --progress=plain -t <image-name> .`

Use the following command to disable progress display.

`sudo docker build -t <image-name> .`

### Pulling Docker Image
Instructions to pull docker image comming soon...

### Starting Docker container
To start docker container please run the following command.

`sudo docker run --shm-size=60gb --gpus all --init -it -v $(pwd):/app <image-name> bash`

### Running Script
Inside the container, the required environment is pre-activated for convenience. The `script.py` is the one that needs to be executed to obtaine vector boundaries of fields, plantation and scrubland. Before running the script, download the model (`india_Airbus_SPOT_model.params`) from [here](https://zenodo.org/records/7315090) and place it inside `Scrubland-Field-Delineation` folder. After this you can run the script using the following command.

`python script.py`

When a new container is started for the first time, this will prompt a link for authenticating into Google Earth Engine. Click the link or copy-paste it into a browser, then log in using a Google account associated with Google Earth Engine. After authentication, you will receive an access key, which should be pasted into the Docker container to complete the authentication process. Once authenticated, the script will process the specified Region of Interest(ROI) and generate vector boundaries for farms, plantations, and scrubland.

You can set the ROI inside the `script.py` after `ee.Initialize` in `__main__` block. A boilerplate code for running on a rectangular region instead of any polygon ROI is also provided after the ROI declaration which commented out for ease of testing. Also set the name of the `directory` which will be used to download images, store predictions and store vector boundaries.
