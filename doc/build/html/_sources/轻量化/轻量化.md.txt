# flare_21

The solution for this challenge is based on the nnUNet repository. Give thanks to Fabian’s great work!

We made three major modifications on top of the original nnUNet:

1. In the inference stage, we only used the two samples with the largest flipping difference as test time augmentation (TTA).
2. In the deep supervision block, we only used the decoder output of the last three layers for weighted average.
3. To reduce performance degradation caused by network simplification, we referred to the idea of OHEM and used Top-k loss in cross-entropy to supervise the voxels with the largest difference, which would help maintain the performance as much as possible under the simplified structure.

## Update 2021.10.23

I have retrained and uploaded the docker file based on the [Abdomen1KCT-fullySupervised](https://abdomenct-1k-fully-supervised-learning.grand-challenge.org/) dataset, which can be obtained through [Docker Hub](https://hub.docker.com/r/ttime/abdomen1k) or [Google Netdisk](https://drive.google.com/file/d/1KrFi6rCQDsce5dWZFS_xaLryOZRCw1iQ/view?usp=sharing). The installation method can refer to the docker chapter below but change the instructions to:

```bash
# (Optional) download from Docker Hub
docker pull ttime/abdomen1k
#subtask 1
docker container run --gpus "all" --name ttime_abdomen1k --rm -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ ttime_abdomen1k:latest /bin/bash -c "sh predict.sh 1"
#subtask 2
docker container run --gpus "all" --name ttime_abdomen1k --rm -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ ttime_abdomen1k:latest /bin/bash -c "sh predict.sh 2"
```

## Installation

### from source (recommend)

1. Clone the directory and its submodule (nnUNet):

    ```bash
    git clone --recursive https://github.com/tea321000/flare21_solution
    cd flare21_solution
    # if you are updating an existing checkout
    git submodule sync
    git submodule update --init --recursive --jobs 0
    ```

2. Follow the installation instructions of [my forked repository](https://github.com/tea321000/nnUNet/tree/flare_21#installation) to install the modified version of nnUNet. Ensure that the relevant environment variables have been set similar to the following commands:

    ```bash
    # some APIs in the recently updated version of batchgenerators have changed, so it is recommended to use the old version
    pip install batchgenerators==0.21
    cd nnUNet
    pip install -e .
    # you can write environment variables into .bashrc
    export nnUNet_raw_data_base="/media/fabian/nnUNet_raw"
    export nnUNet_preprocessed="/media/fabian/nnUNet_preprocessed"
    export RESULTS_FOLDER="/media/fabian/nnUNet_trained_models"
    # check if the environment variable has been set
    echo $nnUNet_raw_data_base
    echo $nnUNet_preprocessed
    echo $RESULTS_FOLDER
    ```

### docker

We also provide [docker images](https://drive.google.com/file/d/1a-atCB_BZkeyzrho2ediEtLzt6rk6YT4/view) for the competition, but this method of installation is not convenient for modification, so it is recommended to use it only for inference. You can use it with the following instructions:

```bash
docker image load < ttime_flare_final.tar.gz
docker container run --gpus "all" --name ttime --rm -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ teamname:latest /bin/bash -c "sh predict.sh"
```

## Training

At present, the 2d network has not been completely modified yet, so my solution can only be used to train the 3d network. If you have any ideas or questions, you can post them in the issue or send an email to me. The training instructions of the 3d network is the same as the original [nnUNet](https://github.com/tea321000/nnUNet/tree/flare_21#3d-full-resolution-u-net), but the Trainer needs to be changed to `nnUNetTrainerV2_flare`:

```bash
# my solution in challenge uses lowres network, but fullres network can also be used
nnUNet_train 3d_lowres nnUNetTrainerV2_flare TaskXXX_MYTASK FOLD --npz
```

## Inference

1. Download the trained model from the [download.sh](https://github.com/tea321000/flare21_solution/blob/main/download.sh) script or the [Google Drive](https://drive.google.com/file/d/1YW8MsLaYUr6lhfpf_LL6kTelPiuJRhq9/view) link(task id is 500):

    ```bash
    bash download.sh
    ```

    If you download from Google Drive link, you need to put the compressed file into the manually created `trained_weight` folder in `flare21_solution` folder and run [download.sh](https://github.com/tea321000/flare21_solution/blob/main/download.sh).

2. Create `inputs` folder under `flare21_solution` , put the samples to be predicted into the folder, and run [predict.sh](https://github.com/tea321000/flare21_solution/blob/main/predict.sh):

    ```bash
    bash predict.sh
    ```

The predictions are saved in the `outputs` folder by default, and the input and output paths can be changed by modifying the script.
