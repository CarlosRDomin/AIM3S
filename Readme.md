## Installation instructions

First, you'll need to install `conda` ([Miniconda3](https://docs.conda.io/en/latest/miniconda.html) recommended)

Then, you have two options:
 1. Install from environment.yml:
    ```bash
    conda env create -f environment.yml
    ```
 2. Manual install:
    ```bash
    conda create -n aim3s python=3 opencv=4 protobuf numpy scipy matplotlib h5py pytz
    conda activate aim3s
    pip install grpcio googleapis-common-protos
    ```
