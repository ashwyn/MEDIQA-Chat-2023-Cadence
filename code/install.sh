#!/bin/bash
wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh -O ./anaconda.sh
bash ./anaconda.sh -b -p ./anaconda
source ./anaconda/bin/activate
conda init bash
conda env create -f environment.yml
conda activate Cadence_tasks_venv
conda deactivate