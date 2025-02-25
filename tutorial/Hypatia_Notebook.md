# Hypatia Notebook Tutorial

## Table of Contents
- [Introduction](#introduction)
- [Setting Up the Environment](#setting-up-the-environment)
- [Running Python Notebooks](#running-python-notebooks)
- [Troubleshooting](#troubleshooting)

## Introduction
This tutorial provides a step-by-step guide on how to run Python notebooks on the UCL Hypatia cluster using VSCode.

## Setting Up the Environment
1. **Loading Anaconda**: Run `source /share/apps/anaconda/3-2019.03/etc/profile.d/conda.sh` to load Anaconda.
2. **Set Up Virtual Environment**: Visit the [README](../README.md) to find the steps for setting up and activating a virtual environment.
3. **Install levee_hunter**: Run `pip install -e .` from within the levee_hunter directory.

## Running Python Notebooks
1. **Create notebooks/_jobs/ directory**: Create a directory within `/notebooks/` named `_jobs`.
2. **Launch Jupyter Server on GPU**: Run `sbatch jupyterlab_gpu.sh [environment name]` from within the `/notebooks/` directory.
3. **Access the Server URL**: The `_jobs/` directory will now have a `.out` file containing details of the Jupyter server URL (copy the URL). It will be under the line saying "Jupyter Server X.XX.X is running at:". It may take up to a few minutes to appear within the `.out` file.
4. **Run Your Notebook**: Open the notebook you wish to run within VSCode. When selecting a kernel for your notebook, select "Existing Jupyter Server" and enter the URL. Hit enter to confirm the next few steps and your notebook should now be running.

## Troubleshooting
**Pytorch**: If Pytorch isn't correctly utilizing CUDA on GPU cores, follow these steps:
1. Uninstall Pytorch: `mamba uninstall pytorch`
2. Reinstall Pytorch using pip: `pip3 install torch torchvision torchaudio`
