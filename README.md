# Levee Hunter Group Project - Fathom


## Table of Contents 
- [Installation](#installation)
- [Getting Data](#getting-data)
- [Recreating Environment on Hypatia](#recreating-environment-on-hypatia)

## Installation

To install Levee Hunter, clone this repository and install:


```bash
git clone https://github.com/p-mucha/levee-hunter.git

cd levee-hunter

pip install -e .
```

A useful tutorial notebook is located inside the tutorial/ directory.


## Getting Data

All data files can be found on Hypatia at:
`/share/gpu5/pmucha/fathom/levee-hunter/data/`

## Computing on Hypatia
A login to the Hypatia cluster is through the login node. After user ssh to hypatia, the terminal will show [username]@hypatia-login, which means user is on the login-node. 

**No computing on the login-node**: It is important that login-node should not be used for any heavy tasks. This includes using conda to solve environment. This can potentially block access to the cluster for all the other users. 

**Changing Node**: To go to other node than the login one, use one of the two scripts: 
```
source cpu_bash.sh
```

Or 
```
source gpu_bash.sh
```

Note not everyone has access to the cpu nodes, therefore for example I have to use gpu nodes only.

**Running Jupyter**: Please see the tutorial/Hypatia_notebook.md.


## Recreating Environment on Hypatia
Please see the 'Computing on Hypatia' section before starting with this.

Those instructions only apply when using conda, as other environment managers might be hard to use on the Hypatia cluster.

1. **Activate Conda**:
```
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
```

2. **Create Environment**: Create a new environment named [environment name] using conda. (Optionally) install mamba in it. Mamba in general can be used exactly the same as Conda (simply replace any conda command with mamba), but is much faster and more reliable.
```
conda create -n [environment name] -c conda-forge mamba

conda activate Fathom
```
3. **Find environment.yml**: Go to the directory where environment.yml is located. Here it is the root directory:
```
cd /path/to/levee-hunter
```
4. **Find the Environment**: Check where this new environment has been installed. Usually it will be ~/.conda/envs/[environment name]
```
mamba env list
```
Which in my case is /home/pmucha/.conda/envs/Fathom.

5. **Update Environment**: use environment.yml to install the required dependencies into [environment name]:
```
mamba env update --prefix /path/to/.conda/envs/[environment name] -f environment.yml
```
Which should finish in a couple of minutes.

## Troubleshooting
Sometimes, an error might occur for example for `mamba --version` but `which mamba` can still find the correct location.
In that case I found
```
unset -f mamba
```
To be helpful.















