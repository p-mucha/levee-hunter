# Levee Hunter Group Project - Fathom


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


## Recreating Environment on Hypatia
Those instructions only apply when using conda, as other environment managers might be hard to use on the Hypatia cluster.

Since on the Hypatia cluster's instructions, it is only mentioned how to use conda and not miniconda, we need to suffer through the fact that base conda has countless packages in it, making the environment solver slow and often resulting in failures. 

The way around this is to use mamba. 

First we need to activate conda:
```
source /share/apps/anaconda/3-2019.03/etc/profile.d/conda.sh
```

Now one could use the conda to recreate the environment accordint to environment.yml, which will probably take ages and eventually fail, due to conda being conda.

Instead, we will create an environment with just the mamba in it for now. 


```
conda create -n Fathom -c conda-forge mamba

conda activate Fathom
```

Next, go to the directory where environment.yml is located, then check the location of this newly created environment. For example:

```
mamba env list
```
Which in my case is /home/pmucha/.conda/envs/Fathom.

Next, update current environment using environment.yml:
```
mamba env update --prefix /home/pmucha/.conda/envs/Fathom -f environment.yml
```

Which should finish in a couple of minutes (tested). 
















