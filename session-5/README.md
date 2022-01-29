# Session 5
Monitoring using Tensorboard and WandB. 

In this session we will solve two very simple tasks: reconstruction and classification
on a subset of the MNIST dataset. In this hands-on the training itself it's not the most important, therefore with only 
5 epochs trained in CPU it is enough to visualize everything we want in the Tensorboard and Wandb dashboards.
## Installation
### With Conda
Create a conda environment by running
```
conda create --name aidl-session-5 python=3.8
```
Then, activate the environment
```
conda activate aidl-session-5
```
and install the dependencies
```
pip install -r requirements.txt
```
## Running the project

To run the project, run
```
python session-5/main.py --task reconstruction --log_framework tensorboard
python session-5/main.py --task reconstruction --log_framework wandb
python session-5/main.py --task classification --log_framework tensorboard
python session-5/main.py --task classification --log_framework wandb
```
To run the project with different arguments, run
```
python session-5/main.py --task reconstruction --log_framework tensorboard --latent_dims 64 --n_epochs 10
```
