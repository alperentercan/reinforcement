# Reinforcement


This repository is under construction. For test purposes, the folders for different algorithms/environments are kept separate. 

# Algorithms :

* DDPG : Working. Use `merge-sm-goal` branch for unified implementation for single and multi goal environments. Use `--alg` to set which implementation to be used:
   - **ddpg** : Original algorithm, proposed in the paper
   - **fm** : Original algorithm + forward model. Uses forward model predictions with TD errors to train Critic.
   - **exp** : Runs the experimental implementation. 
* Option-Critic : Working

# ToDo :  
 * CUDA 
 * MPI
 * Add Stochastic Policy option to Option-Critic

# Note:

`main.py`,`random_process.py`, `evaluator.py`, and `util.py` are from [pytorch-ddpg](https://github.com/ghliu/pytorch-ddpg) repo.
