# FedRL

Hao Jin, Yang Peng, Wenhao Yang, Shusen Wang and Zhihua Zhang. Federated Reinforcement Learning with Environment Heterogeneity. AISTATS, 2022.

## Requirements

- [PyTorch](http://pytorch.org/): version 1.1.0 is required
- [OpenAI Gym](https://gym.openai.com/): ```pip install gym gym[atari]```
- [TensorBoardX](https://github.com/lanpa/tensorboardX): ```pip install tensorboardX```
- [PTAN](https://github.com/Shmuma/ptan): install from sources (delete torch==1.7.0 which is unnecessary in requirements.txt)
- [MuJoCo](https://mujoco.org/): only for HalfCheetah and Hopper environment

## Getting started

### Tabular experiments

The customized environments *RandomMDPs* and *WindyCliffs* are implemented in `utils.py` and `GridWorldEnvironment.py` respectively.

Some utility functions and standard tabular reinforcement learning algorithms are included in `utils.py`.

The algorithms *QAvg*, *SoftPAvg* and *ProjPAvg* are implemented for both the environments in the python files with the corresponding prefixes.

The python files with prefix *heter* are for the experiments in **Table 1** in the paper.

### Deep experiments

The customized environments are implemented in `MyCartPole.py`, `MyAcrobot.py`, `MyHalfCheetah.py` and `MyHopper.py`.

Some utility functions and standard deep reinforcement learning algorithms are included in `DeepRLAlgo.py`.

The algorithms *DQNAvg* and *DDPGAvg* are implemented in `DQNAvg.py` and `DDPGAvg.py` respectively, we apply these methods in the customized environments in the python files with the corresponding prefixes.

The personalized version of the above is implemented in the python files with prefix *Per*.

