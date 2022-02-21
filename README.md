# RL-Gym-PyTorch

The purpose of this repository is to implement Reinforcement Learning algorithms in PyTorch and test them on a variety
of OpenAI Gym environments.

All implementations are specific to each environment with minimum generalization so that the entire structure of the
algorithm can be seen as clearly as possible.

## Environments

Each directory contains:

- Overview and Specifications of the environment
- Algorithms*

`*`: Includes pseudocode, code and result

### Availability

- [CartPole-v1](https://github.com/lexiconium/RL-Gym-PyTorch/tree/main/CartPole)
    - [DQN](https://github.com/lexiconium/RL-Gym-PyTorch/tree/main/CartPole/DQN)
    - [PPO](https://github.com/lexiconium/RL-Gym-PyTorch/tree/main/CartPole/PPO)
- [Pendulum-v0](https://github.com/lexiconium/RL-Gym-PyTorch/tree/main/Pendulum)
    - [DDPG](https://github.com/lexiconium/RL-Gym-PyTorch/tree/main/Pendulum/DDPG)
- [LunarLander-v2](https://github.com/lexiconium/RL-Gym-PyTorch/tree/main/LunarLander)
    - [PPO (Continuous)](https://github.com/lexiconium/RL-Gym-PyTorch/tree/main/LunarLander/Continuous/PPO)
    - [PPO (Discrete)](https://github.com/lexiconium/RL-Gym-PyTorch/tree/main/LunarLander/Discrete/PPO)
    - [TD3 (Continuous)](https://github.com/lexiconium/RL-Gym-PyTorch/tree/main/LunarLander/Continuous/TD3)
- [Walker2d](https://github.com/lexiconium/RL-Gym-PyTorch/tree/main/Walker2d)
    - [TD3](https://github.com/lexiconium/RL-Gym-PyTorch/tree/main/Walker2d/TD3)

## Dependencies

`Python` 3.8.12 \
`Gym` 0.19.0 \
`mujoco-py` 2.1.2.14 \
`PyTorch` 1.10.1 \
`NumPy` 1.21.5
