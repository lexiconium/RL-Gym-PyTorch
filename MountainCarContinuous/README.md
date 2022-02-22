# MountainCarContinuous

## Overview

https://user-images.githubusercontent.com/84180121/155066188-6066eb6b-a2a8-4699-a77a-954d731386bc.mp4

## Specifications [2]

### Observation Space

The observation space is a 2-dim vector, where the 1st element represents the "car position" and the 2nd element
represents the "car velocity".

### Action

The actual driving force is calculated by multiplying the power coef by power (0.0015)

### Reward

Reward of 100 is awarded if the agent reached the flag (position = 0.45)
on top of the mountain. Reward is decrease based on amount of energy consumed each step.

### Starting State

The position of the car is assigned a uniform random value in [-0.6 , -0.4]. The starting velocity of the car is always
assigned to 0.

### Episode Termination

The car position is more than 0.45. Episode length is greater than 200

## References

[1] https://gym.openai.com/envs/MountainCarContinuous-v0 \
[2] https://github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py
