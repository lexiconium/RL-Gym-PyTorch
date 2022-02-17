# Pendulum

## Overview

## Specifications

### Description

The inverted pendulum swingup problem is a classic problem in the control literature. In this version of the problem,
the pendulum starts in a random position, and the goal is to swing it up so it stays upright.

- `x-y`: cartesian coordinates of the pendulum's end in meters.
- `theta`: angle in radians.
- `tau`: torque in `N * m`. Defined as positive _counter-clockwise_.

### Action Space

The action is the torque applied to the pendulum.

| Num | Action | Min  | Max |
|-----|--------|------|-----|
| 0   | Torque | -2.0 | 2.0 |

### Observation Space

The observations correspond to the x-y coordinate of the pendulum's end, and its angular velocity.

| Num | Observation      | Min  | Max |
|-----|------------------|------|-----|
| 0   | x = cos(theta)   | -1.0 | 1.0 |
| 1   | y = sin(angle)   | -1.0 | 1.0 |
| 2   | Angular Velocity | -8.0 | 8.0 |

### Rewards

The reward is defined as:

```
r = -(theta^2 + 0.1*theta_dt^2 + 0.001*torque^2)
```

where `theta` is the pendulum's angle normalized between `[-pi, pi]`. Based on the above equation, the minimum reward
that can be obtained is `-(pi^2 + 0.1*8^2 + 0.001*2^2) = -16.2736044`, while the maximum reward is zero (pendulum is
upright with zero velocity and no torque being applied).

### Starting State

The starting state is a random angle in `[-pi, pi]` and a random angular velocity in `[-1,1]`.

### Episode Termination

An episode terminates after 200 steps. There's no other criteria for termination.

## References

[1] https://gym.openai.com/envs/Pendulum-v0 \
[2] https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
