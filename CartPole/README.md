# CartPole

## Speicification

### Observation Space

| Num | Observation           | Min                  | Max                 |
|-----|-----------------------|----------------------|---------------------|
| 0   | Cart Position         | -4.8*                 | 4.8*                |
| 1   | Cart Velocity         | -Inf                 | Inf                 |
| 2   | Pole Angle            | ~ -0.418 rad (-24°)** | ~ 0.418 rad (24°)** |
| 3   | Pole Angular Velocity | -Inf                 | Inf                 |

- `*`: the cart x-position can be observed between `(-4.8, 4.8)`, but an episode terminates if the cart leaves
  the `(-2.4, 2.4)` range.
- `**`: Similarly, the pole angle can be observed between  `(-.418, .418)` radians or precisely **±24°**, but an episode
  is terminated if the pole angle is outside the `(-.2095, .2095)` range or precisely **±12°**

### Action Space

| Num | Action                 |
|-----|------------------------|
| 0   | Push cart to the left  |
| 1   | Push cart to the right |

## Reference

[1] https://gym.openai.com/envs/CartPole-v1/ \
[2] https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py