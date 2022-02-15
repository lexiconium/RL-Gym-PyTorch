from collections import deque, namedtuple
from itertools import count
import random

import gym
import numpy as np
import torch
from torch import nn, no_grad, optim
from torch.nn import functional as F


def to_torch(func):
    def convert_and_execute(cls, *args, **kwargs):
        args = [torch.as_tensor(arg).float() for arg in args]
        kwargs = {key: torch.as_tensor(arg).float() for key, arg in kwargs.items()}
        return func(cls, *args, **kwargs)

    return convert_and_execute


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)
        self.fields = namedtuple("fields", ["state", "action", "reward", "next_state", "done"])

    def append(self, **kwargs):
        self.memory.append(self.fields(**kwargs))

    def sample(self, batch_size: int):
        sampled_fields_list = random.sample(self.memory, k=batch_size)
        return self.fields(*map(np.array, zip(*sampled_fields_list)))

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.dqn = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    @to_torch
    def forward(self, state: torch.Tensor):
        return self.dqn(state)

    @no_grad()
    def select_action(self, state: torch.Tensor, epsilon: float = 0):
        if random.random() < epsilon:
            return random.randrange(2)
        return self(state).argmax().item()


def deep_q_learning(
    gamma: float = 0.99,
    num_episodes: int = 5000,
    batch_size: int = 64,
    target_update_interval: int = 20
):
    env = gym.make("CartPole-v1")

    replay_memory = ReplayMemory(capacity=50000)
    policy = DQN()

    target_policy = DQN()
    target_policy.load_state_dict(policy.state_dict())
    target_policy.eval()

    optimizer = optim.Adam(policy.parameters(), lr=5e-4)

    total_timesteps = 0
    acc_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        for t in count(1):
            epsilon = max(1 - 0.9 * (total_timesteps / 50000), 0.1)
            action = target_policy.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            replay_memory.append(
                state=state,
                action=action,
                reward=reward / 100,
                next_state=next_state,
                done=done
            )

            if done:
                break
            state = next_state

            if len(replay_memory) < 10000:
                continue

            transition = replay_memory.sample(batch_size)

            actions = torch.as_tensor(transition.action).unsqueeze(dim=-1).long()
            rewards = torch.as_tensor(transition.reward).float()
            dones = torch.as_tensor(transition.done).long()

            estimate = torch.gather(policy(transition.state), dim=-1, index=actions).squeeze(dim=-1)
            td_target = rewards + (1 - dones) * gamma * policy(transition.next_state).max(dim=-1)[0]
            loss = F.mse_loss(estimate, td_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_timesteps += 1
            if total_timesteps % target_update_interval == 0:
                target_policy.load_state_dict(policy.state_dict())

        acc_rewards.append(t)
        if episode % 100 == 0:
            print(
                f"episode: {episode}\n"
                f"avg. reward: {np.mean(acc_rewards[-100:])}\n"
                f"exploration coeff.: {epsilon}\n"
            )

    env.close()

    return target_policy, acc_rewards


if __name__ == "__main__":
    policy, acc_rewards = deep_q_learning(
        gamma=0.99,
        num_episodes=5000,
        batch_size=64,
        target_update_interval=20
    )
    policy.eval()

    from gym.wrappers import Monitor

    env = Monitor(gym.make("CartPole-v1"), directory="./output", force=True)
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = policy.select_action(state)
        state, _, done, _ = env.step(action)

        if done:
            env.close()
            break
