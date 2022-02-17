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


class ReplayBuffer:
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


class UtilsMixin:
    def polyak_averaging(self: nn.Module, source: nn.Module, tau: float):
        for p_self, p_src in zip(self.parameters(), source.parameters()):
            p_self.data.copy_(p_src.data * tau + p_self.data * (1 - tau))


class Critic(nn.Module, UtilsMixin):
    def __init__(self):
        super(Critic, self).__init__()

        self.q = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    @to_torch
    def forward(self, state: torch.Tensor, action: torch.Tensor):
        cat = torch.cat([state, action], dim=-1)
        return self.q(cat)


class Actor(nn.Module, UtilsMixin):
    def __init__(self):
        super(Actor, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 12),
            nn.ReLU(),
            nn.Linear(12, 1),
            nn.Tanh()
        )

    @to_torch
    def forward(self, state: torch.Tensor):
        return 2 * self.mu(state)


class OrnsteinUhlenbeckNoise:
    def __init__(
        self,
        theta: float,
        sigma: float,
        mu: np.ndarray
    ):
        self.theta = theta
        self.sigma = sigma
        self.mu = mu

        self.x = np.zeros_like(mu)

    def __call__(self, dt: float):
        wiener_process = np.random.normal(loc=0, scale=dt, size=self.x.shape)
        self.x += self.theta * (self.mu - self.x) * dt + self.sigma * np.sqrt(dt) * wiener_process
        return self.x


def deep_deterministic_policy_gradient(
    gamma: float = 0.99,
    tau: float = 0.005,
    num_episodes: int = 500,
    batch_size: int = 64
):
    env = gym.make("Pendulum-v0")

    critic = Critic()
    actor = Actor()

    target_critic = Critic()
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()
    target_actor = Actor()
    target_actor.load_state_dict(actor.state_dict())
    target_actor.eval()

    replay_buffer = ReplayBuffer(capacity=100000)

    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)

    acc_rewards = []
    for episode in range(num_episodes):
        noise = OrnsteinUhlenbeckNoise(theta=0.1, sigma=0.1, mu=np.zeros(1))
        state = env.reset()
        acc_reward = 0
        for _ in count(1):
            with no_grad():
                action = np.clip(actor(state).numpy() + noise(dt=0.01), -2, 2)

            next_state, reward, done, _ = env.step(action)
            replay_buffer.append(
                state=state,
                action=action,
                reward=reward / 100,
                next_state=next_state,
                done=done
            )
            acc_reward += reward

            if done:
                break
            state = next_state

            if len(replay_buffer) < batch_size:
                continue

            transition = replay_buffer.sample(batch_size)

            states = transition.state
            actions = transition.action
            rewards = torch.as_tensor(transition.reward).float()
            dones = torch.as_tensor(transition.done).long()
            next_states = transition.next_state

            estimate = critic(states, actions).flatten()
            with no_grad():
                td_target = rewards + gamma * (1 - dones) * target_critic(next_states,
                                                                          target_actor(next_states)).flatten()
            critic_loss = F.mse_loss(estimate, td_target)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            actor_loss = -critic(states, actor(states)).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            target_critic.polyak_averaging(critic, tau=tau)
            target_actor.polyak_averaging(actor, tau=tau)

        acc_rewards.append(acc_reward)
        if episode % 100 == 0:
            print(
                f"episode: {episode}\n"
                f"avg. reward: {np.mean(acc_rewards[-100:])}\n"
            )

    env.close()

    return target_actor, acc_rewards


if __name__ == "__main__":
    policy, acc_rewards = deep_deterministic_policy_gradient(
        gamma=0.99,
        tau=0.005,
        num_episodes=500,
        batch_size=64
    )

    from gym.wrappers import Monitor

    env = Monitor(gym.make("Pendulum-v0"), directory="./output", force=True)
    state = env.reset()
    done = False
    while not done:
        env.render()
        with no_grad():
            action = policy(state).numpy()
        state, _, done, _ = env.step(action)

        if done:
            env.close()
            break
