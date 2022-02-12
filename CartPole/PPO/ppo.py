from collections import deque, namedtuple
import random

import gym
import numpy as np
import torch
from torch import nn, no_grad, optim
from torch.distributions import Categorical
from torch.nn import functional as F


def to_torch(func):
    def convert_and_execute(cls, *args, **kwargs):
        args = [torch.as_tensor(arg).float() for arg in args]
        kwargs = {key: torch.as_tensor(arg).float() for key, arg in kwargs.items()}
        return func(cls, *args, **kwargs)

    return convert_and_execute


class RolloutBuffer:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)
        self.fields = namedtuple("fields", ["state", "action", "log_prob", "value", "reward", "next_state", "done"])

    def append(self, **kwargs):
        self.memory.append(self.fields(**kwargs))

    def sample(self, batch_size: int):
        sampled_fields_list = random.sample(self.memory, k=batch_size)
        return self.fields(*map(np.array, zip(*sampled_fields_list)))

    def __len__(self):
        return len(self.memory)


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()

        self.sync = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.actor = nn.Linear(64, 2)
        self.critic = nn.Linear(64, 1)

    def _action_from_logits(self, logits: torch.Tensor, deterministic: bool = False):
        if deterministic:
            return logits.argmax(dim=-1)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action, log_prob

    @to_torch
    @no_grad()
    def forward(self, state: torch.Tensor):
        latent = self.sync(state)
        logits = self.actor(latent)
        return self._action_from_logits(logits), self.critic(latent)

    @to_torch
    @no_grad()
    def sample_action(self, state: np.ndarray, deterministic: bool = False):
        latent = self.sync(state)
        logits = self.actor(latent)
        return self._action_from_logits(logits, deterministic=deterministic)

    @to_torch
    def evaluate_state(self, state: np.ndarray):
        latent = self.sync(state)
        return self.critic(latent)


def proximal_policy_optimization(
    gamma: float = 0.99,
    lambda_gae: float = 0.95,
    c_clip: float = 0.2,
    horizon: int = 2048,
    num_iterations: int = 1000,
    batch_size: int = 32
):
    env = gym.make("CartPole-v1")

    rollout_buffer = None
    policy = ActorCritic()

    optimizer = optim.Adam(policy.parameters(), lr=5e-4)

    state = env.reset()
    duration = 0
    durations = []
    for iteration in range(num_iterations):
        for _ in range(horizon):
            action_info, value = policy(state)
            action, log_prob = action_info
            next_state, reward, done, _ = env.step(action)
            rollout_buffer.append(
                state=state,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=reward,
                next_state=next_state,
                done=done
            )

            duration += 1

            if done:
                state = env.reset()
                durations.append(duration)
                duration = 0

        # TODO: calc advantage estimates and returns

        for batch in rollout_buffer.rollout(batch_size):
            states = batch.state  # -> value
            actions = batch.action  # -> log_prob

            old_log_porb = torch.as_tensor(batch.log_prob).float()
            ratio = torch.exp(log_prob - old_log_porb)

            advantages = torch.as_tensor(batch.advantage).float()
            clip_loss = torch.min(
                advantages * ratio,
                advantages * torch.clip(ratio, 1 - c_clip, 1 + c_clip)
            ).mean()

            targets = torch.as_tensor(batch.target)
            value_loss = F.mse_loss(value, targets)

            entropy = None

            loss = -(clip_loss - value_loss + entropy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        rollout_buffer.reset()

    env.close()

    return policy, durations
