from collections import namedtuple
from typing import Optional

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
    def __init__(self):
        self.buffer = []
        self.fields = namedtuple("fields", ["state", "value", "action", "log_prob", "reward", "done"])

    def append(self, **kwargs):
        self.buffer.append(self.fields(**{
            key: np.array(arg) for key, arg in kwargs.items()
        }))

    def calc_advantage_and_target(self, gamma: float, lambda_gae: float, next_value: np.ndarray):
        advantage, advantages = 0, []
        target, targets = next_value, []
        for record in self.buffer[::-1]:
            delta = record.reward + gamma * (1 - record.done) * next_value - record.value
            advantage = delta + gamma * lambda_gae * (1 - record.done) * advantage
            advantages.append(advantage)

            target = record.reward + gamma * (1 - record.done) * target
            targets.append(target)

            next_value = record.value

        self.fields = namedtuple(
            "fields",
            ["state", "value", "action", "log_prob", "reward", "done", "advantage", "target"]
        )
        self.buffer = self.fields(*map(np.array, [*zip(*self.buffer), advantages[::-1], targets[::-1]]))

    def rollout(self, batch_size: int):
        indices = np.random.permutation(len(self.buffer[0]))
        start_idx = 0
        while start_idx < len(self.buffer[0]):
            yield self.fields(*[_field[indices[start_idx:start_idx + batch_size]] for _field in self.buffer])
            start_idx += batch_size

    def reset(self):
        self.__init__()


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

    def forward(self, state: torch.Tensor):
        latent = self.sync(state)
        return self.actor(latent), self.critic(latent)

    @to_torch
    def evaluate(self, state: torch.Tensor, action: Optional[torch.Tensor] = None):
        logits, value = self(state)
        distribution = Categorical(logits=logits)

        if action is None:
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
            return value, action, log_prob

        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        return value, log_prob, entropy

    @no_grad()
    @to_torch
    def select_action(self, state: torch.Tensor):
        latent = self.sync(state)
        logits = self.actor(latent)
        return logits.argmax(dim=-1)

    @no_grad()
    @to_torch
    def estimate(self, state: torch.Tensor):
        latent = self.sync(state)
        return self.critic(latent)


def proximal_policy_optimization(
    gamma: float = 0.99,
    lambda_gae: float = 0.95,
    clip_range: float = 0.2,
    c_vf: float = 1,
    c_entropy: float = 0.01,
    horizon: int = 2048,
    num_iterations: int = 100,
    batch_size: int = 64
):
    env = gym.make("CartPole-v1")

    rollout_buffer = RolloutBuffer()
    policy = ActorCritic()

    optimizer = optim.Adam(policy.parameters(), lr=5e-4)

    state = env.reset()
    duration = 0
    durations = []
    for iteration in range(num_iterations):
        for _ in range(horizon):
            with torch.no_grad():
                value, action, log_prob = policy.evaluate(state)
            next_state, reward, done, _ = env.step(action.item())
            rollout_buffer.append(
                state=state,
                value=value,
                action=action,
                log_prob=log_prob,
                reward=reward / 100,
                done=done
            )

            state = next_state
            duration += 1

            if done:
                state = env.reset()
                durations.append(duration)
                duration = 0

        next_value = policy.estimate(state).numpy()
        rollout_buffer.calc_advantage_and_target(
            gamma=gamma,
            lambda_gae=lambda_gae,
            next_value=next_value
        )

        for batch in rollout_buffer.rollout(batch_size):
            values, log_probs, entropy = policy.evaluate(batch.state, batch.action)

            old_log_probs = torch.as_tensor(batch.log_prob).flatten().float()
            ratio = torch.exp(log_probs - old_log_probs)

            advantages = torch.as_tensor(batch.advantage).flatten().float()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            clip_loss = torch.min(
                advantages * ratio,
                advantages * torch.clip(ratio, 1 - clip_range, 1 + clip_range)
            ).mean()

            targets = torch.as_tensor(batch.target).flatten().float()
            value_loss = F.mse_loss(values.flatten(), targets)

            entropy = entropy.mean()
            loss = -(clip_loss - c_vf * value_loss + c_entropy * entropy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        rollout_buffer.reset()

        if iteration % 10 == 0:
            print(
                f"iteration: {iteration}\n"
                f"avg. reward: {np.mean(durations[-100:])}\n"
            )

    env.close()

    return policy, durations


if __name__ == "__main__":
    policy, durations = proximal_policy_optimization(
        gamma=0.99,
        lambda_gae=0.95,
        clip_range=0.2,
        c_vf=1,
        c_entropy=0.01,
        horizon=2048,
        num_iterations=150,
        batch_size=64
    )
    policy.eval()

    from gym.wrappers import Monitor

    env = Monitor(gym.make("CartPole-v1"), directory="./output", force=True)
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = policy.select_action(state)
        state, _, done, _ = env.step(action.item())

        if done:
            env.close()
            break
