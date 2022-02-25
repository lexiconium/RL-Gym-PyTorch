from collections import deque, namedtuple
from itertools import count
import random
from typing import Dict

import gym
import numpy as np
import torch
from torch import nn, no_grad, optim
from torch.nn import functional as F
from tqdm import tqdm


def to_torch(func):
    def convert_and_execute(cls, *args, **kwargs):
        args = [torch.as_tensor(arg).float() for arg in args]
        kwargs = {key: torch.as_tensor(arg).float() for key, arg in kwargs.items()}
        return func(cls, *args, **kwargs)

    return convert_and_execute


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)
        self.fields = namedtuple(
            "fields", ["state", "goal", "achieved_goal", "action", "reward", "next_state", "done"]
        )

    def append(self, **kwargs):
        self.memory.append(self.fields(**kwargs))

    def sample(self, k: int, start: int = None, end: int = None):
        if start is not None and end is not None:
            sampled_fields_list = random.sample(self[start:end], k=min(end - start, k))
        else:
            sampled_fields_list = random.sample(self.memory, k=k)
        return self.fields(*map(np.array, zip(*sampled_fields_list)))

    def stats(self):
        if not len(self):
            return None

        exps = self.fields(*map(np.array, zip(*self[-1000:])))
        return {
            "state": (exps.state.mean(), exps.state.std()),
            "goal": (exps.goal.mean(), exps.goal.std())
        }

    def reset(self):
        self.memory.clear()

    def __getitem__(self, idx):
        return list(self.memory)[idx]

    def __len__(self):
        return len(self.memory)


class UtilsMixin:
    def polyak_averaging(self: nn.Module, source: nn.Module, tau: float):
        for p_self, p_src in zip(self.parameters(), source.parameters()):
            p_self.data.copy_(p_src.data * tau + p_self.data * (1 - tau))

    def set_stats(self: nn.Module, stats: Dict):
        self.stats = stats


class Critic(nn.Module, UtilsMixin):
    def __init__(self, state_dim: int, goal_dim: int, action_dim: int):
        super(Critic, self).__init__()

        self.stats = None
        self.q = nn.Sequential(
            nn.Linear(state_dim + goal_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    @to_torch
    def forward(self, state: torch.Tensor, goal: torch.Tensor, action: torch.Tensor):
        if self.stats is not None:
            state = (state - self.stats["state"][0]) / self.stats["state"][1]
            goal = (goal - self.stats["goal"][0]) / self.stats["goal"][1]

        cat = torch.cat([state, goal, action], dim=-1)
        return self.q(cat)


class Actor(nn.Module, UtilsMixin):
    def __init__(self, state_dim: int, goal_dim: int, action_dim: int):
        super(Actor, self).__init__()

        self.stats = None
        self.mu = nn.Sequential(
            nn.Linear(state_dim + goal_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

    @to_torch
    def forward(self, state: torch.Tensor, goal: torch.Tensor):
        if self.stats is not None:
            state = (state - self.stats["state"][0]) / self.stats["state"][1]
            goal = (goal - self.stats["goal"][0]) / self.stats["goal"][1]

        cat = torch.cat([state, goal], dim=-1)
        return self.mu(cat)


def deep_deterministic_policy_gradient(
    gamma: float = 0.98,
    tau: float = 0.05,
    num_goal_sample: int = 4,
    num_epochs: int = 200,
    num_cycles: int = 50,
    num_episodes: int = 16,
    num_optimization: int = 40,
    num_evals: int = 10,
    batch_size: int = 128
):
    env = gym.make("FetchPickAndPlace-v1")

    state_dim = env.observation_space["observation"].shape[0]
    goal_dim = env.observation_space["desired_goal"].shape[0]
    action_dim = env.action_space.shape[0]

    action_low = env.action_space.low
    action_high = env.action_space.high

    critic = Critic(state_dim=state_dim, action_dim=action_dim, goal_dim=goal_dim)
    actor = Actor(state_dim=state_dim, action_dim=action_dim, goal_dim=goal_dim)

    target_critic = Critic(state_dim=state_dim, action_dim=action_dim, goal_dim=goal_dim)
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()
    target_actor = Actor(state_dim=state_dim, action_dim=action_dim, goal_dim=goal_dim)
    target_actor.load_state_dict(actor.state_dict())
    target_actor.eval()

    episode_buffer = ReplayBuffer(capacity=800)
    replay_buffer = ReplayBuffer(capacity=1_000_000)

    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)

    acc_rewards = []
    critic_losses = []
    actor_losses = []
    for epoch in tqdm(range(num_epochs), "epoch", colour="#00ff00"):
        for cycle in range(num_cycles):
            for episode in range(num_episodes):
                observation = env.reset()
                state, goal = observation["observation"], observation["desired_goal"]

                stats = replay_buffer.stats()
                critic.set_stats(stats)
                actor.set_stats(stats)

                acc_reward = 0
                for t in count(1):
                    if np.random.random() < 0.2:
                        action = env.action_space.sample()
                    else:
                        noise = np.random.normal(
                            loc=np.zeros_like(action_high),
                            scale=0.05 * (action_high - action_low)
                        )
                        with no_grad():
                            action = np.clip(actor(state, goal).numpy() + noise, action_low, action_high)

                    next_obs, reward, done, _ = env.step(action)
                    episode_buffer.append(
                        state=state,
                        goal=goal,
                        achieved_goal=next_obs["achieved_goal"],
                        action=action,
                        reward=reward,
                        next_state=next_obs["observation"],
                        done=done
                    )
                    replay_buffer.append(
                        state=state,
                        goal=goal,
                        achieved_goal=next_obs["achieved_goal"],
                        action=action,
                        reward=reward,
                        next_state=next_obs["observation"],
                        done=done
                    )
                    acc_reward += reward

                    if done:
                        acc_rewards.append(acc_reward)
                        break
                    state = next_obs["observation"]

                for idx in range(t - 1):
                    transition = episode_buffer[idx]
                    sampled = episode_buffer.sample(k=num_goal_sample, start=idx + 1, end=t)
                    for future_achieved_goal in sampled.achieved_goal:
                        replay_buffer.append(
                            state=transition.state,
                            goal=future_achieved_goal,
                            achieved_goal=transition.achieved_goal,
                            action=transition.action,
                            reward=env.compute_reward(
                                achieved_goal=transition.achieved_goal,
                                desired_goal=future_achieved_goal,
                                info=None
                            ),
                            next_state=transition.next_state,
                            done=transition.done
                        )
                episode_buffer.reset()

            stats = replay_buffer.stats()
            critic.set_stats(stats)
            actor.set_stats(stats)
            target_critic.set_stats(stats)
            target_actor.set_stats(stats)
            for _ in range(num_optimization):
                transitions = replay_buffer.sample(batch_size)

                states = transitions.state
                actions = transitions.action
                goals = transitions.goal
                rewards = torch.as_tensor(transitions.reward).float()
                dones = torch.as_tensor(transitions.done).long()
                next_states = transitions.next_state

                estimate = critic(state=states, goal=goals, action=actions).flatten()
                with no_grad():
                    next_values = target_critic(next_states, goals, target_actor(next_states, goals)).flatten()

                td_target = torch.clamp(rewards + gamma * (1 - dones) * next_values, 1 / (gamma - 1), 0)
                critic_loss = F.mse_loss(estimate, td_target)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                critic_losses.append(critic_loss.detach().numpy())

                actor_loss = -critic(states, goals, actor(states, goals)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                actor_losses.append(actor_loss.detach().numpy())

                target_critic.polyak_averaging(critic, tau=tau)
                target_actor.polyak_averaging(actor, tau=tau)

        eval_rewards = []
        eval_reward = 0
        for _ in range(num_evals):
            observation = env.reset()
            state, goal = observation["observation"], observation["desired_goal"]
            for _ in count(1):
                with no_grad():
                    action = target_actor(state, goal).numpy()
                next_obs, reward, done, _ = env.step(action)
                eval_reward += reward
                state = next_obs["observation"]

                if done:
                    eval_rewards.append(eval_reward)
                    eval_reward = 0
                    break

        print(
            f"epoch: {epoch}\n"
            f"avg. running reward: {np.mean(acc_rewards[-100:])}\n"
            f"avg. eval. reward: {np.mean(eval_rewards)}\n"
            f"avg. critic loss: {np.mean(critic_losses[-100:])}\n"
            f"avg. actor loss: {np.mean(actor_losses[-100:])}\n"
            f"buffer length: {len(replay_buffer)}\n"
        )

    env.close()

    return target_actor, acc_rewards


if __name__ == "__main__":
    policy, acc_rewards = deep_deterministic_policy_gradient(
        gamma=0.98,
        tau=0.05,
        num_goal_sample=4,
        num_epochs=100,
        num_cycles=50,
        num_episodes=16,
        num_optimization=40,
        num_evals=10,
        batch_size=128
    )

    from gym.wrappers import Monitor

    env = Monitor(gym.make("FetchPickAndPlace-v1"), directory="./output", force=True)
    observation = env.reset()
    state, goal = observation["observation"], observation["desired_goal"]
    done = False
    while not done:
        env.render()
        with no_grad():
            action = policy(state, goal).numpy()
        next_obs, _, done, _ = env.step(action)
        state = next_obs["observation"]

        if done:
            env.close()
            break
