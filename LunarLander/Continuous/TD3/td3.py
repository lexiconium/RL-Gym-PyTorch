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
        args = [arg if isinstance(arg, torch.Tensor) else torch.as_tensor(arg).float() for arg in args]
        kwargs = {
            key: arg if isinstance(arg, torch.Tensor) else torch.as_tensor(arg).float()
            for key, arg in kwargs.items()
        }
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

        self._q1 = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

        self._q2 = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    @to_torch
    def forward(self, state: torch.Tensor, action: torch.Tensor):
        cat = torch.cat([state, action], dim=-1)
        return self._q1(cat).flatten(), self._q2(cat).flatten()

    @to_torch
    def q1(self, state: torch.Tensor, action: torch.Tensor):
        cat = torch.cat([state, action], dim=-1)
        return self._q1(cat).flatten()


class Actor(nn.Module, UtilsMixin):
    def __init__(self):
        super(Actor, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(8, 40),
            nn.ReLU(),
            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Linear(40, 2),
            nn.Tanh()
        )

    @to_torch
    def forward(self, state: torch.Tensor):
        return self.mu(state)


def twin_delayed_deep_deterministic_policy_gradient(
    gamma: float = 0.99,
    tau: float = 0.005,
    exploration_noise: float = 0.1,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    num_episodes: int = 3000,
    batch_size: int = 64,
    start_timestep: int = 10000,
    update_frequency: int = 2,
):
    env = gym.make("LunarLanderContinuous-v2")

    critic = Critic()
    actor = Actor()

    target_critic = Critic()
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()
    target_actor = Actor()
    target_actor.load_state_dict(actor.state_dict())
    target_actor.eval()

    replay_buffer = ReplayBuffer(capacity=100000)

    critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)
    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)

    train_iteration = 0
    acc_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        acc_reward = 0
        for _ in count(1):
            if len(replay_buffer) > start_timestep:
                with no_grad():
                    action = np.clip(
                        actor(state).numpy() + np.random.normal(loc=0, scale=exploration_noise, size=2), -1, 1
                    )
            else:
                action = env.action_space.sample()

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

            if len(replay_buffer) < start_timestep:
                continue

            transition = replay_buffer.sample(batch_size)

            states = transition.state
            actions = torch.as_tensor(transition.action).float()
            rewards = torch.as_tensor(transition.reward).float()
            dones = torch.as_tensor(transition.done).long()
            next_states = transition.next_state

            q1, q2 = critic(states, actions)

            with no_grad():
                noise = torch.clamp(
                    torch.normal(mean=0, std=policy_noise, size=(2,)),
                    -noise_clip,
                    noise_clip
                )
                next_actions = (target_actor(next_states) + noise).clamp(-1, 1)
                next_q1, next_q2 = target_critic(next_states, next_actions)
            td_target = rewards + gamma * (1 - dones) * torch.min(next_q1, next_q2)

            critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            if train_iteration % update_frequency == 0:
                actor_loss = -critic.q1(states, actor(states)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                target_critic.polyak_averaging(critic, tau=tau)
                target_actor.polyak_averaging(actor, tau=tau)

            train_iteration += 1

        acc_rewards.append(acc_reward)
        if episode % 100 == 0:
            print(
                f"episode: {episode}\n"
                f"avg. reward: {np.mean(acc_rewards[-100:])}\n"
            )

    env.close()

    return target_actor, acc_rewards


if __name__ == "__main__":
    policy, acc_rewards = twin_delayed_deep_deterministic_policy_gradient(
        gamma=0.99,
        tau=0.005,
        exploration_noise=0.1,
        policy_noise=0.2,
        noise_clip=0.5,
        num_episodes=3000,
        batch_size=64,
        start_timestep=10000,
        update_frequency=2,
    )

    from gym.wrappers import Monitor

    env = Monitor(gym.make("LunarLanderContinuous-v2"), directory="./output", force=True)
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
