from itertools import count

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
    def __init__(self, state_dim: int, action_dim: int, capacity: int):
        self.states = np.ndarray(shape=(capacity, state_dim))
        self.actions = np.ndarray(shape=(capacity, action_dim))
        self.rewards = np.ndarray(shape=(capacity,))
        self.next_states = np.ndarray(shape=(capacity, state_dim))
        self.dones = np.ndarray(shape=(capacity,))

        self.capacity = capacity
        self._ptr = 0
        self._len = 0

    def append(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.states[self._ptr] = state
        self.actions[self._ptr] = action
        self.rewards[self._ptr] = reward
        self.next_states[self._ptr] = next_state
        self.dones[self._ptr] = done

        self._ptr = (self._ptr + 1) % self.capacity
        self._len = min(self._len + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self._len, size=batch_size)

        return (
            torch.FloatTensor(self.states[idxs]),
            torch.FloatTensor(self.actions[idxs]),
            torch.FloatTensor(self.rewards[idxs]),
            torch.FloatTensor(self.next_states[idxs]),
            torch.LongTensor(self.dones[idxs])
        )

    def __len__(self):
        return self._len


class UtilsMixin:
    def polyak_averaging(self: nn.Module, source: nn.Module, tau: float):
        for p_self, p_src in zip(self.parameters(), source.parameters()):
            p_self.data.copy_(p_src.data * tau + p_self.data * (1 - tau))


class Critic(nn.Module, UtilsMixin):
    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()

        self._q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 1)
        )

        self._q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 1)
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
    def __init__(self, state_dim: int, action_dim: int, action_high: torch.Tensor):
        super(Actor, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(state_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh()
        )
        self.action_high = action_high

    @to_torch
    def forward(self, state: torch.Tensor):
        return self.action_high * self.mu(state)


def twin_delayed_deep_deterministic_policy_gradient(
    gamma: float = 0.99,
    tau: float = 0.005,
    exploration_noise: float = 0.1,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    num_episodes: int = 5000,
    batch_size: int = 256,
    start_timestep: int = 25000,
    update_frequency: int = 2,
):
    env = gym.make("Walker2d-v2")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    action_low = env.action_space.low
    action_low_tensor = torch.as_tensor(action_low)
    action_high = env.action_space.high
    action_high_tensor = torch.as_tensor(action_high)

    critic = Critic(state_dim=state_dim, action_dim=action_dim)
    actor = Actor(state_dim=state_dim, action_dim=action_dim, action_high=action_high_tensor)

    target_critic = Critic(state_dim=state_dim, action_dim=action_dim)
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()
    target_actor = Actor(state_dim=state_dim, action_dim=action_dim, action_high=action_high_tensor)
    target_actor.load_state_dict(actor.state_dict())
    target_actor.eval()

    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, capacity=100000)

    critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)
    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)

    total_timesteps = 0
    acc_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        acc_reward = 0
        for t in count(1):
            if len(replay_buffer) > start_timestep:
                with no_grad():
                    noise = np.random.normal(loc=0, scale=exploration_noise, size=action_dim)
                    action = np.clip(actor(state).numpy() + noise, action_low, action_high)
            else:
                action = env.action_space.sample()

            next_state, reward, done, _ = env.step(action)
            replay_buffer.append(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )
            acc_reward += reward
            total_timesteps += 1

            if done:
                break
            state = next_state

            if len(replay_buffer) < start_timestep:
                continue

            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

            q1, q2 = critic(states, actions)

            with no_grad():
                noise = torch.clamp(
                    torch.normal(mean=0, std=policy_noise, size=(action_dim,)),
                    -noise_clip,
                    noise_clip
                )
                next_actions = torch.clamp(
                    target_actor(next_states) + noise,
                    min=action_low_tensor,
                    max=action_high_tensor
                )
                next_q1, next_q2 = target_critic(next_states, next_actions)
            td_target = rewards + gamma * (1 - dones) * torch.min(next_q1, next_q2)

            critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            if total_timesteps % update_frequency == 0:
                actor_loss = -critic.q1(states, actor(states)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                target_critic.polyak_averaging(critic, tau=tau)
                target_actor.polyak_averaging(actor, tau=tau)

        print(
            f"episode: {episode}\n"
            f"episode length: {t}\n"
            f"episode reward: {acc_reward}\n"
            f"current timestep: {total_timesteps}\n"
        )

        acc_rewards.append(acc_reward)
        if episode % 100 == 0:
            print(
                f"episode: {episode}\n"
                f"avg. reward: {np.mean(acc_rewards[-100:])}\n"
            )

    env.close()

    return target_actor, acc_rewards


def wrapper(_):
    return twin_delayed_deep_deterministic_policy_gradient(
        gamma=0.99,
        tau=0.005,
        exploration_noise=0.1,
        policy_noise=0.2,
        noise_clip=0.5,
        num_episodes=3500,
        batch_size=256,
        start_timestep=25000,
        update_frequency=2
    )


if __name__ == "__main__":
    # policy, acc_rewards = twin_delayed_deep_deterministic_policy_gradient(
    #     gamma=0.99,
    #     tau=0.005,
    #     policy_noise=0.2,
    #     noise_clip=0.5,
    #     num_episodes=4000,
    #     batch_size=256,
    #     start_timestep=25000,
    #     update_frequency=2
    # )
    #
    # from gym.wrappers import Monitor
    #
    # env = Monitor(gym.make("Walker2d-v2"), directory="./output", force=True)
    # state = env.reset()
    # done = False
    # while not done:
    #     env.render()
    #     with no_grad():
    #         action = policy(state).numpy()
    #     state, _, done, _ = env.step(action)
    #
    #     if done:
    #         env.close()
    #         break

    # from multiprocessing import get_context
    #
    # with get_context("spawn").Pool(5) as pool:
    #     results = pool.map(wrapper, range(5))

    rewards = {}
    policies = []
    for n in range(5):
        policy, acc_rewards = twin_delayed_deep_deterministic_policy_gradient(
            gamma=0.99,
            tau=0.005,
            exploration_noise=0.1,
            policy_noise=0.2,
            noise_clip=0.5,
            num_episodes=3500,
            batch_size=256,
            start_timestep=25000,
            update_frequency=2
        )
        policies.append(policy)
        rewards[n] = acc_rewards


    import json

    with open("rewards.json", "w") as f:
        # acc_rewards = {n: result[1] for n, result in enumerate(results)}
        json.dump(rewards, f)

    from gym.wrappers import Monitor

    for n, policy in enumerate(policies):
        env = Monitor(gym.make("Walker2d-v2"), directory=f"./output/{n}", force=True)
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
