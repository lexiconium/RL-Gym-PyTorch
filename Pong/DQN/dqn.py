from collections import deque, namedtuple
from itertools import count
import random

import gym
import numpy as np
import torch
from torch import nn, no_grad, optim
from torch.nn import functional as F
from skimage import color, transform
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_torch(func):
    def _adddim(x: torch.Tensor):
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        return x

    def convert_and_execute(cls, *args, **kwargs):
        args = [_adddim(torch.as_tensor(arg)).float().to(device) for arg in args]
        kwargs = {key: _adddim(torch.as_tensor(arg)).float().to(device) for key, arg in kwargs.items()}
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


class FeatureExtractor(nn.Module):
    def __init__(self, num_stacked_frames: int, num_features: int):
        super(FeatureExtractor, self).__init__()

        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=num_stacked_frames, out_channels=64, kernel_size=8, stride=4),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=4, stride=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_features)
        )

    def forward(self, x: torch.Tensor):
        return self.extractor(x)


class UtilsMixin:
    def polyak_averaging(self: nn.Module, source: nn.Module, tau: float):
        for p_self, p_src in zip(self.parameters(), source.parameters()):
            p_self.data.copy_(p_src.data * tau + p_self.data * (1 - tau))


class DQN(nn.Module, UtilsMixin):
    def __init__(self, num_stacked_frames: int, num_actions: int):
        super(DQN, self).__init__()

        self.num_actions = 2

        self.feature_extractor = FeatureExtractor(num_stacked_frames=num_stacked_frames, num_features=16)
        self.dqn = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, self.num_actions)
        )

    @to_torch
    def forward(self, state: torch.Tensor):
        features = self.feature_extractor(state)
        return self.dqn(features)

    @no_grad()
    def select_action(self, state: torch.Tensor, epsilon: float = 0):
        if random.random() < epsilon:
            return random.randrange(self.num_actions)
        return self(state).argmax().cpu().numpy()


def _transform(img: np.ndarray):
    img = img[34:194, :, 0]
    img[img != 144] = 1
    img[img == 144] = 0
    img = transform.resize(img, np.array(img.shape) / 4)
    return img


def deep_q_learning(
    gamma: float = 0.99,
    num_episodes: int = 5000,
    num_epochs: int = 10,
    batch_size: int = 64,
    num_consecutive_frames: int = 4
):
    env = gym.make("Pong-v0")

    num_actions = env.action_space.n

    replay_memory = ReplayMemory(capacity=1_000_000)
    policy = DQN(num_stacked_frames=num_consecutive_frames, num_actions=num_actions).to(device)
    target_policy = DQN(num_stacked_frames=num_consecutive_frames, num_actions=num_actions).to(device)
    target_policy.load_state_dict(policy.state_dict())
    target_policy.eval()

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    collected_frames = 0
    acc_rewards = []
    q_values = []
    acc_reward = 0
    for episode in tqdm(range(num_episodes)):
        _state = _transform(env.reset())
        consecutive_frames = deque([_state for _ in range(num_consecutive_frames)], maxlen=num_consecutive_frames)
        state = np.concatenate([list(consecutive_frames)], axis=0)

        for t in count(1):
            epsilon = max(1 - 0.9 * (collected_frames / 1_000_000), 0.1)
            action = target_policy.select_action(state, epsilon)
            _next_state, reward, done, _ = env.step(action + 2)

            _state = _transform(_next_state)
            consecutive_frames.append(_state)
            next_state = np.concatenate([list(consecutive_frames)], axis=0)

            replay_memory.append(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )

            acc_reward += reward
            collected_frames += 1

            if done:
                break
            state = next_state

            if len(replay_memory) < batch_size or collected_frames % (num_consecutive_frames * batch_size):
                continue

            for _ in range(num_epochs):
                transition = replay_memory.sample(batch_size)

                actions = torch.as_tensor(transition.action).unsqueeze(dim=-1).long().to(device)
                rewards = torch.as_tensor(transition.reward).float().to(device)
                dones = torch.as_tensor(transition.done).long().to(device)

                estimate = torch.gather(policy(transition.state), dim=-1, index=actions).squeeze(dim=-1)
                q_values.append(estimate.detach().cpu().numpy())
                td_target = rewards + (1 - dones) * gamma * target_policy(transition.next_state).max(dim=-1)[0]
                loss = F.mse_loss(estimate, td_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                target_policy.polyak_averaging(source=policy, tau=0.05)

        acc_rewards.append(acc_reward)
        acc_reward = 0

        print(
            f"episode: {episode}\n"
            f"avg. q value: {np.mean(q_values[-100:])}\n"
            f"avg. reward: {np.mean(acc_rewards[-100:])}\n"
            f"exploration coeff.: {epsilon}\n"
            f"buffer length: {len(replay_memory)}"
        )

    env.close()

    return target_policy, acc_rewards


if __name__ == "__main__":
    policy, acc_rewards = deep_q_learning(
        gamma=0.99,
        num_episodes=1000,
        num_epochs=5,
        batch_size=256,
        num_consecutive_frames=2
    )
