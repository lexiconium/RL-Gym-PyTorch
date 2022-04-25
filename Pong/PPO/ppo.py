from collections import deque, namedtuple

import gym
import numpy as np
import torch
from torch import nn, no_grad, optim
from torch.distributions import Categorical
from torch.nn import functional as F
from skimage import color, transform
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def in_numpy(fn):
    def convert_and_execute(cls, *args, **kwargs):
        args = [arg.detach().cpu().numpy() if isinstance(arg, torch.Tensor) else np.asarray(arg) for arg in args]
        kwargs = {
            key: arg.detach().cpu().numpy() if isinstance(arg, torch.Tensor) else np.asarray(arg)
            for key, arg in kwargs.items()
        }
        return fn(cls, *args, **kwargs)

    return convert_and_execute


def make_input_valid(x):
    x = torch.as_tensor(x).float()
    if len(x.shape) == 3:
        x = x.unsqueeze(dim=0)
    return x.to(device)


def in_torch(fn):
    def convert_and_execute(cls, *args, **kwargs):
        args = [make_input_valid(arg) for arg in args]
        kwargs = {key: make_input_valid(arg) for key, arg in kwargs.items()}
        return fn(cls, *args, **kwargs)

    return convert_and_execute


class FrameSkipStackedEnv:
    def __init__(self, env: gym.Env, num_skip: int = 4, num_stack: int = 4):
        self.env = env
        self.num_skip = num_skip
        self.num_stack = num_stack
        self.frames = None
        self.frame_buffer = deque(maxlen=num_skip)

    @staticmethod
    def _process(observation: np.ndarray):
        observation = color.rgb2gray(observation)
        observation = transform.resize(observation, (84, 84))
        return observation

    def reset(self):
        observation = self._process(self.env.reset())
        self.frames = deque(
            [observation for _ in range(self.num_stack)],
            maxlen=self.num_stack
        )
        return np.stack(list(self.frames), axis=0)

    def skip_step(self, action: np.ndarray):
        action = action.item()
        reward = 0
        for _ in range(self.num_skip):
            observation, _reward, done, info = self.env.step(action)
            self.frame_buffer.append(self._process(observation))
            reward += _reward
            if done:
                break
        return np.max(list(self.frame_buffer), axis=0), reward, done, info

    @in_numpy
    def step(self, action: np.ndarray):
        observation, reward, done, info = self.skip_step(action)
        self.frames.append(observation)
        return np.stack(list(self.frames), axis=0), reward, done, info

    @property
    def num_actions(self):
        return self.env.action_space.n


def matching_torch_dtype(dtype: np.dtype):
    if dtype == np.bool8:
        return torch.bool
    if dtype in (np.int8, np.int16, np.int32, np.int64):
        return torch.long
    if dtype in (np.float16, np.float32, np.float64):
        return torch.float


class RolloutBuffer:
    preprocess_fields = namedtuple("fields", ["observation", "value", "action", "log_prob", "reward", "done"])
    postprocess_fields = namedtuple(
        "fields",
        ["observation", "value", "action", "log_prob", "reward", "done", "advantage", "target"]
    )

    def __init__(self):
        self.buffer = []
        self.processed = False

    @in_numpy
    def append(self, **kwargs):
        self.buffer.append(self.preprocess_fields(**kwargs))

    @in_numpy
    def process(self, gamma: float, lambda_gae: float, next_value: np.ndarray):
        advantage, advantages = 0, []
        target, targets = next_value, []
        for transition in self.buffer[::-1]:
            delta = transition.reward + gamma * ~transition.done * next_value - transition.value
            advantage = delta + gamma * lambda_gae * ~transition.done * advantage
            advantages.append(advantage)

            target = transition.reward + gamma * ~transition.done * target
            targets.append(target)

            next_value = transition.value

        self.buffer = self.postprocess_fields(
            *map(np.array, [*zip(*self.buffer), advantages[::-1], targets[::-1]])
        )
        self.processed = True

    def rollout(self, batch_size: int):
        dtypes = [matching_torch_dtype(_field[0].dtype) for _field in self.buffer]

        idxs = np.random.permutation(len(self))
        begin = 0
        while begin < len(self):
            yield self.postprocess_fields(
                *[
                    torch.as_tensor(_field[idxs[begin:begin + batch_size]], dtype=dtypes[i]).to(device)
                    for i, _field in enumerate(self.buffer)
                ]
            )
            begin += batch_size

    def reset(self):
        self.__init__()

    def __len__(self):
        return len(self.buffer[0]) if self.processed else len(self.buffer)


class ImageEmbedding(nn.Module):
    def __init__(self, num_stack: int, emb_dim: int):
        super(ImageEmbedding, self).__init__()

        self.embedding = nn.Sequential(
            nn.Conv2d(num_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(7 * 7 * 64, emb_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        return self.embedding(x)


class ActorCritic(nn.Module):
    def __init__(self, num_stack: int, num_actions: int):
        super(ActorCritic, self).__init__()

        emb_dim = 512
        self.embedding = ImageEmbedding(num_stack=num_stack, emb_dim=emb_dim)
        self.actor = nn.Sequential(
            nn.Linear(emb_dim, num_actions),
            nn.LogSoftmax(dim=-1)
        )
        self.critic = nn.Linear(emb_dim, 1)

    @in_torch
    def forward(self, observation: torch.Tensor):
        emb = self.embedding(observation)
        return self.actor(emb), self.critic(emb)

    @no_grad()
    @in_torch
    def action(self, observation: torch.Tensor):
        emb = self.embedding(observation)
        return self.actor(emb).argmax(dim=-1)

    @no_grad()
    @in_torch
    def value(self, observation: torch.Tensor):
        emb = self.embedding(observation)
        return self.critic(emb)


class PPOLog:
    episode: int = 0
    episode_reward: float = 0
    episode_rewards: list = []
    ppo_loss: list = []
    value_loss: list = []
    entropy: list = []

    def done(self):
        self.episode += 1
        self.episode_rewards.append(self.episode_reward)
        self.episode_reward = 0

        print(self)

    @in_numpy
    def append(self, **kwargs):
        for key, value in kwargs.items():
            getattr(self, key).append(value)

    @property
    def avg_ep_reward(self):
        return np.mean(self.episode_rewards[-100:])

    def __repr__(self):
        if self.ppo_loss:
            return (
                f"Episode: {self.episode - 1}\n"
                f"+ Avg. Episode Reward: {np.mean(self.episode_rewards[-100:]):.2f}\n"
                f"+ Avg. PPO Loss: {np.mean(self.ppo_loss[-100:]):.4f}\n"
                f"+ Avg. Value Loss: {np.mean(self.value_loss[-100:]):.4f}\n"
                f"+ Avg. Entropy: {np.mean(self.entropy[-100:]):.4f}\n"
            )
        return "At least one training loop has to be done before printing the results."


def proximal_policy_optimization(
    num_stack: int = 4,
    gamma: float = 0.99,
    lambda_gae: float = 0.95,
    clip_range: float = 0.2,
    c_vf: float = 0.5,
    c_entropy: float = 0.01,
    learning_rate: float = 3e-4,
    horizon: int = 2048,
    num_iterations: int = 2000,
    num_epochs: int = 10,
    batch_size: int = 64,
    max_grad_norm: float = 0.5,
):
    env = gym.make("PongNoFrameskip-v4")
    env = FrameSkipStackedEnv(env, num_stack=num_stack)
    num_actions = env.num_actions

    rollout_buffer = RolloutBuffer()
    policy = ActorCritic(num_stack=num_stack, num_actions=num_actions).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    log = PPOLog()

    observation = env.reset()
    for iteration in range(1, num_iterations + 1):
        for t in range(horizon):
            with torch.no_grad():
                log_prob, value = policy(observation)
                log_prob, value = log_prob.view(-1), value.item()

            distribution = Categorical(probs=torch.exp(log_prob))
            action = distribution.sample()

            next_observation, reward, done, _ = env.step(action=action)
            log.episode_reward += reward
            rollout_buffer.append(
                observation=observation,
                value=value,
                action=action,
                log_prob=log_prob,
                reward=reward,
                done=done
            )
            observation = next_observation

            if done:
                log.done()
                observation = env.reset()

        next_value = policy.value(next_observation).item()
        rollout_buffer.process(
            gamma=gamma,
            lambda_gae=lambda_gae,
            next_value=next_value
        )

        for _ in tqdm(range(num_epochs), "training..."):
            for batch in rollout_buffer.rollout(batch_size):
                log_probs, values = policy(batch.observation)
                values = values.view(-1)

                distribution = Categorical(probs=torch.exp(log_probs))
                action_log_probs = distribution.log_prob(batch.action)
                old_action_log_probs = torch.gather(batch.log_prob, dim=-1, index=batch.action.unsqueeze(-1)).view(-1)
                ratio = torch.exp(action_log_probs - old_action_log_probs)

                advantages = batch.advantage.to(device)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ppo_loss = torch.minimum(
                    advantages * ratio,
                    advantages * torch.clip(ratio, 1 - clip_range, 1 + clip_range)
                ).mean()

                targets = batch.target
                value_loss = F.mse_loss(values, targets)

                entropy = distribution.entropy().mean()

                log.append(
                    ppo_loss=ppo_loss,
                    value_loss=value_loss,
                    entropy=entropy
                )

                loss = -(ppo_loss - c_vf * value_loss + c_entropy * entropy)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        rollout_buffer.reset()

    return policy, log


if __name__ == "__main__":
    policy, log = proximal_policy_optimization()
