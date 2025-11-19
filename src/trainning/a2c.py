import os
import time
from datetime import datetime
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Import your preprocessing and FrameStack
from src.utils import FrameStack, preprocess_observation  # adjust import if needed

# Config: use same VALID_ACTIONS as your DQN inference
VALID_ACTIONS = [0, 1, 2, 3, 4, 5]  # map action index -> actual ALE action
NUM_ACTIONS = len(VALID_ACTIONS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512), nn.ReLU()
        )
        self.policy_logits = nn.Linear(512, n_actions)
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        """
        x: tensor shape (B, 4, 84, 84), dtype float32, in [0,1]
        returns: logits (B, n_actions), values (B)
        """
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        logits = self.policy_logits(x)
        value = self.value(x).squeeze(-1)
        return logits, value


def compute_returns_and_advantages(rewards, dones, values, next_value, gamma, n_steps):
    """
    rewards: list len n_steps
    dones: list len n_steps (bool)
    values: tensor len n_steps (v(s_t))
    next_value: scalar bootstrap value v(s_{t+n})
    Returns:
      returns tensor len n_steps
      advantages tensor len n_steps (returns - values)
    """
    returns = []
    R = next_value
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * (1.0 - float(dones[step]))
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32, device=values.device)
    advantages = returns - values
    return returns, advantages


def train_a2c(
    env_id="ALE/Galaxian-v5",
    episodes=2000,
    n_steps=5,
    gamma=0.99,
    lr=7e-4,
    entropy_coef=0.01,
    value_coef=0.5,
    max_grad_norm=0.5,
    save_path="a2c_galaxian.pt",
    seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(env_id, render_mode=None, full_action_space=True)
    frames = FrameStack()
    model = ActorCritic(NUM_ACTIONS).to(DEVICE)
    optimizer = optim.RMSprop(model.parameters(), lr=lr, eps=1e-5)

    episode_rewards = []
    global_step = 0

    for episode in range(episodes):
        obs, _ = env.reset(seed=None)
        state = frames.reset(obs)
        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            roll_states = []
            roll_actions = []
            roll_rewards = []
            roll_dones = []
            roll_values = []
            roll_log_probs = []

            # Collect n_steps of interaction (or until episode end)
            for step in range(n_steps):
                global_step += 1
                state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                logits, value = model(state_tensor)
                probs = torch.softmax(logits, dim=-1)
                m = Categorical(probs)
                action_idx = m.sample().item()  # action index in [0..NUM_ACTIONS-1]
                action = VALID_ACTIONS[action_idx]  # map to actual env action

                next_obs, reward, done, truncated, _ = env.step(action)
                next_state = frames.step(next_obs)

                # reward clipping (Atari practice)
                clipped_reward = float(np.sign(reward))

                roll_states.append(state)
                roll_actions.append(action_idx)
                roll_rewards.append(clipped_reward)
                roll_dones.append(done)
                roll_values.append(value.squeeze(0))  # tensor scalar
                roll_log_probs.append(m.log_prob(torch.tensor(action_idx, device=DEVICE)))

                state = next_state
                episode_reward += clipped_reward

                if done or truncated:
                    break

            # Bootstrap value for the next state
            if done or truncated:
                next_value = torch.tensor(0.0, dtype=torch.float32, device=DEVICE)
            else:
                with torch.no_grad():
                    st = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    _, next_value = model(st)
                    next_value = next_value.squeeze(0)

            # Convert lists to tensors
            values = torch.stack(roll_values).to(DEVICE)  # shape (T,)
            log_probs = torch.stack(roll_log_probs).to(DEVICE)  # shape (T,)
            returns, advantages = compute_returns_and_advantages(
                roll_rewards, roll_dones, values, next_value, gamma, len(roll_rewards)
            )
            # Policy loss (we do gradient ascent on expected advantage)
            policy_loss = -(log_probs * advantages.detach()).mean()
            # Value loss (MSE)
            value_loss = 0.5 * (advantages.pow(2).mean())
            # Entropy bonus to encourage exploration
            # compute entropy across the batch of logits
            # Recompute logits for the stored states for entropy:
            states_tensor = torch.tensor(np.stack(roll_states), dtype=torch.float32, device=DEVICE)
            logits_batch, _ = model(states_tensor)
            dist = torch.distributions.Categorical(logits=logits_batch)
            entropy = dist.entropy().mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        episode_rewards.append(episode_reward)
        if episode % 10 == 0 or episode == episodes - 1:
            avg_last = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 1 else episode_reward
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Episode {episode:04d} | "
                  f"Reward: {episode_reward:.1f} | Avg50: {avg_last:.2f} | Steps: {global_step}")

        # Save periodic checkpoints
        if episode % 100 == 0 and episode > 0:
            torch.save(model.state_dict(), f"{save_path}.ep{episode}")

    # final save
    torch.save(model.state_dict(), save_path)
    env.close()
    return model


# Policy function usable by your recording script
def a2c_policy(obs, action_space=None, model_path="a2c_galaxian.pt"):
    """
    obs: raw env observation (RGB)
    returns: env-compatible action (int)
    """
    if not hasattr(a2c_policy, "model"):
        # lazy load model to keep function simple
        net = ActorCritic(NUM_ACTIONS).to(DEVICE)
        net.load_state_dict(torch.load(model_path, map_location=DEVICE))
        net.eval()
        a2c_policy.model = net

    state = preprocess_observation(obs)
    if state.ndim == 2:  # (84,84)
        stacked = np.stack([state] * 4, axis=0)
    else:
        stacked = state

    with torch.no_grad():
        st = torch.tensor(stacked, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits, _ = a2c_policy.model(st)
        probs = torch.softmax(logits, dim=-1)
        action_idx = torch.argmax(probs, dim=-1).item()
        return VALID_ACTIONS[action_idx]

