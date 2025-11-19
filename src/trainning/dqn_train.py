import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from src.utils import FrameStack, ReplayBuffer, preprocess_observation
import os
import ale_py
import random

class DQNNetwork(nn.Module):
    def __init__(self, action_space):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*64, 512), nn.ReLU(),
            nn.Linear(512, action_space)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)



def train_dqn(episodes=5000, batch_size=32, gamma=0.99, lr=1e-4):

    env = gym.make("ALE/Galaxian-v5", render_mode=None, full_action_space=True)

    VALID_ACTIONS = [0, 1, 2, 3, 4, 5]
    action_space = len(VALID_ACTIONS)

    frames = FrameStack()
    buffer = ReplayBuffer()

    model = DQNNetwork(action_space).cuda()
    target_model = DQNNetwork(action_space).cuda()
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.35

    step_count = 0
    reward_end = 0
    for episode in range(episodes):
        obs, _ = env.reset()
        state = frames.reset(obs)

        total_reward = 0
        done, truncated = False, False

        while not (done or truncated):

            step_count += 1

            if np.random.rand() < epsilon:
                action = random.choice(VALID_ACTIONS)
            else:
                with torch.no_grad():
                    q_values = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0).cuda())
                    action = VALID_ACTIONS[torch.argmax(q_values).item()]

            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = frames.step(next_obs)
            reward_end += reward
            reward = np.sign(reward)

            buffer.push((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # --- TRAINING STEP ---
            if len(buffer) > batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                states = torch.tensor(states, dtype=torch.float32).cuda()
                next_states = torch.tensor(next_states, dtype=torch.float32).cuda()
                actions = torch.tensor(actions).cuda()
                rewards = torch.tensor(rewards).cuda()
                dones = torch.tensor(dones, dtype=torch.float32).cuda()

                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    next_q = target_model(next_states).max(dim=1)[0]
                    targets = rewards + gamma * next_q * (1 - dones)

                loss = loss_fn(q_values, targets)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()

            # Update target network every 5000 steps
            if step_count % 5000 == 0:
                target_model.load_state_dict(model.state_dict())

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.2f}, Reward End: {reward_end}")

    torch.save(model.state_dict(), "dqn_galaxian.pt")
    env.close()
    return model

def dqn_policy(obs, action_space=None, model_path="dqn_galaxian.pt"):
    dqn_model = DQNNetwork(action_space=6).cuda()
    dqn_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    dqn_model.eval()
        
    state = preprocess_observation(obs).astype(np.float32)
    with torch.no_grad():
        q_values = dqn_model(torch.tensor(state, dtype=torch.float32).unsqueeze(0).cuda())
        relevant_actions = [0, 1, 2, 3, 4, 5]
        q_values = q_values[:, relevant_actions]
        return relevant_actions[torch.argmax(q_values).item()]
