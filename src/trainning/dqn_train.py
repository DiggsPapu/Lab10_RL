import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from src.utils import FrameStack, ReplayBuffer
import os
import ale_py 

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
    action_space = env.action_space.n

    frames = FrameStack()
    buffer = ReplayBuffer()

    model = DQNNetwork(action_space).cuda()
    target_model = DQNNetwork(action_space).cuda()
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=lr)

    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.1

    for episode in range(episodes):
        obs, _ = env.reset()
        state = frames.reset(obs)

        total_reward = 0

        done, truncated = False, False

        while not (done or truncated):
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0).cuda())
                    action = torch.argmax(q_values).item()

            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = frames.step(next_obs)

            buffer.push((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

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

                loss = nn.MSELoss()(q_values, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        if episode % 50 == 0:
            target_model.load_state_dict(model.state_dict())
            print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    torch.save(model.state_dict(), os.path.join(r'C:\\Users\\diego\\Downloads\\UVG\\CODING\\Semestre10\\AprendizajePorRefuerzo\\Lab10', "dqn_galaxian.pt"))
    env.close()
    return model
