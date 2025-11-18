import cv2
import numpy as np
from collections import deque
import random

def preprocess_observation(obs):
    # Convert to grayscale
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    # Resize to 84x84 (Atari standard)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    return obs / 255.0

class FrameStack:
    def __init__(self, k=4):
        self.k = k
        self.frames = deque(maxlen=k)

    def reset(self, obs):
        processed = preprocess_observation(obs)
        for _ in range(self.k):
            self.frames.append(processed)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        processed = preprocess_observation(obs)
        self.frames.append(processed)
        return np.stack(self.frames, axis=0)


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.stack(states), actions, rewards, np.stack(next_states), dones

    def __len__(self):
        return len(self.buffer)
