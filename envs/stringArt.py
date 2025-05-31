import gym
from gym import spaces
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

CANVAS_SIZE = 500
NUM_PINS = 288
NUM_LINES = 1000
BATCH_SIZE = 32
IMAGE_SAVE_DIR = "/content/images"
LOG_DIR = "/content/logs"

# String Art Utilities

def generate_nail_positions(num_pins, radius, center):
    angles = np.linspace(0, 2 * np.pi, num_pins, endpoint=False)
    positions = [
        (
            int(center[0] + radius * np.cos(angle)),
            int(center[1] + radius * np.sin(angle))
        ) for angle in angles
    ]
    return positions

def draw_line(img, pt1, pt2, color=255, thickness=1):
    import cv2
    cv2.line(img, pt1, pt2, color, thickness)

def calculate_similarity(canvas, target):
    return -F.mse_loss(torch.tensor(canvas, dtype=torch.float32), torch.tensor(target, dtype=torch.float32))

# String Art Environment

class StringArtEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.index = 0
        self.canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
        self.target = None
        self.used_lines = 0
        self.last_similarity = 0
        self.radius = CANVAS_SIZE // 2 - 10
        self.center = (CANVAS_SIZE // 2, CANVAS_SIZE // 2)
        self.nail_positions = generate_nail_positions(NUM_PINS, self.radius, self.center)

        self.action_space = gym.spaces.Discrete(NUM_PINS * NUM_PINS)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
        self.reward_range = (-float('inf'), 0)

        if not os.path.exists(IMAGE_SAVE_DIR):
            os.makedirs(IMAGE_SAVE_DIR)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.canvas.fill(0)
        self.used_lines = 0
        self.last_similarity = 0
        self.target = self.dataset[self.index % len(self.dataset)]
        self.index += 1
        return self.canvas.copy()

    def step(self, action):
        pt1 = self.nail_positions[action // NUM_PINS]
        pt2 = self.nail_positions[action % NUM_PINS]
        draw_line(self.canvas, pt1, pt2)

        obs = self.canvas.copy()
        similarity = calculate_similarity(obs, self.target)
        reward = similarity - self.last_similarity
        self.last_similarity = similarity
        self.used_lines += 1

        terminated = self.used_lines >= NUM_LINES
        truncated = False
        info = {}

        return obs, reward.item(), terminated, truncated, info

    def render(self):
        plt.imshow(self.canvas, cmap='gray')
        plt.show()