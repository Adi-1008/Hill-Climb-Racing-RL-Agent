import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import mss
import pydirectinput
import time
import os

class HillClimbEnv(gym.Env):
    def __init__(self):
        super(HillClimbEnv, self).__init__()
        
        # Action Space: 0 = Do Nothing, 1 = Gas, 2 = Brake
        self.action_space = spaces.Discrete(3)
        
        # Observation Space: 84x84 Grayscale image (Standard for CNNs)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        
        # Initialize Vision
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1] # Assumes game is on the primary monitor
        
        # Load the Game Over template image
        template_path = 'game_over_template.png'
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Missing {template_path}! Please create it using the Snipping Tool.")
        self.game_over_template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        
        self.screen_center_x = 1098
        self.screen_center_y = 654
        self.start_button_x = 1500
        self.start_button_y = 750

    def step(self, action):
        # 1. Execute Action
        if action == 1:
            pydirectinput.keyDown('right')
            time.sleep(0.1) 
            pydirectinput.keyUp('right')
        elif action == 2:
            pydirectinput.keyDown('left')
            time.sleep(0.1)
            pydirectinput.keyUp('left')
        else:
            time.sleep(0.1) 

        # 2. Get the new observation
        next_state = self._get_observation()

        done = self._check_game_over()
        
        # 4. Calculate Reward
        if done:
            reward = -100.0 
        else:
            reward = 1.0   

        truncated = False
        
        return next_state, reward, done, truncated, {}

    def reset(self, seed=None):
        super().reset(seed=seed)
        print("\n--- CRASH DETECTED: Resetting Environment ---")
        
        
        pydirectinput.click(self.screen_center_x, self.screen_center_y)
        time.sleep(2)
        
        pydirectinput.click(self.start_button_x, self.start_button_y)
        time.sleep(2)
        
        print("--- Ready for new episode ---\n")
        return self._get_observation(), {}

    def _get_observation(self):
        
        screenshot = np.array(self.sct.grab(self.monitor))
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)
        resized = cv2.resize(gray, (84, 84))
        return np.expand_dims(resized, axis=-1)

    def _check_game_over(self):
        
        screenshot = np.array(self.sct.grab(self.monitor))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)
        
        res = cv2.matchTemplate(frame, self.game_over_template, cv2.TM_CCOEFF_NORMED)
        
        threshold = 0.80 
        loc = np.where(res >= threshold)
        
        if len(loc[0]) > 0:
            return True
        return False

# just a testing loop
if __name__ == "__main__":
    print("Initializing Hill Climb Racing Environment Test...")
    env = HillClimbEnv()
    
    print("You have 10 seconds to focus the game window...")
    time.sleep(10)
    
    episodes = 3
    for ep in range(episodes):
        print(f"Starting Episode {ep + 1}")
        env.reset()
        done = False
        
        while not done:
            
            next_state, reward, done, _, _ = env.step(1)
            
    print("Test Complete! Auto-reset works perfectly.")