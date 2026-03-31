from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from HillClimbEnv import HillClimbEnv
import time

def train_bot():
    print("Initializing Environment...")
    env = HillClimbEnv()

    print("Initializing PPO Model (CNN Policy)...")
    
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./hcr_tensorboard/")

    print("\nModel ready! You have 10 seconds to Alt-Tab into Hill Climb Racing...")
    time.sleep(10)

    print("\n--- Starting Training Loop ---")
    
    model.learn(total_timesteps=10000)

    print("\n--- Training Complete ---")
    model.save("ppo_hillclimb_v1")
    print("Model saved as 'ppo_hillclimb_v1.zip'")

if __name__ == "__main__":
    train_bot()