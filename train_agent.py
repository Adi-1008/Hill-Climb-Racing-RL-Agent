from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from HillClimbEnv import HillClimbEnv
import time

def train_bot():
    print("Initializing Environment...")
    env = HillClimbEnv()

    # Optional: SB3 has a built-in checker to ensure your custom env follows all rules
    # check_env(env) 

    print("Initializing PPO Model (CNN Policy)...")
    # We use CnnPolicy because our observation space is an image
    # verbose=1 prints training metrics to the console
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./hcr_tensorboard/")

    print("\nModel ready! You have 10 seconds to Alt-Tab into Hill Climb Racing...")
    time.sleep(10)

    print("\n--- Starting Training Loop ---")
    # total_timesteps is the number of 'steps' (frames) the agent will play.
    # 10,000 is a very short run just to verify it's learning.
    # A full model might need 500,000+ timesteps.
    model.learn(total_timesteps=10000)

    print("\n--- Training Complete ---")
    model.save("ppo_hillclimb_v1")
    print("Model saved as 'ppo_hillclimb_v1.zip'")

if __name__ == "__main__":
    train_bot()