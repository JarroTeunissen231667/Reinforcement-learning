# ----------------------------------------------------------
## Task 11 - Train a PPO model Jarro Teunissen
# ----------------------------------------------------------
import os
import argparse
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Import the necessary libraries
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import wandb
import typing_extensions as TypeIs
import tensorflow

# After running the setup script we can upload this
from ot2_env_wrapper_rework3 import OT2Env

# Load the API key for wandb
os.environ['WANDB_API_KEY'] = 'c9df9474ff41da6be45ba2e10ff13f6749aec8dd'

# Initialize Weights & Biases
run = wandb.init(project="2024-Y2B-RoboSuite", sync_tensorboard=True)

# Define the environment
env = OT2Env()

# Initialize wandb again (if you want a separate run name for Task 11)
run = wandb.init(project="task11", sync_tensorboard=True)
save_path = f"models/{run.id}"
os.makedirs(save_path, exist_ok=True)

# Set the amount of steps for the training
timesteps = 5_000_000

# Define the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--gamma", type=float, default=0.98)
parser.add_argument("--policy", type=str, default="MlpPolicy")
parser.add_argument("--clip_range", type=float, default=0.15)
parser.add_argument("--value_coefficient", type=float, default=0.5)
# This argument allows us to pass JSON-encoded policy kwargs, e.g. '{"net_arch":[256,256]}'
parser.add_argument("--policy_kwargs", type=str, default="{}")

args = parser.parse_args()

# Parse the JSON for policy_kwargs
policy_kwargs = json.loads(args.policy_kwargs)

# Create the PPO Model
model = PPO(
    policy=args.policy,
    env=env,
    verbose=1,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    n_epochs=args.n_epochs,
    gamma=args.gamma,
    clip_range=args.clip_range,
    vf_coef=args.value_coefficient,
    policy_kwargs=policy_kwargs,  # Pass the parsed policy kwargs here
    tensorboard_log=f"runs/{run.id}"
)

# Callback for wandb
wandb_callback = WandbCallback(
    model_save_freq=1_000_000,
    model_save_path=f"modelspc/{run.id}",
    verbose=2
)

# Train the model
model.learn(
    total_timesteps=timesteps,
    callback=wandb_callback,
    progress_bar=True,
    reset_num_timesteps=False,
    tb_log_name=f"runs/{run.id}"
)

# Optionally save the final model locally
# model.save(f"models/{run.id}/{timesteps}_baseline")

# Save the model to W&B
wandb.save(f"models/{run.id}/{timesteps}_baseline")


# python train.py --learning_rate 0.0001 --batch_size 32 --n_steps 2048 --n_epochs 10 --gamma 0.98 --policy MlpPolicy --clip_range 0.15 --value_coefficient 0.5
# python trainownpc.py --learning_rate 0.0003 --batch_size 64 --n_steps 2048 --n_epochs 10 --gamma 0.99 --policy MlpPolicy --clip_range 0.2 --value_coefficient 0.5
#python trainownpc.py --learning_rate 0.0003 --batch_size 128 --n_steps 1024 --n_epochs 15 --gamma 0.98 --policy MlpPolicy --clip_range 0.2 --value_coefficient 0.5
# python trainownpc.py --learning_rate 0.0003 --batch_size 128 --n_steps 1024 --n_epochs 15 --gamma 0.98 --policy MlpPolicy --clip_range 0.2 --value_coefficient 0.5 --policy_kwargs "{\"net_arch\":[256,256]}"
