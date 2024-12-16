from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import argparse
from clearml import Task
import wandb
import typing_extensions as TypeIs
import tensorflow
import os

# After running the setup script we can upload this
from ot2_env_wrapper import OT2Env
# Load the API key for wandb
os.environ['WANDB_API_KEY'] = 'c9df9474ff41da6be45ba2e10ff13f6749aec8dd'
# Initiate the remote task.
task = Task.init(project_name="Mentor Group D/Group 3/JarroTeunissen",
                    task_name='Experiment1')



task.set_base_docker('deanis/2023y2b-rl:latest')

task.execute_remotely(queue_name="default")

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=1000)

args = parser.parse_args()

env = OT2Env(render=False)

run = wandb.init(project="sb3_RL_OT2",sync_tensorboard=True)

model = PPO('MlpPolicy', env, verbose=1, 
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs, 
            tensorboard_log=f"runs/{run.id}",)

wandb_callback = WandbCallback(model_save_freq=1000,
                                model_save_path=f"models/{run.id}",
                                verbose=2,
                                )

time_steps = 100000
for i in range(10):
    model.learn(total_timesteps=time_steps, callback=wandb_callback, progress_bar=True, reset_num_timesteps=False,tb_log_name=f"runs/{run.id}")
    model.save(f"models/{run.id}/{time_steps*(i+1)}")