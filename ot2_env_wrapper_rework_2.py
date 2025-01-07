import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=100000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps
        self.sim = Simulation(num_agents=1, render=self.render)
        
        # Define the boundaries for the environment
        self.x_min, self.x_max = -0.1872, 0.2531
        self.y_min, self.y_max = -0.1711, 0.2201
        self.z_min, self.z_max = 0.1691, 0.2896
        
        # Define the action space (3D continuous space)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Define the observation space (6D continuous space)
        self.observation_space = spaces.Box(
            low=np.array([self.x_min, self.y_min, self.z_min, -self.x_max, -self.y_max, -self.z_max], dtype=np.float32),
            high=np.array([self.x_max, self.y_max, self.z_max, self.x_max, self.y_max, self.z_max], dtype=np.float32),
            dtype=np.float32
        )
        self.steps = 0

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Reset the simulation and get the initial observation
        initial_obs = self.sim.reset(num_agents=1)
        
        # Randomly generate a goal position within the defined boundaries
        self.goal_position = np.array([
            np.random.uniform(0.2531, -0.1872),
            np.random.uniform(0.2201, -0.1711),
            np.random.uniform(0.2896, 0.1691)
        ])
        
        # Extract the pipette position from the initial observation
        robot_key = next(iter(initial_obs))
        pipette_position = np.array(initial_obs[robot_key]['pipette_position'], dtype=np.float32)
        
        # Concatenate the pipette position and goal position to form the observation
        observation = np.concatenate((pipette_position, self.goal_position)).astype(np.float32)
        self.steps = 0
        return observation, {}

    def step(self, action):
        # Extend the action with a zero value (for compatibility with the simulation)
        extended_action = np.append(action, [0])
        
        # Run the simulation with the given action
        sim_obs = self.sim.run([extended_action])
        
        # Extract the pipette position from the simulation observation
        robot_key = next(iter(sim_obs))
        robot_data = sim_obs[robot_key]
        pipette_position = np.array(robot_data['pipette_position'], dtype=np.float32)
        
        # Calculate the distance to the goal position
        distance = np.linalg.norm(np.array(pipette_position) - np.array(self.goal_position))
        
        # Calculate the reward (negative distance to the goal)
        reward = -distance
        
        # Check if the goal has been reached
        if np.linalg.norm(pipette_position - self.goal_position) <= 0.001:
            terminated = True
        else:
            terminated = False
        
        # Check if the maximum number of steps has been reached
        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False
        
        # Concatenate the pipette position and goal position to form the observation
        observation = np.concatenate((pipette_position, self.goal_position), axis=0).astype(np.float32)
        info = {}
        self.steps += 1
        return observation, reward, terminated, truncated, {}

    def render(self, mode='human'):
        if self.render:
            self.sim.render()

    def close(self):
        self.sim.close()
