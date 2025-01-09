import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=self.render)

        # Define the min/max ranges for the goal position
        self.x_min, self.x_max = -0.187,  0.2531
        self.y_min, self.y_max = -0.1705, 0.2195
        self.z_min, self.z_max =  0.1195, 0.2895

        # Define action space: 3 continuous actions (x, y, z)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Define observation space: 6D [pipette_x, pipette_y, pipette_z, goal_x, goal_y, goal_z]
        self.observation_space = spaces.Box(
            low=np.array([ self.x_min,  self.y_min,  self.z_min, -self.x_max, -self.y_max, -self.z_max], dtype=np.float32),
            high=np.array([self.x_max, self.y_max, self.z_max,  self.x_max,  self.y_max,  self.z_max], dtype=np.float32),
            dtype=np.float32
        )

        # Track the current step and the previous distance to the goal
        self.steps = 0
        self.prev_distance = None

    def reset(self, seed=None):
        # If provided, set the random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Randomize the goal position in the given min/max range
        x = np.random.uniform(self.x_min, self.x_max)
        y = np.random.uniform(self.y_min, self.y_max)
        z = np.random.uniform(self.z_min, self.z_max)
        self.goal_position = np.array([x, y, z], dtype=np.float32)

        # Reset the simulation
        self.sim.reset(num_agents=1)
        pipette_position = self.sim.get_pipette_position(self.sim.robotIds[0])

        # Combine pipette and goal positions into one observation vector
        observation = np.concatenate(
            (pipette_position, self.goal_position),
            axis=0
        ).astype(np.float32)

        # Reset counters
        self.steps = 0
        self.prev_distance = np.linalg.norm(pipette_position - self.goal_position)

        info = {}
        return observation, info

    def step(self, action):
        """Take one action step in the environment.

        Args:
            action (np.ndarray): A 3D action specifying movement in (x, y, z).

        Returns:
            observation (np.ndarray): The next observation [pipette_x, pipette_y, pipette_z, goal_x, goal_y, goal_z].
            reward (float): The reward signal from this step.
            terminated (bool): Whether the episode ended successfully.
            truncated (bool): Whether the episode ended due to time steps hitting max_steps.
            info (dict): Additional debugging info (empty here).
        """
        # Convert action to float32 and append a 0 if needed by your sim
        action = np.append(np.array(action, dtype=np.float32), 0.0)
        # Execute the action in the simulation
        self.sim.run([action])
        pipette_position = self.sim.get_pipette_position(self.sim.robotIds[0])

        # Compute distance to the goal
        distance = np.linalg.norm(pipette_position - self.goal_position)

        # 1) Distance-based shaping: difference from last step
        distance_diff = self.prev_distance - distance
        reward = distance_diff * 10.0  # Scale factor (tune as needed)

        # 2) Small time/step penalty to encourage quicker reaching
        reward -= 0.001

        # 3) Success bonus if within threshold
        if distance <= 0.001:
            reward += 5.0
            terminated = True
        else:
            terminated = False

        # 4) Check if we hit the max steps
        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False

        # Update the observation
        observation = np.concatenate(
            (pipette_position, self.goal_position),
            axis=0
        ).astype(np.float32)

        # Update the stored variables
        self.steps += 1
        self.prev_distance = distance

        info = {}
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        # If you have custom rendering logic, implement it here
        pass

    def close(self):
        self.sim.close()
