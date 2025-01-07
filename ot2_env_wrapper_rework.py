import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

class OT2CustomEnv(gym.Env):
    def __init__(self, render_mode=False, max_episode_steps=1000):
        super(OT2CustomEnv, self).__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        # Initialize simulation environment
        self.simulator = Simulation(num_agents=1, render=self.render_mode)

        # Define workspace boundaries
        self.bounds_x = (-0.187, 0.2531)
        self.bounds_y = (-0.1705, 0.2195)
        self.bounds_z = (0.1195, 0.2895)

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([self.bounds_x[0], self.bounds_y[0], self.bounds_z[0], 
                          -self.bounds_x[1], -self.bounds_y[1], -self.bounds_z[1]], dtype=np.float32),
            high=np.array([self.bounds_x[1], self.bounds_y[1], self.bounds_z[1], 
                           self.bounds_x[1], self.bounds_y[1], self.bounds_z[1]], dtype=np.float32),
            dtype=np.float32
        )

        self.current_steps = 0

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Randomize goal position within defined boundaries
        goal_x = np.random.uniform(*self.bounds_x)
        goal_y = np.random.uniform(*self.bounds_y)
        goal_z = np.random.uniform(*self.bounds_z)
        self.target_position = np.array([goal_x, goal_y, goal_z])

        # Reset the simulator and get initial observation
        simulator_observation = self.simulator.reset(num_agents=1)
        pipette_position = self.simulator.get_pipette_position(self.simulator.robotIds[0])

        # Concatenate pipette position and goal position for the observation
        observation = np.concatenate((pipette_position, self.target_position), axis=0).astype(np.float32)

        self.current_steps = 0
        self.previous_distance = None
        self.starting_distance = None

        return observation, {}

    def step(self, action):
        # Extend action with an additional zero for simulation requirements
        extended_action = np.append(np.array(action, dtype=np.float32), 0)
        simulator_observation = self.simulator.run([extended_action])
        pipette_position = self.simulator.get_pipette_position(self.simulator.robotIds[0])

        # Compute distance to the goal
        current_distance = np.linalg.norm(pipette_position - self.target_position)

        # Initialize distances if they haven't been set
        if self.previous_distance is None:
            self.previous_distance = current_distance
            self.starting_distance = current_distance

        # Reward for moving closer to the goal
        progress_reward = (self.previous_distance - current_distance) * 10
        self.previous_distance = current_distance

        # Additional milestone rewards
        milestone_rewards = [0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999]
        milestone_bonus = 0
        for milestone in milestone_rewards[:]:
            if (self.previous_distance > milestone * self.starting_distance and
                current_distance <= milestone * self.starting_distance):
                milestone_bonus += 20 * milestone
                milestone_rewards.remove(milestone)

        # Overall reward
        reward = progress_reward + milestone_bonus - 0.01

        # Check for goal achievement
        goal_threshold = 0.001
        if current_distance <= goal_threshold:
            reward += 100
            terminated = True
        else:
            terminated = False

        # Check if the episode should end due to max steps
        truncated = self.current_steps >= self.max_episode_steps

        # Create new observation
        observation = np.concatenate((pipette_position, self.target_position), axis=0).astype(np.float32)

        self.current_steps += 1
        return observation, reward, terminated, truncated, {}

    def render(self, mode='human'):
        pass

    def close(self):
        self.simulator.close()
