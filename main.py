import numpy as np
from pettingzoo import AECEnv
from gymnasium import spaces

class CustomPettingZooEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "custom_petting_zoo_env"}

    def __init__(self, num_agents=2, grid_size=(5, 5), max_steps=100):
        super().__init__()
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0

        # Set up agent identifiers
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.possible_agents = self.agents[:]
        
        # Define action and observation spaces for agents
        self.action_spaces = {agent: spaces.Discrete(4) for agent in self.agents}  # e.g., 4 actions (up, down, left, right)
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(grid_size[0], grid_size[1]), dtype=np.float32)
            for agent in self.agents
        }

        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Initialize each agent's position randomly within the grid
        self.agent_positions = {agent: (np.random.randint(self.grid_size[0]), np.random.randint(self.grid_size[1])) 
                                for agent in self.agents}

        # Initial observation for each agent
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        return observations

    def _get_observation(self, agent):
        # Example observation: a grid with the agent's position marked
        obs = np.zeros(self.grid_size, dtype=np.float32)
        pos = self.agent_positions[agent]
        obs[pos] = 1
        return obs

    def step(self, action):
        agent = self.agent_selection

        # Update the agent's position based on the action
        self._move_agent(agent, action)
        
        # Example: Assign rewards, check if done, and update info
        self.rewards[agent] = 1  # Set reward based on specific conditions
        self.dones[agent] = self.current_step >= self.max_steps
        self.infos[agent] = {}  # Set any additional info here

        # Move to the next agent
        self.current_step += 1
        self.agent_selection = self._agent_selector.next()

    def _move_agent(self, agent, action):
        # Example movement logic (up, down, left, right)
        x, y = self.agent_positions[agent]
        if action == 0:  # up
            self.agent_positions[agent] = (max(0, x - 1), y)
        elif action == 1:  # down
            self.agent_positions[agent] = (min(self.grid_size[0] - 1, x + 1), y)
        elif action == 2:  # left
            self.agent_positions[agent] = (x, max(0, y - 1))
        elif action == 3:  # right
            self.agent_positions[agent] = (x, min(self.grid_size[1] - 1, y + 1))

    def render(self, mode="human"):
        grid = np.zeros(self.grid_size, dtype=str)
        grid[:] = "."
        for agent, pos in self.agent_positions.items():
            grid[pos] = agent[0]  # Mark agent positions on the grid
        print("\n".join(" ".join(row) for row in grid))

    def close(self):
        pass
