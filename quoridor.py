import functools
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

class Quoridor(AECEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {"render.modes": ["human"], "name": "quoridor_aec_v0"}
    #Done
    def __init__(self):
        """Initialize the AEC Quoridor environment."""
        self.board_size = 9
        self.max_walls = 10  # Number of walls each player can place
        self.num_wall_positions = 128 #64 horionzal walls and 64 vertical walls

        # Agents
        self.possible_agents = ["player_1", "player_2"]
        self.agents = copy(self.possible_agents)
        self.agent_selector = agent_selector(self.possible_agents)
        self.agent_selection = self.agent_selector.next() 

        # Game state
        self.timestep = 0
        self.player_positions = {"player_1": (0, 4), "player_2": (8, 4)}
        self.player_jump = {"player_1": True, "player_2": True}
        self.remaining_walls = {"player_1": self.max_walls, "player_2": self.max_walls}
        self.wall_positions = np.zeros((self.board_size-1, self.board_size-1, 2)) #horizontal then vertical

        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}  # Add terminations


    #Done
    def reset(self, seed=None, options=None):
        """Resets the environment to the starting state."""
        self.agents = copy(self.possible_agents)
        self.agent_selector = agent_selector(self.possible_agents)
        self.agent_selection = self.agent_selector.next() 

        self.timestep = 0
        self.player_positions = {"player_1": (0, 4), "player_2": (8, 4)}
        self.player_jump = {"player_1": True, "player_2": True}
        self.remaining_walls = {"player_1": self.max_walls, "player_2": self.max_walls}
        self.wall_positions = np.zeros((self.board_size-1, self.board_size-1, 2))

        self.terminations = {agent: False for agent in self.possible_agents}  # Add terminations
        self.truncations = {agent: False for agent in self.possible_agents}  # Add terminations


        player_1_action_mask = np.ones(4 + 4 + self.num_wall_positions)
        player_1_action_mask[0] = 0
        player_1_action_mask[4] = 0

        player_2_action_mask = np.ones(4 + 4 + self.num_wall_positions)
        player_2_action_mask[1] = 0
        player_2_action_mask[5] = 0

        self.observation = {
            "player_1" : {
                "observation": np.concatenate([np.array(self.player_positions["player_1"]), self.wall_positions.flatten()]),
                "action_mask": player_1_action_mask
            },

            "player_2" : {
                "observation": np.concatenate([np.array(self.player_positions["player_2"]), self.wall_positions.flatten()]),
                "action_mask": player_2_action_mask
            }
        }
        
        self.rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        return self.observation

    def step(self, action):
        """Executes the selected action for the current agent."""
        """Assume that the action given is valid"""
        current_agent = self.agent_selection

        if action < 4: # Pawn jump
            self.player_jump[current_agent] = False

            #Removing players jump actions
            self.observation[current_agent]["action_mask"][0:4] = np.zeros(4)

        if action < 8:  # Pawn movement: 0/4=up, 1/5=down, 2/6=left, 3/7=right
            self._move_pawn(current_agent, action)
        else:  # Wall placement
            wall_index = action - 8
            self._place_wall(current_agent, wall_index)

        # Check game end conditions
        terminations = self._check_termination()
        truncations = {agent: self.timestep > 100 for agent in self.agents}
        active_agents = [agent for agent in self.agents if not (terminations[agent] or truncations[agent])]
        
        # If game ends, clear agents
        if not active_agents:
            self.agents = []

        self.rewards = {agent: 0 for agent in active_agents}
        if terminations:
            for agent, terminated in terminations.items():
                if terminated:
                    self.rewards[agent] = 1
                    for other_agent in self.agents:
                        if other_agent != agent:
                            self.rewards[other_agent] = -1
            # rewards = self._calculate_rewards()
            # self.agents = []
        else:
            # Rotate to the next agent
            self.agent_selection = self.agent_order.next()

            #UPDATE REWARDS HERE FOR NON TERMINATION REWARDS
            self.rewards = {agent: 0 for agent in self.possible_agents}

        # Update timestep
        self.timestep += 1

        # Generate observations and return
        observations = self.observation
        self.terminations = terminations
        self.truncations = truncations
        infos = self.infos
        return observations, self.rewards, terminations, truncations, infos

    #Done
    def _move_pawn(self, agent, direction):
        """Handles pawn movement based on the given direction."""
        x, y = self.player_positions[agent]
        if direction == 0:      # Up
            self.player_positions[agent] = (x - 1, y)
        elif direction == 1:    # Down
            self.player_positions[agent] = (x + 1, y)
        elif direction == 2:    # Left
            self.player_positions[agent] = (x, y - 1)
        elif direction == 3:    # Right
            self.player_positions[agent] = (x, y + 1)
        
        #Update action mask after move
        #up, down, left, right
        action_mask_update = np.ones(4)
        x, y = self.player_positions[agent]
        if x == 0 or (y > 0 and self.wall_positions[x-1][y-1][0] == 1) or (y < 8 and self.wall_positions[x-1][y][0]) == 1:
            action_mask_update[0] = 0
        
        if x == 8 or (y > 0 and self.wall_positions[x][y-1][0] == 1) or (y < 8 and self.wall_positions[x][y][0] == 1):
            action_mask_update[1] = 0
        
        if y == 0 or (x > 0 and self.wall_positions[x-1][y-1][1] == 1) or (x < 8 and self.wall_positions[x][y-1][1] == 1):
            action_mask_update[2] = 0
        
        if y == 8 or (x > 0 and self.wall_positions[x-1][y][1] == 1) or (x < 8 and self.wall_positions[x][y][1] == 1):
            action_mask_update[3] = 0

        self.observation[agent]["action_mask"][4:8] = action_mask_update

    #Done
    def _place_wall(self, agent, wall_index):
        """Places a wall at the given index if valid."""
        row, col, orientation = self._decode_wall_index(wall_index)
        self.wall_positions[row, col, orientation] = 1
        self.remaining_walls[agent] -= 1

        #update action mask for both agents
        action_mask_player_1 = self.observation[self.possible_agents[0]]["action_mask"]
        action_mask_player_2 = self.observation[self.possible_agents[1]]["action_mask"]

        action_mask_player_1[8 + wall_index] = 0
        action_mask_player_2[8 + wall_index] = 0

        #If a wall is placed horizontally across a wall cannot be place veritcally going through it and vice versa
        opposite_orientation_wall_index = wall_index + 64 if orientation == 0 else wall_index - 64
        action_mask_player_1[8 + opposite_orientation_wall_index] = 0
        action_mask_player_2[8 + opposite_orientation_wall_index] = 0

        player_1_x, player_1_y = self.player_positions["player_1"]
        player_2_x, player_2_y = self.player_positions["player_2"]

        player_1_action_mask_update = np.ones(4)
        player_2_action_mask_update = np.ones(4)

        if orientation == 0: #horizontal
            #up
            if player_1_x + 1 == row and (player_1_y == col or player_1_y == col+1):
                player_1_action_mask_update[0] = 0

            #down
            elif player_1_x == row and (player_1_y == col or player_1_y == col+1):
                player_1_action_mask_update[1] = 0
            
            #repeat for player 2
            #up
            if player_2_x + 1 == row and (player_2_y == col or player_2_y == col+1):
                player_2_action_mask_update[0] = 0

            #down
            elif player_2_x == row and (player_2_y == col or player_2_y == col+1):
                player_2_action_mask_update[1] = 0

        else: #vertical
            #left
            if player_1_y -1 == col and (player_1_x == row or player_1_x == row+1):
                player_1_action_mask_update[2] = 0

            #right
            if player_1_y == col and (player_1_x == row or player_1_x == row+1):
                player_1_action_mask_update[3] = 0

            #repeat for player 2
            #left
            if player_2_y -1 == col and (player_2_x == row or player_2_x == row+1):
                player_2_action_mask_update[2] = 0

            #right
            if player_2_y == col and (player_2_x == row or player_2_x == row+1):
                player_2_action_mask_update[3] = 0

        self.observation[self.possible_agents[0]]["action_mask"][4:8] = player_1_action_mask_update
        self.observation[self.possible_agents[1]]["action_mask"][4:8] = player_2_action_mask_update

    #Done
    def _decode_wall_index(self, index):
        """Decodes a wall index into row, col, and orientation."""
        orientation = 0 if index < 64 else 1
        index = index % 64
        row = index // (self.board_size - 1)
        col = index % (self.board_size - 1)
        return row, col, orientation

    #not used right now
    def _get_observation(self):
        """Generates the observation for all agents."""
        observations = {}
        for agent in self.agents:
            x, y = self.player_positions[agent]
            walls = self.wall_positions.flatten()
            observations[agent] = {
                "observation": np.concatenate([[x, y], walls]),
                "action_mask": self._get_action_mask(agent),
            }
        return observations
    
    #not used right now
    def _get_action_mask(self, agent):
        """Generates an action mask for the given agent."""
        action_mask = np.ones(4 + 4 + self.num_wall_positions, dtype=np.int8)
        x, y = self.player_positions[agent]

        # prevent jumping twice
        if not self.player_jump[agent]:
            action_mask[0:4] = 0
        # Prevent movement off the board
        if x == 0:
            action_mask[4] = 0  # No up
        if x == self.board_size - 1:
            action_mask[5] = 0  # No down
        if y == 0:
            action_mask[6] = 0  # No left
        if y == self.board_size - 1:
            action_mask[7] = 0  # No right
        return action_mask
    
    #Done
    def _check_termination(self):
        """Checks if the game has ended."""
        terminations = {
            "player_1": self.player_positions["player_1"][0] == 8,
            "player_2": self.player_positions["player_2"][0] == 0
        }

        #apparently needs to remove the player?
        for agent, terminated in terminations.items():
            if terminated and agent in self.agents:
                self.agents.remove(agent)
        return terminations


    #TODO
    def _calculate_rewards(self):
        """Calculates the rewards for the agents."""
        return {
            agent: 0 #TODO
            for agent in self.agents
        }

    def observe(self, agent):
        """Returns the observation for the specified agent."""
        return self.observation[agent]["observation"]

    def last(self, observe=True):
        """Returns the observation, reward, termination, truncation, and info for the current agent."""
        agent = self.agent_selection
        observation = self.observe(agent) if observe and agent in self.agents else None
        reward = self.rewards.get(agent, 0)
        termination = self.terminations.get(agent, False)
        truncation = self.truncations.get(agent, False)
        info = self.infos.get(agent, {})
        return observation, reward, termination, truncation, info


    def render(self):
        """Renders the board with players, walls, and spaces."""
        # Define a filler character for empty cells
        filler = "O"
        horizontal_wall = "--"
        vertical_wall = "|"
        empty_space = "  "  # Empty space for spacing

        # Create an empty grid that accommodates spaces and walls
        render_grid = [[" " for _ in range(self.board_size * 2 - 1)] for _ in range(self.board_size * 2 - 1)]

        # Place the players
        for agent, (x, y) in self.player_positions.items():
            render_grid[x * 2][y * 2] = agent[0].upper()

        # Add horizontal and vertical walls
        for row in range(self.board_size - 1):
            for col in range(self.board_size - 1):
                if self.wall_positions[row, col, 0] == 1:  # Horizontal wall
                    render_grid[row * 2 + 1][col * 2] = horizontal_wall
                    render_grid[row * 2 + 1][col * 2+1] = horizontal_wall
                if self.wall_positions[row, col, 1] == 1:  # Vertical wall
                    render_grid[row * 2][col * 2 + 1] = vertical_wall
                    render_grid[row * 2+2][col * 2 + 1] = vertical_wall


        # Fill remaining spaces with filler
        for row in range(0, len(render_grid), 2):
            for col in range(0, len(render_grid[row]), 2):
                if render_grid[row][col] == " ":
                    render_grid[row][col] = filler

        # Print the grid row by row
        for row in render_grid:
            print("".join(row))


    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return MultiDiscrete([self.board_size - 1, self.board_size - 1] + [2] * self.num_wall_positions + [2])


    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4 + 4 + self.num_wall_positions)