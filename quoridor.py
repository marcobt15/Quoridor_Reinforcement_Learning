import functools
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from a_star import a_star
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pygame

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
        self.truncations = {agent: False for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.reset()

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

        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents} 
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}

        self.rewards = {agent: 0 for agent in self.agents}

        self.player_1_action_mask = np.ones(4 + 4 + self.num_wall_positions, dtype='int8')
        self.player_1_action_mask[0] = 0
        self.player_1_action_mask[4] = 0

        self.player_2_action_mask = np.ones(4 + 4 + self.num_wall_positions, dtype='int8')
        self.player_2_action_mask[1] = 0
        self.player_2_action_mask[5] = 0

        self.infos = {"player_1" : {"action_mask" : self.player_1_action_mask},
                      "player_2" : {"action_mask" : self.player_2_action_mask}
                    }
        self.observation = self.get_observations(self.agent_selection)
        
        return self.observation

    def get_observations(self, agent):
        """Generate observations for all agents."""

        opponent = "player_1" if agent == "player_2" else "player_2"
        # observations = { "observation" : 
        #     np.concatenate([
        #             np.array(self.player_positions[agent]),  # Player's position
        #             np.array(self.player_positions[opponent]),  # Opponent's position
        #             self.wall_positions.flatten(),  # Wall positions
        #             [self.remaining_walls[agent]],  # Player's remaining walls
        #             [self.remaining_walls[opponent]],  # Opponent's remaining walls
        #             [int(self.player_jump[agent])],  # Player's jump availability
        #             [int(self.player_jump[opponent])]  # Opponent's jump availability
        #         ]),
        #     # "action_mask": self.get_action_mask(agent)
        # }

        observation = np.concatenate([
                    np.array(self.player_positions[agent]),  # Player's position
                    np.array(self.player_positions[opponent]),  # Opponent's position
                    self.wall_positions.flatten(),  # Wall positions
                    [self.remaining_walls[agent]],  # Player's remaining walls
                    [self.remaining_walls[opponent]],  # Opponent's remaining walls
                    [int(self.player_jump[agent])],  # Player's jump availability
                    [int(self.player_jump[opponent])]  # Opponent's jump availability
                ])

        return observation

        # observations = {}
        # for agent in self.agents:
        #     opponent = "player_1" if agent == "player_2" else "player_2"
        #     observations[agent] = {
        #         "observation": np.concatenate([
        #             np.array(self.player_positions[agent]),  # Player's position
        #             np.array(self.player_positions[opponent]),  # Opponent's position
        #             self.wall_positions.flatten(),  # Wall positions
        #             [self.remaining_walls[agent]],  # Player's remaining walls
        #             [self.remaining_walls[opponent]],  # Opponent's remaining walls
        #             [int(self.player_jump[agent])],  # Player's jump availability
        #             [int(self.player_jump[opponent])]  # Opponent's jump availability
        #         ]),
        #         "action_mask": self.get_action_mask(agent)
        #     }
        # return observations

    def get_action_mask(self, agent):
        if agent == "player_1":
            # print(self.player_1_action_mask)
            return self.player_1_action_mask
        else:
            # print(self.player_2_action_mask)
            return self.player_2_action_mask

    #Doesn't return anything, just update states in the object so last() can use them
    def step(self, action):
        """Executes the selected action for the current agent."""
        """Assume that the action given is valid"""
        current_agent = self.agent_selection
        
        #call A* for current player
        optimal_path, cost = a_star(self.player_positions[current_agent], current_agent, self.wall_positions)
        if cost == -1:
            print("no valid path found") #need to use this check for valid moves
        else:
            optimal_move = optimal_path[1]
            
        #after we make step, check if optimal move is the one agent made, then add rewards
            
        # print(f"current player position{self.player_positions[current_agent]}")
        # print(f"optimal path for:{current_agent} is {optimal_path}, cost is {cost}")
        # print(f"optimal move for:{current_agent} is {optimal_move}")


        if (
            self.terminations[current_agent]
            or self.truncations[current_agent]
        ):
            self._was_dead_step(action)
            return

        if current_agent == "player_1":
            curr_action_mask = self.player_1_action_mask
        else:
            curr_action_mask = self.player_2_action_mask

        if action and curr_action_mask[action] == 1:

            if action < 4: # Pawn jump
                self.player_jump[current_agent] = False

                #Removing players jump actions
                curr_action_mask[0:4] = np.zeros(4)

            if action < 8:  # Pawn movement: 0/4=up, 1/5=down, 2/6=left, 3/7=right
                self._move_pawn(current_agent, action)
            else:  # Wall placement
                wall_index = action - 8
                self._place_wall(current_agent, wall_index)

        # Check game end conditions
        if self.timestep >= 100:
            print('HAS TRUNCATED ON', current_agent)
            
            self.truncations = {"player_1" : True, "player_2" : True}
        
        terminations = self._check_termination(current_agent)
        self.rewards = {}
        if terminations[current_agent]:
            # Reward the winning agent
            self.rewards[current_agent] = 1
            self._cumulative_rewards[agent] += 1
            # Penalize others
            for other_agent in self.agents:
                if other_agent != current_agent:
                    self.rewards[other_agent] = -1
                    self._cumulative_rewards[other_agent] += -1
            
        else:
            for agent in self.agents:
                self.rewards[agent] = 0
                self._cumulative_rewards[agent] += 0

        self.agent_selection = self.agent_selector.next()

        # Update timestep
        self.timestep += 1
        
        self.infos = {"player_1" : {"action_mask" : self.player_1_action_mask},
                      "player_2" : {"action_mask" : self.player_2_action_mask}}

    #Done
    def _move_pawn(self, agent, direction):
        """Handles pawn movement based on the given direction."""

        direction = direction % 4
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
        action_mask_update = np.ones(8)
        x, y = self.player_positions[agent]
        if self.player_jump[agent] == False:
                action_mask_update[0:4] = np.zeros(4)
                
        print(agent, "moved to", x, y)
    
        #cant jump or move up - edge of board
        if x == 0:
            action_mask_update[0] = 0
            action_mask_update[4] = 0
        
        #check wall up -> cant move up
        elif (y > 0 and self.wall_positions[x-1][y-1][0] == 1) or (y < 8 and self.wall_positions[x-1][y][0]== 1) :
            action_mask_update[4] = 0
        
        #cant jump or move down - edge of board
        if x == 8:
            action_mask_update[1] = 0
            action_mask_update[5] = 0
            
        #check wall down -> cant move down
        elif (y > 0 and self.wall_positions[x][y-1][0] == 1) or (y < 8 and self.wall_positions[x][y][0] == 1):
            action_mask_update[5] = 0
        
        #cant jump or move left - edge of board
        if y == 0:
            action_mask_update[2] = 0
            action_mask_update[6] = 0
        
        #check wall left -> cant move left
        elif (x > 0 and self.wall_positions[x-1][y-1][1] == 1) or (x < 8 and self.wall_positions[x][y-1][1] == 1):
            action_mask_update[6] = 0
        
        #cant jump or move right - edge of board
        if y == 8:
            
            action_mask_update[3] = 0
            action_mask_update[7] = 0
            
        #check wall right -> cant move right
        elif (x > 0 and self.wall_positions[x-1][y][1] == 1) or (x < 8 and self.wall_positions[x][y][1] == 1):
            action_mask_update[7] = 0

        if agent == "player_1":
            self.player_1_action_mask[0:8] = action_mask_update
        else:
            self.player_2_action_mask[0:8] = action_mask_update

    def _place_wall(self, agent, wall_index):
        """Places a wall at the given index if valid."""
        row, col, orientation = self._decode_wall_index(wall_index)
        self.wall_positions[row, col, orientation] = 1
        self.remaining_walls[agent] -= 1

        ##CHECK IF WALLS NOW BLOCK A PLAYERS PATH TO THE END

        #If a wall is placed horizontally across a wall cannot be place veritcally going through it and vice versa
        opposite_orientation_wall_index = wall_index + 64 if orientation == 0 else wall_index - 64
        self.player_1_action_mask[8 + opposite_orientation_wall_index] = 0
        self.player_2_action_mask[8 + opposite_orientation_wall_index] = 0

        #if a wall is placed at 5,5 for example then horizontal walls cannot be placed directly above or below
        #same kind of case for vertical walls
        if orientation == 0: #horizontal
            if col != 0:
                self.player_1_action_mask[wall_index-1] = 0
                self.player_2_action_mask[wall_index-1] = 0
            if col != 7:
                self.player_1_action_mask[wall_index+1] = 0
                self.player_2_action_mask[wall_index+1] = 0

        else: #vertical
            if row != 0:
                self.player_1_action_mask[wall_index-8] = 0
                self.player_2_action_mask[wall_index-8] = 0
            if row != 7:
                self.player_1_action_mask[wall_index+8] = 0
                self.player_2_action_mask[wall_index+8] = 0

        print(agent, "placed a wall")
        print(agent, "has this many walls left", self.remaining_walls[agent])
        print("wall was placed at", row, col, orientation)

        if self.remaining_walls[agent] == 0:
            if agent == "player_1":
                self.player_1_action_mask[8:] = np.zeros(128)
            else:
                self.player_2_action_mask[8:] = np.zeros(128)

        #update action mask for both agents
        self.player_1_action_mask[8 + wall_index] = 0
        self.player_2_action_mask[8 + wall_index] = 0

        player_1_x, player_1_y = self.player_positions["player_1"]
        player_2_x, player_2_y = self.player_positions["player_2"]

        player_1_action_mask_update = self.player_1_action_mask[4:8]
        player_2_action_mask_update = self.player_2_action_mask[4:8]

        if orientation == 0: #horizontal
            #up
            if player_1_x - 1 == row and (player_1_y == col or player_1_y == col-1):
                player_1_action_mask_update[0] = 0

            #down
            elif player_1_x == row and (player_1_y == col or player_1_y == col-1):
                player_1_action_mask_update[1] = 0
            
            #repeat for player 2
            #up
            if player_2_x - 1 == row and (player_2_y == col or player_2_y == col-1):
                player_2_action_mask_update[0] = 0

            #down
            elif player_2_x == row and (player_2_y == col or player_2_y == col-1):
                player_2_action_mask_update[1] = 0

        else: #vertical
            #left
            if player_1_y - 1 == col and (player_1_x == row or player_1_x == row+1):
                player_1_action_mask_update[2] = 0

            #right
            if player_1_y == col and (player_1_x == row or player_1_x == row+1):
                player_1_action_mask_update[3] = 0

            #repeat for player 2
            #left
            if player_2_y - 1 == col and (player_2_x == row or player_2_x == row+1):
                player_2_action_mask_update[2] = 0

            #right
            if player_2_y == col and (player_2_x == row or player_2_x == row+1):
                player_2_action_mask_update[3] = 0

        self.player_1_action_mask[4:8] = player_1_action_mask_update
        self.player_2_action_mask[4:8] = player_2_action_mask_update

    #Done
    def _decode_wall_index(self, index):
        """Decodes a wall index into row, col, and orientation."""
        orientation = 0 if index < 64 else 1
        index = index % 64
        row = index // (self.board_size - 1)
        col = index % (self.board_size - 1)
        return row, col, orientation
    
    #Done
    def _check_termination(self, agent):

        # if agent == "player_1":
        #     if self.player_positions[agent][0] == 8:
        #         return True

        """Checks if the game has ended."""
        terminations = {
            "player_1": self.player_positions["player_1"][0] == 8,
            "player_2": self.player_positions["player_2"][0] == 0
        }

        # print(terminations)

        #apparently needs to remove the player?
        # for agent, terminated in terminations.items():
        #     if terminated and agent in self.agents:
        #         self.agents.remove(agent)
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
        return self.get_observations(agent)

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
        """Renders the board using pygame."""
        # Initialize pygame window
        pygame.init()
        window_size = 800
        cell_size = window_size // self.board_size
        screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Board Render")
        screen.fill((255, 255, 255))  # White background

        # Colors
        black = (0, 0, 0)
        blue = (0, 0, 255)
        red = (255, 0, 0)
        brown = (139, 69, 19)
        white = (255, 255, 255)

        # Draw the grid
        for x in range(self.board_size + 1):
            pygame.draw.line(screen, black, (0, x * cell_size), (window_size, x * cell_size), 1)  # Horizontal lines
            pygame.draw.line(screen, black, (x * cell_size, 0), (x * cell_size, window_size), 1)  # Vertical lines

        # Draw the players
        for agent, (x, y) in self.player_positions.items():
            center_x = y * cell_size + cell_size // 2
            center_y = x * cell_size + cell_size // 2
            color = blue if agent.lower() == "player_1" else red
            pygame.draw.circle(screen, color, (center_x, center_y), cell_size // 3)
            pygame.draw.circle(screen, black, (center_x, center_y), cell_size // 3, 2)  # Outline
            font = pygame.font.Font(None, cell_size // 2)
            text = font.render("P1" if agent == 'player_1' else 'P2', True, white)
            text_rect = text.get_rect(center=(center_x, center_y))
            screen.blit(text, text_rect)

        # Draw the walls
        for row in range(self.board_size - 1):
            for col in range(self.board_size - 1):
                if self.wall_positions[row, col, 0] == 1:  # Horizontal wall
                    start_pos = (col * cell_size, row * cell_size + cell_size)
                    end_pos = (start_pos[0] + 2 * cell_size, start_pos[1])
                    pygame.draw.line(screen, brown, start_pos, end_pos, 5)
                if self.wall_positions[row, col, 1] == 1:  # Vertical wall
                    start_pos = (col * cell_size + cell_size, row * cell_size)
                    end_pos = (start_pos[0], start_pos[1] + 2 * cell_size)
                    pygame.draw.line(screen, brown, start_pos, end_pos, 5)

        # Update display
        pygame.display.flip()

        # Wait for a short time to visualize the render
        pygame.time.wait(5000)
        pygame.quit()




    #If render is defined then close has to be defined
    #render doesn't open any windows (so far) so it doesn't need to do anything
    def close(self):
        pass


    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return MultiDiscrete([self.board_size, self.board_size, self.board_size, self.board_size] + [2] * self.num_wall_positions + [self.max_walls+1, self.max_walls+1, 2, 2])


    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4 + 4 + self.num_wall_positions)