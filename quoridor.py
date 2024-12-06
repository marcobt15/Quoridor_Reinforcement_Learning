import functools
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Dict, Box

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from a_star import a_star, test_new_walls

import pygame

def env(**kwargs):
    env = Quoridor(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class Quoridor(AECEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {"name": "quoridor_aec_v6"}
    #Done
    def __init__(self, args=None):
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

        #render
        self.allow_player_control = True

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
        self.player_wall_placement = np.zeros((self.board_size-1, self.board_size-1, 2), dtype=object)


        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents} 
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}

        self.rewards = {agent: 0 for agent in self.agents}

        self.wall_action_mask = np.ones(self.num_wall_positions)

        self.player_1_action_mask = np.ones(4 + 4 + self.num_wall_positions, dtype='int8')
        self.player_1_action_mask[0] = 0
        self.player_1_action_mask[4] = 0

        self.player_2_action_mask = np.ones(4 + 4 + self.num_wall_positions, dtype='int8')
        self.player_2_action_mask[0] = 0
        self.player_2_action_mask[4] = 0

        self.infos = {"player_1" : {"position": self.player_positions["player_1"], "wall_locations" : self.wall_positions},
                      "player_2" : {"position": self.player_positions["player_2"], "wall_locations" : self.wall_positions}}
        
        self.observation = self.get_observations(self.agent_selection)
        
        return self.observation

    def get_observations(self, agent):
        """Generate observations for all agents."""

        opponent = "player_1" if agent == "player_2" else "player_2"

        if agent == "player_2":
            player_pos = (
                self.board_size - 1 - self.player_positions[agent][0],
                self.board_size - 1 - self.player_positions[agent][1],
            )  # Swap perspective
            opponent_pos = (
                self.board_size - 1 - self.player_positions[opponent][0],
                self.board_size - 1 - self.player_positions[opponent][1],
            )  # Swap perspective
            wall_positions = np.flip(self.wall_positions, axis=(0, 1))  # Flip walls
        else:
            player_pos = self.player_positions[agent]
            opponent_pos = self.player_positions[opponent]
            wall_positions = self.wall_positions

        observation = np.concatenate([
                    np.array(player_pos),  # Player's position
                    np.array(opponent_pos),  # Opponent's position
                    wall_positions.flatten(),  # Wall positions
                    [self.remaining_walls[agent]],  # Player's remaining walls
                    [self.remaining_walls[opponent]],  # Opponent's remaining walls
                    [int(self.player_jump[agent])],  # Player's jump availability
                    [int(self.player_jump[opponent])] # Oppoenent's jump availability
                ]).astype(np.int16)

        observations = {
            "observation" : observation,
            "action_mask" : self.get_action_mask(agent)
        }

        return observations

    def action_mask(self):
        return self.get_action_mask(self.agent_selection)

    def get_action_mask(self, agent):
        if agent == "player_1":
            return self.player_1_action_mask
        else:
            return self.player_2_action_mask

    #Doesn't return anything, just update states in the object so last() can use them
    def step(self, action):
        """Executes the selected action for the current agent."""
        """Assume that the action given is valid"""
        current_agent = self.agent_selection
        opponent = "player_1" if current_agent == "player_2" else "player_2"

        if (
            self.terminations[current_agent]
            or self.truncations[current_agent]
        ):
            self._was_dead_step(action)
            return
        
        original_action = action
        if current_agent == "player_2":
            if action == 0:
                action = 1
            elif action == 1:
                action = 0
            elif action == 2:
                action = 3
            elif action == 3:
                action = 2
            elif action == 4:
                action = 5
            elif action == 5:
                action = 4
            elif action == 6:
                action = 7
            elif action == 7:
                action = 6
            elif action < 72: #horizontal wall
                temp_action = action - 8 #removing the moving actions
                reversed_action = 63 - temp_action #reversing the action so it's the proper way
                action = reversed_action + 8 #add back the movement actions
            else:
                temp_action = action - 72 #removing moving and horizontal wall placements
                reversed_action = 63 - temp_action
                action = reversed_action + 72

        #call A* for current player
        pre_optimal_path, pre_cost = a_star(self.player_positions[current_agent], current_agent, self.wall_positions)

        #call A* for opponent player
        pre_opp_optimal_path, pre_opp_cost = a_star(self.player_positions[opponent], opponent, self.wall_positions)
            
        #after we make step, check if optimal move is the one agent made, then add rewards
            
        # print(f"current player position{self.player_positions[current_agent]}")
        # print(f"optimal path for:{current_agent} is {optimal_path}, cost is {cost}")
        # print(f"optimal move for:{current_agent} is {optimal_move}")

        if current_agent == "player_1":
            curr_action_mask = self.player_1_action_mask
        else:
            curr_action_mask = self.player_2_action_mask
            # print(curr_action_mask)

        if action != None and curr_action_mask[original_action] == 1:

            if action < 4: # Pawn jump
                self.player_jump[current_agent] = False

                #Removing players jump actions
                curr_action_mask[0:4] = np.zeros(4)

            if action < 8:  # Pawn movement: 0/4=up, 1/5=down, 2/6=left, 3/7=right
                self._move_pawn(current_agent, action)
            else:  # Wall placement
                wall_index = action - 8
                self._place_wall(current_agent, opponent, wall_index)
        else:
            print("no a valid move")
            print(action)
            print(curr_action_mask[action])
            print(curr_action_mask)

        # Check game end conditions
        if self.timestep >= 100:
            print('HAS TRUNCATED ON', current_agent)
            self.truncations = {"player_1" : True, "player_2" : True}

        terminations = {
            "player_1": self.player_positions["player_1"][0] == 8,
            "player_2": self.player_positions["player_2"][0] == 0
        }

        self.terminations = terminations

        post_optimal_path, post_cost = a_star(self.player_positions[current_agent], current_agent, self.wall_positions)

        post_opp_optimal_path, post_opp_cost = a_star(self.player_positions[opponent], opponent, self.wall_positions)

        if terminations[current_agent]:
            print(current_agent, " HAS WON")
            self.terminations = {agent: True for agent in self.agents}

            self.rewards[current_agent] = 1000
            self.rewards[opponent] = -1000
            
        elif self.truncations[current_agent]:

            if post_cost > post_opp_cost:
                self.rewards[current_agent] = -1000
            else:
                self.rewards[current_agent] = -500
            
            self.rewards[opponent] = -100

        else: #not terminated or truncated
            if pre_optimal_path or post_optimal_path == -1:
                curr_reward = -1000
            elif action < 4:
                curr_reward = 10*(pre_cost > post_cost) if  pre_cost > post_cost else -5
            #just not passing api test and i don't know what to do to fix it
            if action < 8 and action >= 4:
                #best path doesn't involve jumping so if they jump it should reduce the path cost by more than one getting higher reward
                curr_reward = 1 if pre_cost > post_cost else -1
            else:
                #the more they block their opponent the better the reward
                curr_reward = 10*(post_opp_cost-pre_opp_cost) if pre_opp_cost < post_opp_cost else -20 
            
            self.rewards[current_agent] = curr_reward
            self.rewards[opponent] = -curr_reward

        self._accumulate_rewards()

        self.agent_selection = self.agent_selector.next()

        # Update timestep
        self.timestep += 1

        self.infos = {"player_1" : {"position": self.player_positions["player_1"], "wall_locations" : self.wall_positions},
                      "player_2" : {"position": self.player_positions["player_2"], "wall_locations" : self.wall_positions}}

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
                
        # print(agent, "moved to", x, y)
    
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
            action_mask_update[0], action_mask_update[1] = action_mask_update[1], action_mask_update[0]
            action_mask_update[2], action_mask_update[3] = action_mask_update[3], action_mask_update[2]
            action_mask_update[4], action_mask_update[5] = action_mask_update[5], action_mask_update[4]
            action_mask_update[6], action_mask_update[7] = action_mask_update[7], action_mask_update[6]
            self.player_2_action_mask[0:8] = action_mask_update

    def _place_wall(self, agent, opponent, wall_index):
        """Places a wall at the given index if valid."""
        row, col, orientation = self._decode_wall_index(wall_index)
        self.wall_positions[row, col, orientation] = 1
        if agent == "player_1":
            player = 1
            self.player_wall_placement[row, col, orientation] = 1
        else:
            player = 2
            self.player_wall_placement[row, col, orientation] = 2
            
        placement_number = self.max_walls - self.remaining_walls[agent]
        self.player_wall_placement[row, col, orientation] = (player, placement_number)
        self.remaining_walls[agent] -= 1

        #update action mask for both agents
        # self.player_1_action_mask[8 + wall_index] = 0
        # self.player_2_action_mask[8 + wall_index] = 0
        self.wall_action_mask[wall_index] = 0

        ##CHECK IF WALLS NOW BLOCK A PLAYERS PATH TO THE END
        # test_new_walls(self.player_positions, self.wall_positions, self.player_1_action_mask, self.player_2_action_mask)
        test_new_walls(self.player_positions, self.wall_positions, self.wall_action_mask)

        #If a wall is placed horizontally across a wall cannot be place veritcally going through it and vice versa
        opposite_orientation_wall_index = wall_index + 64 if orientation == 0 else wall_index - 64
        self.wall_action_mask[opposite_orientation_wall_index] = 0

        # self.player_1_action_mask[8 + opposite_orientation_wall_index] = 0
        # self.player_2_action_mask[8 + opposite_orientation_wall_index] = 0

        #if a wall is placed at 5,5 for example then horizontal walls cannot be placed directly above or below
        #same kind of case for vertical walls
        if orientation == 0: #horizontal
            if col != 0:
                self.wall_action_mask[wall_index - 1] = 0
                # self.player_1_action_mask[8 + wall_index - 1] = 0
                # self.player_2_action_mask[8 + wall_index - 1] = 0
            if col != 7:
                self.wall_action_mask[wall_index + 1] = 0
                # self.player_1_action_mask[8 + wall_index + 1] = 0
                # self.player_2_action_mask[8 + wall_index + 1] = 0

        else: #vertical
            if row != 0:
                self.wall_action_mask[wall_index - 8] = 0
                # self.player_1_action_mask[8 + wall_index - 8] = 0
                # self.player_2_action_mask[8 + wall_index - 8] = 0
            if row != 7:
                self.wall_action_mask[wall_index + 8] = 0
                # self.player_1_action_mask[8 + wall_index + 8] = 0
                # self.player_2_action_mask[8 + wall_index + 8] = 0

        # print(agent, "placed a wall")
        # print("wall was placed at", row, col, orientation)
        # print(agent, "has this many walls left", self.remaining_walls[agent])

        #if player_1 has no walls left then remove the actions it can take
        if self.remaining_walls["player_1"] == 0:
            self.player_1_action_mask[8:] = np.zeros(128)
        #if they do have walls left then update action mask
        else:
            self.player_1_action_mask[8:] = self.wall_action_mask

        #if player_2 has walls left 
        if self.remaining_walls["player_2"] == 0:
            self.player_2_action_mask[8:] = np.zeros(128)
        #if they do have walls left then update action mask
        else:
            self.player_2_action_mask[8:72] = self.wall_action_mask[:64][::-1]
            self.player_2_action_mask[72:] = self.wall_action_mask[64:][::-1]    

        player_1_x, player_1_y = self.player_positions["player_1"]
        player_2_x, player_2_y = self.player_positions["player_2"]

        player_1_action_mask_update = self.player_1_action_mask[4:8]

        player_2_action_mask_update = self.player_2_action_mask[4:8]
        player_2_action_mask_update[0], player_2_action_mask_update[1] = player_2_action_mask_update[1], player_2_action_mask_update[0]
        player_2_action_mask_update[2], player_2_action_mask_update[3] = player_2_action_mask_update[3], player_2_action_mask_update[2]

        if orientation == 0: #horizontal
            #up
            if player_1_x - 1 == row and (player_1_y == col or player_1_y == col+1):
                player_1_action_mask_update[0] = 0

            #down
            elif player_1_x == row and (player_1_y == col or player_1_y == col+1):
                player_1_action_mask_update[1] = 0
            
            #repeat for player 2
            #up
            if player_2_x - 1 == row and (player_2_y == col or player_2_y == col+1):
                player_2_action_mask_update[0] = 0

            #down
            elif player_2_x == row and (player_2_y == col or player_2_y == col+1):
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

        #swapping player 2's action mask
        player_2_action_mask_update[0], player_2_action_mask_update[1] = player_2_action_mask_update[1], player_2_action_mask_update[0]
        player_2_action_mask_update[2], player_2_action_mask_update[3] = player_2_action_mask_update[3], player_2_action_mask_update[2]
        self.player_2_action_mask[4:8] = player_2_action_mask_update

        # print(self.player_1_action_mask)
        # print(self.player_2_action_mask)

    #Done
    def _decode_wall_index(self, index):
        """Decodes a wall index into row, col, and orientation."""
        orientation = 0 if index < 64 else 1
        index = index % 64
        row = index // (self.board_size - 1)
        col = index % (self.board_size - 1)
        return row, col, orientation

    def observe(self, agent):
        """Returns the observation for the specified agent."""
        return self.get_observations(agent)

    def close(self):
        """Closes the Pygame window."""
        if hasattr(self, "screen"):
            pygame.quit()

    def render(self):
        if not hasattr(self, "screen"):
            # Initialize pygame window only once
            pygame.init()
            self.window_size = 800
            self.cell_size = self.window_size // self.board_size
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Quoridor Render")

        # Colors
        black = (0, 0, 0)
        blue = (0, 0, 255)
        red = (255, 0, 0)
        white = (255, 255, 255)

        # Clear the screen
        self.screen.fill(white)  # White background

        # Draw the grid
        for x in range(self.board_size + 1):
            pygame.draw.line(
                self.screen, black, (0, x * self.cell_size), (self.window_size, x * self.cell_size), 1
            )  # Horizontal lines
            pygame.draw.line(
                self.screen, black, (x * self.cell_size, 0), (x * self.cell_size, self.window_size), 1
            )  # Vertical lines

        # Draw the players
        for agent, (x, y) in self.player_positions.items():
            center_x = y * self.cell_size + self.cell_size // 2
            center_y = x * self.cell_size + self.cell_size // 2
            color = blue if agent.lower() == "player_1" else red
            pygame.draw.circle(self.screen, color, (center_x, center_y), self.cell_size // 3)
            pygame.draw.circle(self.screen, black, (center_x, center_y), self.cell_size // 3, 2)  # Outline

            font = pygame.font.Font(None, self.cell_size // 2)
            text_color = black if self.player_jump.get(agent, False) else white  # Check player_jump status
            text = font.render("P1" if agent == "player_1" else "P2", True, text_color)
            text_rect = text.get_rect(center=(center_x, center_y))
            self.screen.blit(text, text_rect)

        # Draw the walls
        font = pygame.font.Font(None, self.cell_size // 3)
        for row in range(self.board_size - 1):
            for col in range(self.board_size - 1):
                for orientation in [0, 1]:  # Check both horizontal and vertical walls
                    if self.player_wall_placement[row, col, orientation] != 0:
                        player, placement_number = self.player_wall_placement[row, col, orientation]
                        color = blue if player == 1 else red

                        if orientation == 0:  # Horizontal wall
                            start_pos = (col * self.cell_size, row * self.cell_size + self.cell_size)
                            end_pos = (start_pos[0] + 2 * self.cell_size, start_pos[1])
                        else:  # Vertical wall
                            start_pos = (col * self.cell_size + self.cell_size, row * self.cell_size)
                            end_pos = (start_pos[0], start_pos[1] + 2 * self.cell_size)

                        pygame.draw.line(self.screen, color, start_pos, end_pos, 5)

                        # Render placement number
                        if orientation == 0:
                            mid_x = (start_pos[0] + end_pos[0]) // 2
                            mid_y = start_pos[1]
                            text_rect = font.render(str(placement_number), True, black).get_rect(
                                center=(mid_x, mid_y - 10)
                            )
                        else:
                            mid_x = start_pos[0]
                            mid_y = (start_pos[1] + end_pos[1]) // 2
                            text_rect = font.render(str(placement_number), True, black).get_rect(
                                center=(mid_x + 10, mid_y)
                            )
                        self.screen.blit(font.render(str(placement_number), True, black), text_rect)

        pygame.time.wait(50)
        pygame.display.flip()

    #def wait_for_user_action(self):
       

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        observation = Dict(
            {
                "observation": MultiDiscrete([self.board_size, self.board_size, self.board_size, self.board_size] + [2] * self.num_wall_positions + [self.max_walls+1, self.max_walls+1, 2, 2], dtype=np.int16),
                "action_mask": Box(low=0, high=1, shape=(136,), dtype=np.int8),
            }
        )
        return observation


    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4 + 4 + self.num_wall_positions)