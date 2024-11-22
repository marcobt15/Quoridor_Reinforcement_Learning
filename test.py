from quoridor import Quoridor
from pettingzoo.test import api_test
from a_star import test_new_walls
import numpy as np

def test_render():
    # Initialize the Quoridor environment
    env = Quoridor()

    # Test case 1: Initial board state
    print("=== Test Case 1: Initial Board ===")
    env.reset()
    env.render()
    print("\n")

    # Test case 2: Players in new positions
    print("=== Test Case 2: Players Moved ===")
    env.reset()
    env.player_positions["player_1"] = (1, 4)
    env.player_positions["player_2"] = (7, 4)
    env.render()
    print("\n")

    # Test case 3: Horizontal wall placed
    print("=== Test Case 3: Horizontal Wall Placed ===")
    env.reset()
    env.wall_positions[3, 4, 0] = 1  # Place horizontal wall at row 3, col 4
    env.wall_positions[3, 6, 0] = 1
    env.render()
    print("\n")

    # Test case 4: Vertical wall placed
    print("=== Test Case 4: Vertical Wall Placed ===")
    env.reset()
    env.wall_positions[2, 3, 1] = 1  # Place vertical wall at row 2, col 3
    env.render()
    print("\n")

    # Test case 5: Combined scenario
    print("=== Test Case 5: Combined Scenario ===")
    env.reset()
    env.player_positions["player_1"] = (3, 4)
    env.player_positions["player_2"] = (6, 4)
    env.wall_positions[2, 4, 0] = 1  # Horizontal wall
    env.wall_positions[5, 3, 1] = 1  # Vertical wall
    env.render()
    print("\n")

if __name__ == "__main__":
    # test_render()
    env = Quoridor()
    api_test(env, num_cycles=100, verbose_progress=True)

    # player_positions = {"player_1": (0, 4), "player_2": (8, 4)}
    # wall_positions = np.zeros((8, 8, 2)) #horizontal then vertical
    # wall_positions[0][3][1] = 1
    # wall_positions[1][3][0] = 1
    # player_1_action_map = np.ones(136)
    # player_2_action_map = np.ones(136)
    # test_new_walls(player_positions, wall_positions, player_1_action_map, player_2_action_map)
    # print(player_1_action_map)
    # print(player_2_action_map)
