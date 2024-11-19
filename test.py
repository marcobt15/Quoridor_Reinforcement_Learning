from quoridor import Quoridor
from other import CustomActionMaskedEnvironment
from pettingzoo.test import api_test, parallel_api_test
from quoridor2 import QuoridorEnv

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
    # env = QuoridorEnv()
    env = Quoridor()
    # env.reset()
    # print(env.last())
    # print(env.observation_space("player_1"))
    api_test(env, num_cycles=1000, verbose_progress=True)
    # env = CustomActionMaskedEnvironment()
    # parallel_api_test(env, num_cycles=1_000_000)
