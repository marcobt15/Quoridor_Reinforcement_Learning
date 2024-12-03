import quoridor
import quoridor_v0
import old_qouridor
from pettingzoo.test import api_test
import glob
from sb3_contrib import MaskablePPO
import os

def test_render():
    # Initialize the Quoridor environment
    env = quoridor.Quoridor()

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


def eval_action_mask(env_fn, num_games=100, render_mode=None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(**env_kwargs)

    print(
        f"Starting evaluation vs a random agent. Trained agent will play as {env.possible_agents[0]}."
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    print(latest_policy)

    model = MaskablePPO.load(latest_policy)

    scores = {agent: 0 for agent in env.possible_agents}
    total_rewards = {agent: 0 for agent in env.possible_agents}
    round_rewards = []

    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            # Separate observation and action mask
            observation, action_mask = obs.values()

            if termination or truncation:
                # If there is a winner, keep track, otherwise don't change the scores (tie)
                if (
                    env.rewards[env.possible_agents[0]]
                    != env.rewards[env.possible_agents[1]]
                ):
                    winner = max(env.rewards, key=env.rewards.get)
                    scores[winner] += env.rewards[
                        winner
                    ]  # only tracks the largest reward (winner of game)
                # Also track negative and positive rewards (penalizes illegal moves)
                for a in env.possible_agents:
                    total_rewards[a] += env.rewards[a]
                # List of rewards by round, for reference
                round_rewards.append(env.rewards)
                break
            else:
                if agent == env.possible_agents[1]:
                    act = env.action_space(agent).sample(action_mask)
                else:
                    # Note: PettingZoo expects integer actions # TODO: change chess to cast actions to type int?
                    act = int(
                        model.predict(
                            observation, action_masks=action_mask, deterministic=True
                        )[0]
                    )
            env.step(act)
            env.render() #how we render but it's bad
    env.close()

if __name__ == "__main__":
    option = int(input("1 for api test, 2 for model test, 3 for render test:"))
    if option == 1:
        # env = old_qouridor.Quoridor()
        # env = quoridor_v0.Quoridor_v0()
        env = quoridor.Quoridor()
        api_test(env, num_cycles=50000, verbose_progress=True)
    elif option == 2:
        env_fn = quoridor
        # env_fn = quoridor_v0
        eval_action_mask(env_fn)
    else:
        test_render()
