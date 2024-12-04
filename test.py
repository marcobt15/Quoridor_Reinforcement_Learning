import quoridor
import new
from pettingzoo.test import api_test
import glob
from sb3_contrib import MaskablePPO
import os
from a_star import a_star

def eval_action_mask(env_fn, num_games=100, a_star_flag=False, render_mode=None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(**env_kwargs)

    print(
        f"Starting evaluation vs a random agent. Trained agent will play as {env.possible_agents[0]}."
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
        # latest_policy = max(
        #     glob.glob(f"quoridor_aec_v3_new_movement2.zip"), key=os.path.getctime
        # )
    except ValueError:
        print("Policy not found.")
        exit(0)

    print("loading", latest_policy)

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
                    # print("player 2 move")
                    if not a_star_flag:
                        act = env.action_space(agent).sample(action_mask)
                    else:
                        # print("a star decision")
                        curr_position = info["position"]
                        optimal_path, cost = a_star(curr_position, agent, info["wall_locations"])
                        next_position = optimal_path[1]

                        # print(next_position)
                        # print(curr_position)

                        if next_position[0] > curr_position[0]:
                            # print("move up")
                            act = 4
                        elif next_position[0] < curr_position[0]:
                            # print("move down")
                            act = 5
                        elif next_position[1] < curr_position[1]:
                            # print("move left")
                            act = 6
                        elif next_position[1] > curr_position[1]:
                            # print("move right")
                            act = 7

                        # print()
                        # print()
                else:
                    # Note: PettingZoo expects integer actions # TODO: change chess to cast actions to type int?
                    act = int(
                        model.predict(
                            observation, action_masks=action_mask, deterministic=True
                        )[0]
                    )
            env.step(act)
            env.render()
    env.close()

if __name__ == "__main__":
    option = int(input("1 for api test, 2 for model test vs random, 3 for model test vs A*: "))
    if option == 1:
        env = quoridor.Quoridor()
        # env = new.Quoridor()
        api_test(env, num_cycles=50000, verbose_progress=True)
    elif option == 2:
        env_fn = quoridor
        # env_fn = new
        eval_action_mask(env_fn)
    elif option == 3:
        env_fn = quoridor
        # env_fn = new
        eval_action_mask(env_fn, a_star_flag=True)