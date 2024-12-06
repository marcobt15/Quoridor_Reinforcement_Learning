import quoridor
import new
from pettingzoo.test import api_test
import glob
from sb3_contrib import MaskablePPO
import os
from a_star import a_star

import matplotlib.pyplot as plt

def eval_action_mask(env_fn, num_games=100, a_star_flag=False, simulate=False, render_mode=True, model_opponent = "", agent_player = "player_1", real_player=False, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(**env_kwargs)
    if a_star_flag == True:
        opponent = "A star"
    elif model_opponent != "":
        opponent = model_opponent
    elif real_player == True:
        jump = True
        opponent = "you"
    else :
        opponent = "random agent"
        
    print(
        f"Starting evaluation vs {opponent}. Trained agent will play as {agent_player}."
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
        if model_opponent != "":     
            opponent_policy = max(
                glob.glob(f"{opponent}.zip"), key=os.path.getctime
            )
            model_opp = MaskablePPO.load(opponent_policy)
    except ValueError:
        print("Policy not found.")
        exit(0)

    print("loading", latest_policy)

    model = MaskablePPO.load(latest_policy)

    scores = {agent: 0 for agent in env.possible_agents}
    truncation_count = 0
    loss_count = 0
    winrate_progression = []
    truncation_rate_progression = []
    loss_rate_progression = []

    for i in range(num_games):
        env.reset()
        if render_mode:
            env.render()

        game_truncated = False
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            # Separate observation and action mask
            observation, action_mask = obs.values()

            if termination or truncation:
                if truncation:
                    game_truncated = True

                if (
                    env.rewards[env.possible_agents[0]]
                    != env.rewards[env.possible_agents[1]]
                ):
                    winner = max(env.rewards, key=env.rewards.get)
                    scores[winner] += 1  # Increment winner's score
                else:
                    loss_count += 1  # Increment loss count if it's not a win
                break

            #game still going on
            else:
                if agent == agent_player:
                    act = int(
                            model.predict(
                                observation, action_masks=action_mask, deterministic=True
                            )[0]
                        )
                else:
                    if a_star_flag:
                        curr_position = info["position"]
                        optimal_path, cost = a_star(curr_position, agent, info["wall_locations"])
                        next_position = optimal_path[1]
                        #print(optimal_path)
                        if next_position[0] > curr_position[0]:
                            act = 5
                        elif next_position[0] < curr_position[0]:
                            act = 4
                        elif next_position[1] < curr_position[1]:
                            if agent == "player_2":
                                act = 7
                            else:
                                act = 6
                        elif next_position[1] > curr_position[1]:
                            if agent == "player_2":
                                act = 6
                            else:
                                act = 7
                            
                    elif model_opponent != "":
                        #print("player 2 thinking")
                        act = int(
                            model_opp.predict(
                                observation, action_masks=action_mask, deterministic=True
                            )[0]
                        )

                    elif real_player:
                        choice = int(input("1 to place a wall and 2 to move/jump: "))
                        if choice == 1:
                            orientation = input("h to place a horizontal wall, v to place a vertical wall: ")
                            orientation = 0 if orientation == "h" else 1
                            row = int(input("What row would you like to place the wall: "))
                            col = int(input("What column would you like to place the wall: "))
                            wall_pos = 8*row + col
                            act = wall_pos if orientation == 0 else wall_pos + 64

                        elif choice == 2:
                            if jump:
                                move = input("use wasd to move and ijkl to jump")
                            else:
                                move = input("use wasd to move")
                            while not jump and move in ['ijkl']:
                                print("cannot jump anymore")
                                move = input("use wasd to move and ijkl to jump")
                            
                            if jump and move == 'w':
                                jump = False
                                direction = 0
                            elif jump and move == 'a':
                                jump = False
                                direction = 2
                            elif jump and move == 's':
                                jump = False
                                direction = 1
                            elif jump and move == 'd':
                                jump = False
                                direction = 3  
                            elif move == 'w':
                                direction = 4
                            elif move == 'a':
                                direction = 5
                            elif move == 's':
                                direction = 6
                            elif move == 'd':
                                direction = 7

                            if agent == "player_2":
                                if direction == 4:
                                    direction = 5
                                elif direction == 5:
                                    direction = 4
                                elif direction == 6:
                                    direction = 7
                                else:
                                    direction = 6
                            act = direction

                    else:
                        act = env.action_space(agent).sample(action_mask)
  
            env.step(act)
            if render_mode:
                env.render()

        # Track truncation
        if game_truncated:
            truncation_count += 1

        # Calculate metrics
        if simulate:
            trained_agent_wins = scores[env.possible_agents[0]]
            winrate = (trained_agent_wins / (i + 1)) * 100  # Convert to percentage
            truncation_rate = (truncation_count / (i + 1)) * 100  # Convert to percentage
            loss_rate = (loss_count / (i + 1)) * 100  # Convert to percentage

            winrate_progression.append(winrate)
            truncation_rate_progression.append(truncation_rate)
            loss_rate_progression.append(loss_rate)

    env.close()

    # Plot metrics if simulate is enabled
    if simulate:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_games + 1), winrate_progression, label="Winrate (%)", marker='o')
        plt.plot(range(1, num_games + 1), truncation_rate_progression, label="Truncation Rate (%)", marker='x')
        plt.plot(range(1, num_games + 1), loss_rate_progression, label="Loss Rate (%)", marker='s')
        plt.xlabel('Number of Games')
        plt.ylabel('Rate (%)')
        plt.ylim(0, 100)  # Scale y-axis from 0 to 100%
        plt.title(f"Performance Metrics of {env.metadata['name']} vs {opponent}")
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    option = int(input("1 for api test, 2 for model test vs random, 3 for model test vs A*, 4 for simulate, 5 to play agaisnt an agent: "))
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
    elif option == 4:
        env_fn = quoridor
        eval_action_mask(env_fn, a_star_flag=False, simulate=True, render_mode=False, agent_player = "player_1", model_opponent = "quoridor_aec_v4_best_agent")
    elif option == 5:
        env_fn = quoridor
        player = int(input("Input 1 to be player1 and 2 to be player2: "))
        player = "player_2" if player == 1 else "player_1"
        eval_action_mask(env_fn, render_mode=True, agent_player=player, real_player=True)