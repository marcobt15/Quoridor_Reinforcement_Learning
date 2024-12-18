import time
import glob
import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy, MaskableActorCriticCnnPolicy
from sb3_contrib.common.wrappers import ActionMasker

import pettingzoo.utils
import quoridor

class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper):
    """Wrapper to allow PettingZoo environments to be used with SB3 illegal action masking."""

    def reset(self, seed=None, options=None):
        """Gymnasium-like reset function which assigns obs/action spaces to be the same for each agent.

        This is required as SB3 is designed for single-agent RL and doesn't expect obs/action spaces to be functions
        """
        super().reset(seed, options)

        # Strip the action mask out from the observation space
        self.observation_space = super().observation_space(self.possible_agents[0])[
            "observation"
        ]
        self.action_space = super().action_space(self.possible_agents[0])

        # Return initial observation, info (PettingZoo AEC envs do not by default)
        return self.observe(self.agent_selection), {}

    def step(self, action):
        """Gymnasium-like step function, returning observation, reward, termination, truncation, info."""
        super().step(action)
        return super().last()

    def observe(self, agent):
        """Return only raw observation, removing action mask."""
        return super().observe(agent)["observation"]

    def action_mask(self):
        """Separate function used in order to access the action mask."""
        return super().observe(self.agent_selection)["action_mask"]


def mask_fn(env):
    #return action mask for current env
    return env.action_mask()


def train_action_mask(env_fn, steps=10_000, seed=0, **env_kwargs):
    """Train a single model to play as each agent in a zero-sum game environment using invalid action masking."""
    env = env_fn.env(**env_kwargs)

    print(f"Starting training on {str(env.metadata['name'])}.")

    # Custom wrapper to convert PettingZoo envs to work with SB3 action masking
    env = SB3ActionMaskWrapper(env)

    env.reset(seed=seed)  # Must call reset() in order to re-define the spaces

    env = ActionMasker(env, mask_fn)  # Wrap to enable masking (SB3 function)
    # MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
    # with ActionMasker. If the wrapper is detected, the masks are automatically
    # retrieved and used when learning. Note that MaskablePPO does not accept
    # a new action_mask_fn kwarg, as it did in an earlier draft.
    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1, device="cuda")
    model.set_random_seed(seed)
    model.learn(total_timesteps=steps)

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")

    env.close()

def continue_training(env_fn):
    env = env_fn.env()
    
    try:
        latest_policy = max( # quoridor_aec_v3_movement+jump.zip
            glob.glob(f"quoridor_aec_v6_runs_and_walls.zip"), key=os.path.getctime
        )
        # latest_policy = "quoridor_aec_v1_only_good_one.zip"
    except ValueError:
        print("Policy not found.")
        exit(0)
    
    print("training", latest_policy)

    env = SB3ActionMaskWrapper(env)
    env.reset()  # Must call reset() in order to re-define the spaces
    env = ActionMasker(env, mask_fn)

    model = MaskablePPO.load(latest_policy, env, device="cuda")

    model.learn(total_timesteps=100_000)

    model.save(f"quoridor_aec_v6_runs_and_walls2.zip")

    print("Model has been saved.")

    print(f"Finished training on quoridor_aec_v6_runs_and_wallst2.zip.\n")

if __name__ == "__main__":
    env_fn = quoridor
    env_kwargs = {}

    choice = int(input("1 to train a new model, 2 to continue training a model: "))

    if choice == 1:
        train_action_mask(env_fn, steps=100_000, seed=0, **env_kwargs)
    else:
        continue_training(env_fn)