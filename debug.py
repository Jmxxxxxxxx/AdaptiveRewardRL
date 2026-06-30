import os
import sys

# Force CPU mode. Avoid CUDA error on RTX 5060 / unsupported PyTorch CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.argv = [
    "run.py",
    "--env_name", "toy_test",
    "--total_timesteps", "10000",
    "--total_run", "1",
    "--episode_step", "25",
    "--reward_types", "p",
    "--default_setting", "True",
    "--seed", "0",
    "--algo_name", "dqn",
    "--use_adrs", "True",
    "--node_embedding", "True",
    "--eval_freq", "100",
]

import run