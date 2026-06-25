import sys

sys.argv = [
    "run.py",
    "--env_name", "toy",
    "--total_timesteps", "10000",
    "--total_run", "1",
    "--episode_step", "25",
    "--reward_types", "p",
    "--default_setting", "True", #learning_param.py
    "--seed", "0",
    "--algo_name", "dqn",
    "--use_adrs", "True",
    "--node_embedding", "True",
    "--eval_freq", "100", #隔多久测试一次
]

import run