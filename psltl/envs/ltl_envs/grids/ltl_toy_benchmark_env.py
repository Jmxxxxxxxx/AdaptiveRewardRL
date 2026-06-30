from psltl.envs.common.grids.craft_world import CraftWorld
from psltl.envs.ltl_envs.grids.ltl_grid_env import LTLGridEnv
from psltl.envs.ltl_envs.grids.ltl_toy_env import LTLToyEnv
from psltl.ltl.partial_sat_atm_load import LoadedPartialSatATM
from psltl.envs.skeletons.env_default_settings import reward_kwargs, setting


class LTLToyBenchmarkEnv(LTLToyEnv):
    def __init__(
        self,
        atm: LoadedPartialSatATM,
        max_episode_steps: int = 50,
        action_dim: int = 4,
        reward_kwargs: dict = reward_kwargs,
        setting: dict = setting,
):
        env = CraftWorld("/psltl/envs/common/grids/maps/toy_benchmark.txt")
        LTLGridEnv.__init__(self, env, atm, max_episode_steps, action_dim, reward_kwargs, setting)
