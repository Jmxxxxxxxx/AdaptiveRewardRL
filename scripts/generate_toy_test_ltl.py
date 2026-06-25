from psltl.ltl.partial_sat_atm import PartialSatATM
from psltl.ltl.ltl_utils import save_atm

import os

os.environ["PATH"] += os.pathsep + "/home/jiamin/Github/AdaptiveRewardRL/psltl/ltl"

# toy 环境已有标签：o, b, y
AP = ["o", "b", "y"]

# 新任务：不能碰 y，直到到达 o
ltl = "(!y) U o"

atm = PartialSatATM(ltl, AP)
atm.print_results()

save_atm(
    atm,
    "psltl/ltl/ltl_infos/toy_test/info.pkl",
    "psltl/ltl/ltl_infos/toy_test/delta.pkl",
)

print("Saved toy_test automaton successfully.")