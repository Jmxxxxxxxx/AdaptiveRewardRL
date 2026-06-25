from psltl.ltl.partial_sat_atm import PartialSatATM
from psltl.ltl.ltl_utils import save_atm

# toy 环境已有的 atomic propositions
AP = ["o", "b", "y"]

# 新 LTL：一直不能碰 y，直到到达 o
ltl = "(!y) U o"

atm = PartialSatATM(ltl, AP)
atm.print_results()

save_atm(
    atm,
    "psltl/ltl/ltl_infos/toy_test/info.pkl",
    "psltl/ltl/ltl_infos/toy_test/delta.pkl",
)

print("Saved toy_test automaton successfully.")