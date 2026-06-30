import argparse
import os
import pickletools
import shutil
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


SMOKE_TEST = False
BENCHMARK_ENV = True
BENCHMARK_MODE = "medium"

TOY_TASKS = [
    "toy_reach_o",
    "toy_safe_reach_o",
    "toy_safe_reach_o_or_b",
    "toy_safe_b_then_o",
    "toy_safe_o_and_b_any_order",
]

METHOD_CONFIGS = {
    "progress_adrs": ("p", "True"),
    "hybrid_adrs": ("h", "True"),
    "hybrid": ("h", "False"),
    "progress": ("p", "False"),
}

EXPECTED_LTLS = {
    "toy_reach_o": "F o",
    "toy_safe_reach_o": "(!y) U o",
    "toy_safe_reach_o_or_b": "(!y) U (o | b)",
    "toy_safe_b_then_o": "(!y) U (b & ((!y) U o))",
    "toy_safe_o_and_b_any_order": "(!y) U ((o & ((!y) U b)) | (b & ((!y) U o)))",
}

if SMOKE_TEST and BENCHMARK_MODE == "medium":
    METHODS = [
        "progress_adrs",
        "hybrid_adrs",
        "hybrid",
        "progress",
    ]
    SEEDS = [0]
elif SMOKE_TEST:
    METHODS = ["progress_adrs"]
    SEEDS = [0]
elif BENCHMARK_MODE == "medium":
    METHODS = [
        "progress_adrs",
        "hybrid_adrs",
        "hybrid",
        "progress",
    ]
    SEEDS = list(range(5, 10))
else:
    METHODS = [
        "progress_adrs",
        "hybrid_adrs",
        "hybrid",
        "progress",
    ]
    SEEDS = list(range(10))

ALGO_NAME = "dqn"
THETA = "20"
ADRS_UPDATE = "20"

RUNS = [
    (method, *METHOD_CONFIGS[method])
    for method in METHODS
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sequential multiseed runner for toy LTL experiments."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun even if the target seed directory already has evaluations.npz.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Seed list to run. Example: --seeds 0 1 2 3 4",
    )
    parser.add_argument(
        "--env_name",
        choices=["toy", "toy_benchmark", "toy_benchmark_medium", "toy_benchmark_hard"],
        default=default_env_name(),
        help="Base toy environment name to run.",
    )
    return parser.parse_args()


def common_args(env_name):
    return [
        "--env_name",
        env_name,
        "--total_timesteps",
        "10000",
        "--total_run",
        "1",
        "--episode_step",
        "25",
        "--default_setting",
        "True",
        "--algo_name",
        ALGO_NAME,
        "--node_embedding",
        "True",
        "--eval_freq",
        "100",
        "--theta",
        THETA,
        "--adrs_update",
        ADRS_UPDATE,
    ]


def default_env_name():
    if BENCHMARK_MODE == "toy_original":
        return "toy"
    if BENCHMARK_MODE == "medium":
        return "toy_benchmark_medium"
    if BENCHMARK_MODE == "hard":
        return "toy_benchmark_hard"
    if BENCHMARK_ENV:
        return "toy_benchmark"
    return "toy"


def expected_action_space():
    if BENCHMARK_MODE == "toy_original":
        return "Discrete(2)"
    if BENCHMARK_MODE in ["benchmark", "medium", "hard"] or BENCHMARK_ENV:
        return "Discrete(4)"
    return "Discrete(2)"


def expected_max_episode_steps():
    if BENCHMARK_MODE == "toy_original":
        return 25
    if BENCHMARK_MODE == "medium":
        return 80
    if BENCHMARK_MODE == "hard":
        return 200
    if BENCHMARK_ENV:
        return 50
    return 25


def reward_dir_name(reward_type, use_adrs):
    reward_name = {"p": "progress", "h": "hybrid"}[reward_type]
    if use_adrs == "True":
        reward_name += "_adrs"
    return f"{reward_name}_theta{THETA}_update{ADRS_UPDATE}"


def expected_eval_file(env_name, toy_task, reward_type, use_adrs, seed):
    return (
        task_log_dir(result_env_name(env_name), toy_task, reward_type, use_adrs, seed)
        / "evaluations.npz"
    )


def source_eval_file(env_name, reward_type, use_adrs, seed):
    return (
        Path("log")
        / env_name
        / ALGO_NAME
        / reward_dir_name(reward_type, use_adrs)
        / str(seed)
        / "evaluations.npz"
    )


def task_log_dir(env_name, toy_task, reward_type, use_adrs, seed):
    return (
        Path("log")
        / env_name
        / toy_task
        / ALGO_NAME
        / reward_dir_name(reward_type, use_adrs)
        / str(seed)
    )


def result_env_name(env_name):
    if BENCHMARK_MODE == "toy_original":
        return "toy_original_rerun"
    if SMOKE_TEST:
        return f"{env_name}_smoke"
    return env_name


def fixed_automaton_paths(env_name):
    base = Path("psltl") / "ltl" / "ltl_infos" / env_name
    return base / "info.pkl", base / "delta.pkl"


def task_automaton_paths(env_name, toy_task):
    base = Path("psltl") / "ltl" / "ltl_infos" / env_name / toy_task
    return base / "info.pkl", base / "delta.pkl"


def read_ltl_from_info(info_path):
    previous_string = None
    with info_path.open("rb") as file:
        for opcode, arg, _ in pickletools.genops(file):
            if opcode.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
                if previous_string == "ltl":
                    return arg
                previous_string = arg
    return "<missing ltl>"


def install_task_automaton(env_name, toy_task):
    src_info, src_delta = task_automaton_paths(env_name, toy_task)
    dst_info, dst_delta = fixed_automaton_paths(env_name)

    if not src_info.exists() or not src_delta.exists():
        raise FileNotFoundError(f"Missing automaton files for {toy_task}: {src_info}, {src_delta}")

    shutil.copy2(src_info, dst_info)
    shutil.copy2(src_delta, dst_delta)


def print_task_debug(env_name, toy_task, method, seed, save_dir):
    info_path, delta_path = task_automaton_paths(env_name, toy_task)

    print("==============================")
    print(f"[MODE] {BENCHMARK_MODE}")
    print(f"[BENCHMARK_ENV] {BENCHMARK_ENV}")
    print(f"[BENCHMARK_MODE] {BENCHMARK_MODE}")
    print(f"[ENV_NAME] {env_name}")
    print(f"[SMOKE_TEST] {SMOKE_TEST}")
    print(f"[TOY_TASK] {toy_task}")
    print(f"[LTL] {read_ltl_from_info(info_path)}")
    print(f"[EXPECTED_LTL] {EXPECTED_LTLS[toy_task]}")
    print(f"[METHOD] {method}")
    print(f"[SEED] {seed}")
    print(f"[INFO_PKL] {info_path}")
    print(f"[DELTA_PKL] {delta_path}")
    print(f"[SAVE_DIR] {save_dir}")
    print(f"[ACTION_SPACE] {expected_action_space()}")
    print(f"[MAX_EPISODE_STEPS] {expected_max_episode_steps()}")
    print("==============================")


def backup_fixed_automaton(env_name, backup_dir):
    backup = {}
    for path in fixed_automaton_paths(env_name):
        if path.exists():
            backup[path] = backup_dir / path.name
            shutil.copy2(path, backup[path])
        else:
            backup[path] = None
    return backup


def restore_fixed_automaton(backup):
    for path, backup_path in backup.items():
        if backup_path is None:
            if path.exists():
                path.unlink()
        else:
            shutil.copy2(backup_path, path)


def build_command(env_name, reward_types, use_adrs, seed):
    return [
        sys.executable,
        "run.py",
        *common_args(env_name),
        "--reward_types",
        reward_types,
        "--use_adrs",
        use_adrs,
        "--seed",
        str(seed),
    ]


def format_command(command):
    return " ".join(shlex.quote(part) for part in command)


def main():
    args = parse_args()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    with tempfile.TemporaryDirectory() as temp_dir:
        backup = backup_fixed_automaton(args.env_name, Path(temp_dir))
        try:
            for toy_task in TOY_TASKS:
                print(f"=== toy_task={toy_task} ===")
                install_task_automaton(args.env_name, toy_task)

                for run_name, reward_types, use_adrs in RUNS:
                    for seed in args.seeds:
                        eval_file = expected_eval_file(
                            args.env_name, toy_task, reward_types, use_adrs, seed
                        )

                        if eval_file.exists() and not args.overwrite:
                            print(
                                f"SKIP {toy_task} {run_name} seed={seed}: "
                                f"existing result {eval_file}"
                            )
                            continue

                        print_task_debug(
                            args.env_name,
                            toy_task,
                            run_name,
                            seed,
                            eval_file.parent,
                        )
                        command = build_command(args.env_name, reward_types, use_adrs, seed)
                        print(format_command(command))
                        result = subprocess.run(command, env=env)
                        if result.returncode != 0:
                            print(
                                f"FAILED returncode={result.returncode}: "
                                f"{format_command(command)}"
                            )
                            return result.returncode

                        source_file = source_eval_file(
                            args.env_name, reward_types, use_adrs, seed
                        )
                        if not source_file.exists():
                            print(f"FAILED missing expected result: {source_file}")
                            return 1

                        eval_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_file, eval_file)
        finally:
            restore_fixed_automaton(backup)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
