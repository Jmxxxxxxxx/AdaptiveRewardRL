import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

try:
    import seaborn as sns
except ImportError:
    sns = None


SMOKE_TEST = False
BENCHMARK_ENV = True
BENCHMARK_MODE = "medium"

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
    SEEDS = list(range(10))
else:
    METHODS = [
        "progress_adrs",
        "hybrid_adrs",
        "hybrid",
        "progress",
    ]
    SEEDS = list(range(10))

LINEWIDTH = 2.2
SHADE_ALPHA = 0.18
FIGSIZE = (13.5, 4.8)
DPI = 600

TOY_TASKS = [
    "toy_reach_o",
    "toy_safe_reach_o",
    "toy_safe_reach_o_or_b",
    "toy_safe_b_then_o",
    "toy_safe_o_and_b_any_order",
]

PALETTE = {
    "Adaptive Hybrid": "#0072B2",
    "Hybrid": "#56B4E9",
    "Adaptive Progression": "#D55E00",
    "Progress": "#E69F00",
}

RUN_CONFIGS = {
    "hybrid_adrs": {
        "label": "Adaptive Hybrid",
        "dir": "hybrid_adrs_theta20_update20",
        "color": PALETTE["Adaptive Hybrid"],
    },
    "hybrid": {
        "label": "Hybrid",
        "dir": "hybrid_theta20_update20",
        "color": PALETTE["Hybrid"],
    },
    "progress_adrs": {
        "label": "Adaptive Progression",
        "dir": "progress_adrs_theta20_update20",
        "color": PALETTE["Adaptive Progression"],
    },
    "progress": {
        "label": "Progress",
        "dir": "progress_theta20_update20",
        "color": PALETTE["Progress"],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot multiseed reward/success curves for toy LTL tasks."
    )
    parser.add_argument(
        "--env_name",
        choices=["toy", "toy_benchmark", "toy_benchmark_medium", "toy_benchmark_hard"],
        default=default_env_name(),
        help="Base toy environment name to plot.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for plots and summary. Defaults to <env_name>_until_o_plots.",
    )
    return parser.parse_args()


def output_dir(env_name, output_dir_arg):
    if output_dir_arg is not None:
        return Path(output_dir_arg)
    return Path(f"{env_name}_LTL_multiseed_plots")


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


def result_env_name(env_name):
    if BENCHMARK_MODE == "toy_original":
        return "toy_original_rerun"
    if SMOKE_TEST:
        return f"{env_name}_smoke"
    return env_name


def make_runs(env_name, toy_task):
    base = Path("log") / result_env_name(env_name) / toy_task / "dqn"
    runs = []
    for method in METHODS:
        config = RUN_CONFIGS[method]
        runs.append(
            {
                "name": method,
                "label": config["label"],
                "path": base / config["dir"],
                "color": config["color"],
            }
        )
    return runs


def reduce_eval_array(array):
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        return array
    return np.mean(array, axis=1)


def load_seed(path):
    with np.load(path, allow_pickle=True) as data:
        timesteps = np.asarray(data["timesteps"], dtype=float)
        rewards = reduce_eval_array(data["results"])
        successes = reduce_eval_array(data["successes"])
        ep_lengths = reduce_eval_array(data["ep_lengths"])

    min_len = min(len(timesteps), len(rewards), len(successes), len(ep_lengths))
    return {
        "timesteps": timesteps[:min_len],
        "reward": rewards[:min_len],
        "success": successes[:min_len],
        "ep_length": ep_lengths[:min_len],
    }


def common_timesteps(seed_results):
    timestep_sets = [set(result["timesteps"]) for result in seed_results]
    common = sorted(set.intersection(*timestep_sets))
    return np.asarray(common, dtype=float)


def align_metric(seed_results, timesteps, metric):
    aligned = []
    for result in seed_results:
        values_by_timestep = dict(zip(result["timesteps"], result[metric]))
        aligned.append([values_by_timestep[timestep] for timestep in timesteps])
    return np.asarray(aligned, dtype=float)


def aggregate_run(run):
    seed_results = []
    used_seeds = []

    for seed in SEEDS:
        path = run["path"] / str(seed) / "evaluations.npz"
        if not path.exists():
            print(f"[MISSING] {path}")
            continue

        seed_results.append(load_seed(path))
        used_seeds.append(seed)

    if not seed_results:
        print(f"WARNING no data found for {run['label']}")
        return None

    timesteps = common_timesteps(seed_results)
    if len(timesteps) == 0:
        print(f"WARNING no common timesteps for {run['label']}")
        return None

    reward = align_metric(seed_results, timesteps, "reward")
    success = align_metric(seed_results, timesteps, "success")
    ep_length = align_metric(seed_results, timesteps, "ep_length")

    return {
        "name": run["name"],
        "label": run["label"],
        "color": run["color"],
        "seeds": used_seeds,
        "timesteps": timesteps,
        "reward_mean": np.mean(reward, axis=0),
        "reward_std": np.std(reward, axis=0),
        "success_mean": np.mean(success, axis=0),
        "success_std": np.std(success, axis=0),
        "ep_length_mean": np.mean(ep_length, axis=0),
        "ep_length_std": np.std(ep_length, axis=0),
    }


def first_success_timestep(timesteps, success_mean):
    indices = np.where(success_mean > 0)[0]
    if len(indices) == 0:
        return ""
    return int(timesteps[indices[0]])


def output_stem(toy_task):
    if BENCHMARK_MODE == "toy_original":
        return f"{toy_task}_toy_original"
    if BENCHMARK_MODE == "medium" and SMOKE_TEST:
        return f"{toy_task}_benchmark_medium_smoke"
    if BENCHMARK_MODE == "medium":
        return f"{toy_task}_benchmark_medium"
    if BENCHMARK_MODE == "hard" and SMOKE_TEST:
        return f"{toy_task}_benchmark_hard_smoke"
    if BENCHMARK_MODE == "hard":
        return f"{toy_task}_benchmark_hard"
    if BENCHMARK_ENV and SMOKE_TEST:
        return f"{toy_task}_benchmark_smoke"
    if BENCHMARK_ENV:
        return f"{toy_task}_benchmark"
    if SMOKE_TEST:
        return f"{toy_task}_smoke"
    return toy_task


def write_summary(output_path, toy_task, aggregated):
    path = output_path / f"{output_stem(toy_task)}_multiseed_summary.csv"
    fieldnames = [
        "run_name",
        "label",
        "num_seeds",
        "seeds",
        "final_timestep",
        "mean_final_reward",
        "std_final_reward",
        "mean_final_success_rate",
        "std_final_success_rate",
        "mean_final_episode_length",
        "std_final_episode_length",
        "first_success_timestep_mean_curve",
    ]
    if SMOKE_TEST:
        fieldnames = [
            "toy_task",
            "method",
            *fieldnames,
            "final_reward",
            "final_success",
            "final_ep_length",
            "first_success_timestep",
        ]

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in aggregated:
            row = {
                "run_name": result["name"],
                "label": result["label"],
                "num_seeds": len(result["seeds"]),
                "seeds": " ".join(str(seed) for seed in result["seeds"]),
                "final_timestep": int(result["timesteps"][-1]),
                "mean_final_reward": result["reward_mean"][-1],
                "std_final_reward": result["reward_std"][-1],
                "mean_final_success_rate": result["success_mean"][-1],
                "std_final_success_rate": result["success_std"][-1],
                "mean_final_episode_length": result["ep_length_mean"][-1],
                "std_final_episode_length": result["ep_length_std"][-1],
                "first_success_timestep_mean_curve": first_success_timestep(
                    result["timesteps"], result["success_mean"]
                ),
            }
            if SMOKE_TEST:
                row.update(
                    {
                        "toy_task": toy_task,
                        "method": result["name"],
                        "final_reward": result["reward_mean"][-1],
                        "final_success": result["success_mean"][-1],
                        "final_ep_length": result["ep_length_mean"][-1],
                        "first_success_timestep": first_success_timestep(
                            result["timesteps"], result["success_mean"]
                        ),
                    }
                )
            writer.writerow(row)

    return path


def draw_plot(output_path, toy_task, aggregated):
    if sns is not None:
        sns.set_theme(
            style="darkgrid",
            font_scale=1.5,
            rc={
                "axes.facecolor": "#EAEAF2",
                "grid.color": "white",
                "grid.linewidth": 1.2,
                "axes.edgecolor": "#EAEAF2",
            },
        )

    plt.rcParams.update({"font.family": "Arial"})

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 2))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=FIGSIZE)

    for result in aggregated:
        x = result["timesteps"]
        color = result["color"]
        label = result["label"]

        axes[0].plot(
            x,
            result["reward_mean"],
            label=label,
            linestyle="-",
            color=color,
            linewidth=LINEWIDTH,
        )
        axes[0].fill_between(
            x,
            result["reward_mean"] - result["reward_std"],
            result["reward_mean"] + result["reward_std"],
            alpha=SHADE_ALPHA,
            color=color,
            linewidth=0,
        )

        axes[1].plot(
            x,
            result["success_mean"],
            label=label,
            linestyle="-",
            color=color,
            linewidth=LINEWIDTH,
        )
        axes[1].fill_between(
            x,
            result["success_mean"] - result["success_std"],
            result["success_mean"] + result["success_std"],
            alpha=SHADE_ALPHA,
            color=color,
            linewidth=0,
        )

    axes[0].set_title(r"$\theta = 20$", fontsize=16, fontname="Arial")
    axes[0].set_xlabel("Training Steps", fontdict={"fontsize": 15, "fontname": "Arial"})
    axes[0].set_ylabel("Reward", fontdict={"fontsize": 15, "fontname": "Arial"})

    axes[1].set_title("Update Interval: 20", fontsize=16, fontname="Arial")
    axes[1].set_xlabel("Training Steps", fontdict={"fontsize": 15, "fontname": "Arial"})
    axes[1].set_ylabel("Success Rate", fontdict={"fontsize": 15, "fontname": "Arial"})
    axes[1].set_ylim(-0.2, 1.08)

    for ax in axes:
        ax.set_facecolor("#EAEAF2")
        ax.set_xlim(0, 10000)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(labelsize=13)
        ax.grid(True, color="white", linewidth=1.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(
        loc="lower right",
        ncol=1,
        prop={"size": 13, "family": "Arial"},
        frameon=False,
    )

    fig.tight_layout()
    png_path = output_path / f"{output_stem(toy_task)}_reward_success.png"
    pdf_path = output_path / f"{output_stem(toy_task)}_reward_success.pdf"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def main():
    args = parse_args()
    out_dir = output_dir(args.env_name, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plotted = False
    for toy_task in TOY_TASKS:
        print(f"=== toy_task={toy_task} ===")
        aggregated = []
        for run in make_runs(args.env_name, toy_task):
            result = aggregate_run(run)
            if result is not None:
                aggregated.append(result)

        if not aggregated:
            print(f"WARNING no results found for {toy_task}. Nothing to plot.")
            continue

        summary_path = write_summary(out_dir, toy_task, aggregated)
        png_path, pdf_path = draw_plot(out_dir, toy_task, aggregated)
        plotted = True

        print(f"Saved summary: {summary_path}")
        print(f"Saved plot: {png_path}")
        print(f"Saved plot: {pdf_path}")

    if not plotted:
        message = "WARNING no results found. Nothing to plot."
        if SMOKE_TEST:
            print(message)
            return
        raise SystemExit(message)


if __name__ == "__main__":
    main()
