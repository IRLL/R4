"""
Entry point for rMSE reward learning.
"""

import argparse
import os
import uuid
from copy import deepcopy

import numpy as np
import torch

from optimization import NNOptimization
from rewardFunctions.reward_functions import RewardFunctionSimple, get_reward_functionNN
from testing.sac_test import SAC_Test
from utils import create_model, read_bin_data, read_env_config, read_pickle, write_pickle


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mini", help="The model to be optimized")
parser.add_argument("--num_generations", type=int, default=1500, help="Number of optimization iterations")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for optimization")
parser.add_argument(
    "--loss_func",
    type=str,
    default="ranking_mse",
    help="Loss function (ranking_mse, ranking_mse_ot)",
)
parser.add_argument("--ranking_assumption", type=str, default="bin_ranking", help="full_ranking or bin_ranking")
parser.add_argument("--env_name", type=str, default="hungrythirsty", help="Environment name")
parser.add_argument("--num_trajectories", type=int, default=1000, help="Number of trajectories to pick")
parser.add_argument("--num_files_to_read", type=int, default=1, help="Number of data files to read")
parser.add_argument("--traj_file_to_read", type=str, default=None, help="Specify the file prefix to read")
parser.add_argument("--sample_using_bins", type=bool, default=True, help="Sample equal number of trajectories from each bin")
parser.add_argument("--bin_type", type=str, default="default", help="default or equal")
parser.add_argument("--num_bins", type=int, default=5, help="Number of bins for sampling trajectories")
parser.add_argument("--bin_distribution", type=str, default="uniform", help="uniform or imbalanced")
parser.add_argument("--imbalanced_bin_factor", type=float, default=0.95, help="Factor for imbalanced binning")
parser.add_argument("--noisy_bins", type=bool, default=False, help="Use noisy bins")
parser.add_argument("--noise_factor", type=float, default=0.1, help="Noise factor for noisy bins")
parser.add_argument("--human_data", type=bool, default=False, help="Use human labeled data")
parser.add_argument("--rmse_regularization_strength", type=float, default=1.0, help="Regularization strength for ranking mse loss")
parser.add_argument("--test_type", type=str, default="q_test", help="q_test, dq_test, sac_test, or none")
parser.add_argument("--test_seeds", type=int, nargs="+", default=[1, 2], help="Seeds to use for tests")
parser.add_argument("--num_test_episodes", type=int, default=5000, help="Number of test episodes")
args = parser.parse_args()

state_action = []
gt_returns = []

env_config = read_env_config(args.env_name)
args.input_size = env_config["input_dim"]
if "num_test_episodes" in env_config:
    args.num_test_episodes = env_config["num_test_episodes"]

run_guid = str(uuid.uuid4()).split("-")[0]
print(f"Run guid: {run_guid}")
print(args)

if args.traj_file_to_read is not None:
    state_action = read_pickle(args.traj_file_to_read + "_state_action.pkl")
    gt_returns = read_pickle(args.traj_file_to_read + "_gt_returns.pkl")
    last_after_slash = args.traj_file_to_read.split("/")[-1]
    checkpoint_dir = f"../checkpoints/ours/{last_after_slash}"
    os.makedirs(checkpoint_dir, exist_ok=True)
else:
    checkpoint_dir = f"../checkpoints/ours/{args.num_files_to_read}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Phase 1: read only returns to build a global index of (file_idx, traj_idx)
    if not args.human_data:
        flat_returns = []
        flat_indices = []
        for i in range(args.num_files_to_read):
            file_returns = read_pickle(
                f"../data/{args.env_name}/trajectories_(0, 0, 1.0, 1.0)_seed{i}_returns.pkl"
            )
            flat_returns.extend(file_returns)
            flat_indices.extend([(i, j) for j in range(len(file_returns))])

    if args.human_data:
        person_id = args.num_files_to_read - 1
        state_action_file = (
            f"../data/{args.env_name}_human/trajectories_(0, 0, 1.0, 1.0)_p{person_id}_state_action_pairs.pkl"
        )
        bin_file = f"../data/{args.env_name}_human/trajectories_(0, 0, 1.0, 1.0)_p{person_id}_labels.pkl"
        gt_returns_file = (
            f"../data/{args.env_name}_human/trajectories_(0, 0, 1.0, 1.0)_p{person_id}_returns.pkl"
        )

        state_action_ = read_pickle(state_action_file)
        labels = read_pickle(bin_file)
        gt_returns_ = read_pickle(gt_returns_file)
        unique_labels = list(set(labels))

        args.num_bins = len(unique_labels)
        args.bins, t_per_bin = read_bin_data(args.env_name, bin_type="human", args=args)
        args.t_per_bin = t_per_bin

        # group state_action and gt_returns by labels
        state_action_binned = [[] for _ in range(args.num_bins)]
        gt_returns_binned = [[] for _ in range(args.num_bins)]
        for sa, ret, label in zip(state_action_, gt_returns_, labels):
            label_idx = unique_labels.index(label)
            state_action_binned[label_idx].append(sa)
            gt_returns_binned[label_idx].append(ret)

        # sample from each bin according to t_per_bin
        state_action = [[] for _ in range(args.num_bins)]
        gt_returns = [[] for _ in range(args.num_bins)]
        for b in range(args.num_bins):
            if len(state_action_binned[b]) < t_per_bin[b]:
                chosen = np.random.choice(len(state_action_binned[b]), t_per_bin[b], replace=True)
            else:
                chosen = np.random.choice(len(state_action_binned[b]), t_per_bin[b], replace=False)

            state_action[b] = deepcopy([state_action_binned[b][i] for i in chosen])
            gt_returns[b] = deepcopy([gt_returns_binned[b][i] for i in chosen])
        print("Number of trajectories per bin: ", [len(state_action[b]) for b in range(args.num_bins)])

    elif not args.sample_using_bins:
        random_traj_indices = np.random.choice(len(state_action), args.num_trajectories, replace=False)
        state_action = [state_action[i] for i in random_traj_indices]
        gt_returns = [gt_returns[i] for i in random_traj_indices]
    else:
        # Bin-based sampling; build selections in index space first
        args.bins, t_per_bin = read_bin_data(args.env_name, bin_type=args.bin_type, args=args)

        max_trajs_per_bin = max(t_per_bin)
        print(f"Number of trajectories per bin: {t_per_bin}")

        selected_per_bin = [[] for _ in range(len(args.bins) - 1)]

        for b in range(len(args.bins) - 1):
            bin_start = args.bins[b]
            bin_end = args.bins[b + 1]
            bin_global_indices = [gi for gi, ret in enumerate(flat_returns) if ret >= bin_start and ret <= bin_end]
            print(f"Bin {b}: {bin_start} - {bin_end}, num trajectories: {len(bin_global_indices)}")

            if len(bin_global_indices) < t_per_bin[b]:
                chosen = np.random.choice(bin_global_indices, t_per_bin[b], replace=True)
            else:
                chosen = np.random.choice(bin_global_indices, t_per_bin[b], replace=False)

            if t_per_bin[b] < max_trajs_per_bin:
                extra = np.random.choice(chosen, max_trajs_per_bin - t_per_bin[b], replace=True)
                chosen = np.concatenate((chosen, extra))

            for gi in chosen:
                fidx, tidx = flat_indices[gi]
                selected_per_bin[b].append((fidx, tidx, flat_returns[gi]))

        state_action_ = [[] for _ in range(len(args.bins) - 1)]
        gt_returns_ = [[] for _ in range(len(args.bins) - 1)]

        selections_by_file = {}
        for b in range(len(selected_per_bin)):
            for (fidx, tidx, ret) in selected_per_bin[b]:
                if fidx not in selections_by_file:
                    selections_by_file[fidx] = []
                selections_by_file[fidx].append((b, tidx))

        for fidx, selections in selections_by_file.items():
            sa_file = f"../data/{args.env_name}/trajectories_(0, 0, 1.0, 1.0)_seed{fidx}_state_action_pairs.pkl"
            r_file = f"../data/{args.env_name}/trajectories_(0, 0, 1.0, 1.0)_seed{fidx}_returns.pkl"
            print(f"Reading state_action pairs from {sa_file}")

            sa_data = read_pickle(sa_file)
            r_data = read_pickle(r_file)

            for (b, tidx) in selections:
                state_action_[b].append(sa_data[tidx])
                gt_returns_[b].append(r_data[tidx])

            del sa_data
            del r_data

        if args.ranking_assumption == "bin_ranking" and args.noisy_bins:
            num_trajectories_to_shuffle = int(args.noise_factor * args.num_trajectories)
            print(f"Shuffling {num_trajectories_to_shuffle} trajectories across bins for noise")

            for _ in range(num_trajectories_to_shuffle):
                src_bin = np.random.randint(len(state_action_))

                if src_bin == 0:
                    dst_bin = 1
                elif src_bin == len(state_action_) - 1:
                    dst_bin = src_bin - 1
                else:
                    dst_bin = src_bin + 1 if np.random.rand() < 0.5 else src_bin - 1

                src_idx = np.random.randint(len(state_action_[src_bin]))
                dst_idx = np.random.randint(len(state_action_[dst_bin]))

                state_action_[src_bin][src_idx], state_action_[dst_bin][dst_idx] = (
                    state_action_[dst_bin][dst_idx],
                    state_action_[src_bin][src_idx],
                )

                gt_returns_[src_bin][src_idx], gt_returns_[dst_bin][dst_idx] = (
                    gt_returns_[dst_bin][dst_idx],
                    gt_returns_[src_bin][src_idx],
                )

        state_action = state_action_
        gt_returns = gt_returns_

# Convert to torch tensors
if args.ranking_assumption == "full_ranking":
    state_action = [torch.tensor(state_action, dtype=torch.float32) for state_action in state_action]
    gt_returns = torch.tensor(gt_returns, dtype=torch.float32)
elif args.ranking_assumption == "bin_ranking":
    for i in range(len(state_action)):
        state_action[i] = [torch.tensor(state_action, dtype=torch.float32) for state_action in state_action[i]]
    for i in range(len(gt_returns)):
        gt_returns[i] = torch.tensor(gt_returns[i], dtype=torch.float32)
else:
    raise NotImplementedError("Ranking assumption not implemented")

model = create_model("ga", args.model, args.input_size, 1)

batch_size = args.batch_size
if batch_size > args.num_trajectories:
    batch_size = args.num_trajectories
if args.ranking_assumption == "bin_ranking":
    if batch_size * (len(args.bins) - 1) > args.num_trajectories:
        batch_size = args.num_trajectories

print(f"num trajectories: {args.num_trajectories}, batch size: {batch_size}")
print(f"Number of iterations: {args.num_generations}")

optimizer = NNOptimization(
    model=model,
    data_state_action=state_action,
    data_gt_returns=gt_returns,
    loss_func=args.loss_func,
    batch_size=batch_size,
    lr=3e-4,
    optimizer="adam",
    ranking_assumption=args.ranking_assumption,
    state_range=env_config["state_range"],
    action_dim=env_config["action_dim"],
    continuous_action_space=env_config["continuous_action_space"],
    action_range=env_config["action_range"],
    regularize_ood=env_config["regularization"],
    args=args,
)
model, solution_fitness = optimizer.run()

write_pickle(f"{checkpoint_dir}/{run_guid}_{solution_fitness}_state_action.pkl", state_action)
write_pickle(f"{checkpoint_dir}/{run_guid}_{solution_fitness}_gt_returns.pkl", gt_returns)

config = {
    "model": args.model,
    "input_size": args.input_size,
    "num_generations": args.num_generations,
    "num_trajectories": sum(args.t_per_bin) if args.sample_using_bins and args.human_data else args.num_trajectories,
    "num_files_to_read": args.num_files_to_read,
    "traj_file_to_read": args.traj_file_to_read,
    "run_guid": run_guid,
    "solution_fitness": solution_fitness,
    "bin_distribution": args.bin_distribution,
    "imbalanced_bin_factor": args.imbalanced_bin_factor,
    "noisy_bins": args.noisy_bins,
    "noise_factor": args.noise_factor,
    "num_bins": len(args.bins) - 1 if args.sample_using_bins else None,
    "env_name": args.env_name,
    "rmse_regularization_strength": args.rmse_regularization_strength,
}
write_pickle(f"{checkpoint_dir}/{run_guid}_{solution_fitness}_config.pkl", config)

if args.test_type != "none":
    reward_fn = get_reward_functionNN(model, args.env_name)

    if args.env_name == "hungrythirsty":
        def fitness_fn(state, action, new_state):
            return int(new_state["hungry"] is False)
    else:
        fitness_fn = None
    
    sac_test = SAC_Test(
        reward_fn,
        fitness_fn,
        env_seeds_to_test=args.test_seeds,
        num_episodes=args.num_test_episodes,
        learning_start=100,
        record_freq=10,
        learning_performance_dir=checkpoint_dir,
        run_guid=run_guid,
        try_num=0,
        env_name=env_config["env_name"],
        config=config,
    )
    sac_test.test()
