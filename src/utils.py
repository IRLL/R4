import json
import pickle
import sys

import torch

from models import large, medium, mini


def create_model(search_type, model, input_size, output_size):
    if search_type == "ga":
        if model == "mini":
            return mini.Mini(input_size, output_size)
        elif model == "medium":
            return medium.Medium(input_size, output_size)
        elif model == "medium_s":
            return medium.MediumStochastic(input_size, output_size)
        elif model == "large":
            return large.Large(input_size, output_size)
        elif model == "large2":
            return large.Large2(input_size, output_size)
        else:
            print("Model not recognized")
            sys.exit(1)
    elif search_type == "mh":
        if model == "mini":
            return mini.MiniMPH(input_size, output_size)
        else:
            print("Model not recognized")
            sys.exit(1)
    else:
        print("Search type not recognized")
        sys.exit(1)


def read_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def write_pickle(file_path, data):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def read_bin_data(env_name, bin_type="default", args=None, num_bins=None):
    with open("configs/bins.json", "r") as f:
        bins = json.load(f)
    if env_name not in bins:
        print(f"Environment {env_name} not found in bins.json")
        sys.exit(1)
    if bin_type == "default":
        bins = bins[env_name]["default"]
    elif bin_type == "equal":
        bins_min = bins[env_name]["min"]
        bins_max = bins[env_name]["max"]
        num_bins = args.num_bins if num_bins is None else num_bins
        bin_size = (bins_max - bins_min) / num_bins
        bins = [bins_min + i * bin_size for i in range(num_bins + 1)]
    elif bin_type == "human":
        num_bins = args.num_bins
        bins = [0] * (num_bins + 1)

    if args.bin_distribution == "uniform":
        trajs_per_bin = args.num_trajectories // (len(bins) - 1)
        remaining_trajs = args.num_trajectories % (len(bins) - 1)
        t_per_bin = [trajs_per_bin for _ in range(len(bins) - 1)]
        for i in range(remaining_trajs):
            t_per_bin[-(i + 1)] += 1
    elif args.bin_distribution == "imbalanced":
        trajs_unbalanced_bin = int(args.num_trajectories * args.imbalanced_bin_factor)
        trajs_per_bin = (args.num_trajectories - trajs_unbalanced_bin) // (len(bins) - 2)
        remaining_trajs = (args.num_trajectories - trajs_unbalanced_bin) % (len(bins) - 2)
        t_per_bin = [trajs_per_bin for _ in range(len(bins) - 2)]
        for i in range(remaining_trajs):
            t_per_bin[-(i + 1)] += 1
        t_per_bin = [trajs_unbalanced_bin] + t_per_bin
    return bins, t_per_bin


def read_online_env_config(env_name):
    with open("configs/online_envs.json", "r") as f:
        envs = json.load(f)
    if env_name not in envs:
        print(f"Environment {env_name} not found in online_envs.json")
        sys.exit(1)
    return envs[env_name]


def read_env_config(env_name):
    with open("configs/envs.json", "r") as f:
        envs = json.load(f)
    if env_name not in envs:
        print(f"Environment {env_name} not found in envs.json")
        sys.exit(1)

    config = envs[env_name]

    if len(config["state_low"]) == 1:
        config["state_low"] = [config["state_low"][0]] * (config["input_dim"] - config["action_dim"])
        config["state_high"] = [config["state_high"][0]] * (config["input_dim"] - config["action_dim"])

    config["state_range"] = (
        torch.tensor(config["state_low"], dtype=torch.float32),
        torch.tensor(config["state_high"], dtype=torch.float32),
    )
    config["action_range"] = (
        (torch.tensor(config["action_low"], dtype=torch.float32), torch.tensor(config["action_high"], dtype=torch.float32))
        if config["continuous_action_space"]
        else None
    )

    return config
