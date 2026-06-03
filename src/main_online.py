"""Entry point for online training with preference-based reward updating."""

import argparse
import json
import os
import uuid
from copy import deepcopy
from datetime import datetime

import torch

from rewardFunctions.reward_function_online import RewardFunction
from train_online import Trainer
from utils import create_model, read_online_env_config, write_pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=1000000, help="Total environment steps for online training")
    parser.add_argument("--record_freq", type=int, default=10, help="Episode frequency to record metrics")

    parser.add_argument(
        "--reward_update_every",
        type=int,
        default=[10000, 10000, 20000, 30000, 30000, 30000, 40000, 40000, 40000, 50000],
        help="Steps between reward model updates",
    )
    parser.add_argument("--num_generations", type=int, default=2000, help="Gradient steps per reward update")
    parser.add_argument("--num_preferences", type=int, default=10, help="Number of trajectories to label per batch")
    parser.add_argument("--budget", type=int, default=100, help="Total preference budget")

    parser.add_argument(
        "--bins",
        type=float,
        nargs="+",
        default={
            "start": [0, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 500, 600, 800, 1000],
            "end": [0, 30, 60, 100, 200, 300, 400, 500, 600, 800, 1000],
        },
        help="Return bin boundaries (monotonic)",
    )
    parser.add_argument("--subtrajectory_length", type=int, default=50, help="Length of each subtrajectory")

    parser.add_argument("--checkpoint_root", type=str, default="../checkpoints/online", help="Root dir for runs")
    parser.add_argument("--run_name", type=str, default=None, help="Optional manual run name")

    parser.add_argument(
        "--use_identity_reward",
        action="store_true",
        help="If set, use env fitness directly as reward",
    )
    parser.add_argument(
        "--reward_model_arch",
        type=str,
        default="large",
        help="Architecture for learned reward model",
    )
    parser.add_argument(
        "--reward_l2_lambda",
        type=float,
        default=0.01,
        help="L2 regularization lambda for reward model",
    )

    parser.add_argument("--env_name", type=str, default="walker_walk", help="Environment identifier")
    parser.add_argument("--task_name", type=str, default="walk", help="Task name within the environment")
    parser.add_argument("--unsupervised_steps", type=int, default=0, help="Steps for unsupervised pretraining")

    args = parser.parse_args()
    if args.run_name is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"run_{timestamp}"

    online_env_config = read_online_env_config(args.env_name)
    for key, value in online_env_config.items():
        setattr(args, key, value)
    args.bins_ = deepcopy(args.bins)
    return args


class IdentityReward:
    def __call__(self, s, a, ns):
        return 0.0

    def update(self):
        return None


def main():
    args = parse_args()

    os.makedirs(args.checkpoint_root, exist_ok=True)
    run_id = str(uuid.uuid4()).split("-")[0]
    run_dir = os.path.join(args.checkpoint_root, f"{args.run_name}_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    if args.use_identity_reward:
        reward_fn = None
    else:
        reward_fn = "placeholder"

    trainer = Trainer(
        reward_fn=None if reward_fn == "placeholder" else reward_fn,
        pref_freq=args.reward_update_every,
        batch_sz_reward=args.num_preferences,
        budget=args.budget,
        args=args,
        intrinsic_reward_fn="placeholder",
    )

    obs_dim = trainer.env.observation_space.shape[0]
    act_dim = trainer.env.action_space.shape[0]
    model = create_model("ga", args.reward_model_arch, obs_dim + act_dim, 1)
    reward_fn_obj = RewardFunction(model)

    class RewardWrapper:
        def __init__(self, rf_obj):
            self.rf_obj = rf_obj

        def __call__(self, s, a, ns):
            return self.rf_obj.reward_fn(s, a, ns)

        def update(self, *args, **kwargs):
            return self.rf_obj.update(*args, **kwargs)

        def update_average_reward(self, new_reward):
            self.rf_obj.average_reward = new_reward

        def get_average_reward(self):
            return self.rf_obj.average_reward

    wrapper = RewardWrapper(reward_fn_obj)
    trainer.reward_fn = wrapper

    learning_performance = trainer.train()

    config = {
        "max_steps": args.max_steps,
        "record_freq": args.record_freq,
        "reward_update_every": args.reward_update_every,
        "reward_update_steps": args.num_generations,
        "num_preferences": args.num_preferences,
        "budget": args.budget,
        "bins": args.bins_,
        "run_name": args.run_name,
        "run_id": run_id,
        "env_name": args.env_name,
        "use_identity_reward": args.use_identity_reward,
        "reward_model_arch": args.reward_model_arch,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "reward_l2_lambda": args.reward_l2_lambda,
        "subtrajectory_length": args.subtrajectory_length,
    }

    write_pickle(os.path.join(run_dir, "results.pkl"), learning_performance)
    write_pickle(os.path.join(run_dir, "config.pkl"), config)

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "run_name": args.run_name,
                "run_id": run_id,
                "total_feedback": trainer.total_feedback,
                "num_pref_bins": len(trainer.preferences),
                "num_records": len(learning_performance),
            },
            f,
            indent=2,
        )

    print(f"Run complete. Artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
