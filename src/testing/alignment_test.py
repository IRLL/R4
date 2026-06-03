import torch

from fitnessFunctions import kendalltau
from utils import read_pickle


class AlignmentTest:
    def __init__(self, reward_fn, env_seeds_to_test, env_name):
        self.reward_fn = reward_fn
        self.env_seeds_to_test = env_seeds_to_test
        self.env_name = env_name

        self.state_action = []
        self.gt_returns = []
        for seed in self.env_seeds_to_test:
            self.state_action += read_pickle(
                f"../data/{self.env_name}/trajectories_(0, 0, 1.0, 1.0)_seed{seed}_state_action_pairs.pkl"
            )
            self.gt_returns += read_pickle(
                f"../data/{self.env_name}/trajectories_(0, 0, 1.0, 1.0)_seed{seed}_returns.pkl"
            )

    def test(self):
        output_returns = []
        for trajectory in self.state_action:
            trajectory = torch.tensor(trajectory, dtype=torch.float32)
            output_rewards = self.reward_fn(trajectory)
            output_returns.append(output_rewards.sum().item())

        kt_corr = kendalltau.kendalltau_distance(self.gt_returns, output_returns)
        return kt_corr
