from torch.utils.data import Dataset
import torch


class BinRankingDataset(Dataset):
    def __init__(self, bins):
        self.bins = bins
        self.num_bins = len(bins)
        self.shuffle_bins()
        self.max_len_trajectory = max([len(traj) for bin in bins for traj in bin])

    def shuffle_bins(self):
        for i in range(self.num_bins):
            self.bins[i] = self.bins[i][torch.randperm(len(self.bins[i]))]

    def __len__(self):
        return min([len(bin) for bin in self.bins])

    def __getitem__(self, idx):
        sampled_trajectories = []
        bin_indices = []

        for bin_idx, bin in enumerate(self.bins):
            trajectory = bin[idx % len(bin)]
            sampled_trajectories.append(trajectory)
            bin_indices.append(bin_idx)

        sampled_trajectories = torch.stack(sampled_trajectories)
        bin_indices = torch.tensor(bin_indices, dtype=torch.float32)

        return sampled_trajectories, bin_indices
