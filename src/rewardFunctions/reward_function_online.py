import numpy as np
import torch
from copy import deepcopy

from optimization import NNOptimization


class RewardFunction:
    def __init__(self, model, ensamble_number=3):
        self.ensamble_number = ensamble_number
        self.ensamble = []
        for _ in range(ensamble_number):
            model_ = deepcopy(model)
            for layer in model_.children():
                layer.reset_parameters()
            self.ensamble.append(deepcopy(model_))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ensamble = [model_.to(self.device) for model_ in self.ensamble]
        self.average_reward = 0

    def reward_fn(self, state, action, new_state=None):
        with torch.no_grad():
            sa = torch.tensor([*state, *action], dtype=torch.float32).to(self.device)
            sa = sa.unsqueeze(0)
            reward = 0
            for model_ in self.ensamble:
                reward += model_(sa).cpu()
            reward /= self.ensamble_number
            return reward.item() - self.average_reward

    def update(self, preferences, batch_size=64, args=None):
        state_action_, gt_returns_ = self.create_data_from_preferences(preferences)

        ensamble_ = []
        for model_ in self.ensamble:
            trainer = NNOptimization(
                model=model_.to("cpu"),
                data_state_action=state_action_,
                data_gt_returns=gt_returns_,
                loss_func="ranking_mse",
                batch_size=batch_size,
                lr=3e-4,
                optimizer="adam",
                ranking_assumption="bin_ranking",
                args=args,
                regularize_ood=False,
                l2_lambda=args.reward_l2_lambda,
            )
            model_, _ = trainer.run()
            model_.to(self.device)
            ensamble_.append(model_)
            self.average_reward = 0
        self.ensamble = ensamble_

    def create_data_from_preferences(self, preferences):
        state_action_ = []
        gt_returns_ = []
        sorted_bins_indices = sorted(preferences.keys())
        trajs_per_bin = []
        for idx in sorted_bins_indices:
            state_action_.append(deepcopy(preferences[idx]["state_action"]))
            gt_returns_.append(deepcopy(preferences[idx]["gt_returns"]))
            trajs_per_bin.append(len(preferences[idx]["state_action"]))

        max_trajs = max(trajs_per_bin)
        for i in range(len(state_action_)):
            if len(state_action_[i]) < max_trajs:
                indices = np.random.choice(len(state_action_[i]), size=(max_trajs - len(state_action_[i])), replace=True)
                for idx in indices:
                    state_action_[i].append(deepcopy(preferences[sorted_bins_indices[i]]["state_action"][idx]))
                    gt_returns_[i].append(deepcopy(preferences[sorted_bins_indices[i]]["gt_returns"][idx]))

        for i in range(len(state_action_)):
            state_action_[i] = [torch.tensor(state_action, dtype=torch.float32) for state_action in state_action_[i]]
            gt_returns_[i] = torch.tensor(gt_returns_[i], dtype=torch.float32)

        return state_action_, gt_returns_
