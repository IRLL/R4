import torch
from lossFunctions.rMSE_ot import ranking_mse_ot
from lossFunctions.ranking_mse import ranking_mse
from tqdm import tqdm
from datasets.rMSE_dataset import BinRankingDataset


class NNOptimization:
    def __init__(
        self,
        model,
        data_state_action,
        data_gt_returns,
        loss_func,
        batch_size,
        lr,
        optimizer,
        ranking_assumption,
        state_range=None,
        action_dim=None,
        ood_batch_size=1024,
        ood_regularization_coeff=0.1,
        regularize_ood=True,
        continuous_action_space=False,
        action_range=None,
        args=None,
        l2_lambda=0,
        reward_l2_lambda=None,
    ):
        self.model = model
        self.ranking_assumption = ranking_assumption
        self.regularize_ood = regularize_ood
        self.state_range = state_range
        self.action_dim = action_dim
        self.continuous_action_space = continuous_action_space
        if self.continuous_action_space and action_range is None:
            raise ValueError("Action range must be provided for continuous action space")
        elif self.continuous_action_space:
            self.action_range = action_range
        self.ood_batch_size = ood_batch_size
        self.ood_regularization_coeff = ood_regularization_coeff
        self.args = args
        self.l2_lambda = l2_lambda if reward_l2_lambda is None else reward_l2_lambda

        if ranking_assumption == "full_ranking":
            self.data_state_action = torch.stack(data_state_action)
            self.data_gt_returns = data_gt_returns
        elif ranking_assumption == "bin_ranking":
            max_traj_length = max([len(traj) for bin in data_state_action for traj in bin])
            for bin in range(len(data_state_action)):
                for traj in range(len(data_state_action[bin])):
                    if len(data_state_action[bin][traj]) < max_traj_length:
                        padding = torch.full(
                            (max_traj_length - len(data_state_action[bin][traj]), data_state_action[bin][traj].shape[1]),
                            float("nan"),
                        )
                        data_state_action[bin][traj] = torch.cat((data_state_action[bin][traj], padding), dim=0)
            self.data_state_action = []
            for i in range(len(data_state_action)):
                self.data_state_action.append(torch.stack(data_state_action[i]))
        else:
            raise NotImplementedError("Ranking assumption not implemented")

        if loss_func == "ranking_mse":
            self.loss_func = lambda pred, gt: ranking_mse(
                pred, gt, regularization_strength=self.args.rmse_regularization_strength
            )
        elif loss_func == "ranking_mse_ot":
            self.loss_func = ranking_mse_ot
        else:
            raise NotImplementedError("Loss function not implemented")

        self.batch_size = batch_size
        self.lr = lr
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.device = torch.device("cpu")
        self.model.to(self.device)

    def run(self):
        if self.ranking_assumption == "full_ranking":
            dataset = torch.utils.data.TensorDataset(self.data_state_action, self.data_gt_returns)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        elif self.ranking_assumption == "bin_ranking":
            dataset = BinRankingDataset(self.data_state_action)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        else:
            raise NotImplementedError("Ranking assumption not implemented")

        self.model.train()
        progress_bar = tqdm(range(self.args.num_generations), desc="Iterations")
        iteration = 0
        while True:
            if self.ranking_assumption == "bin_ranking":
                dataset.shuffle_bins()

            for _, (state_action, gt_returns) in enumerate(dataloader):
                iteration += 1
                progress_bar.update(1)

                if self.ranking_assumption == "bin_ranking":
                    gt_returns_shape = gt_returns.shape
                    if torch.isnan(state_action).sum() == 0:
                        has_nans = False
                    else:
                        mask = torch.logical_not(torch.isnan(state_action).any(dim=3).unsqueeze(3)).float()
                        state_action = torch.nan_to_num(state_action, nan=0.0)
                        mask = mask.view(-1, mask.shape[2], mask.shape[3])
                        has_nans = True
                    state_action = state_action.view(-1, state_action.shape[2], state_action.shape[3])
                elif self.ranking_assumption == "full_ranking":
                    gt_returns = gt_returns.unsqueeze(1)

                state_action = state_action.to(self.device)
                if self.regularize_ood:
                    state_ood = (self.state_range[0] - self.state_range[1]) * torch.rand(
                        self.ood_batch_size, self.state_range[0].shape[0]
                    ) + self.state_range[1]
                    if self.continuous_action_space:
                        action_ood = (self.action_range[0] - self.action_range[1]) * torch.rand(
                            self.ood_batch_size, self.action_dim
                        ) + self.action_range[1]
                        s_a_ood = torch.cat((state_ood, action_ood), dim=1)
                    else:
                        action_ood = torch.randint(0, self.action_dim, (self.ood_batch_size,))
                        s_a_ood = torch.cat((state_ood, action_ood.unsqueeze(1)), dim=1)
                gt_returns = gt_returns.to(self.device)

                self.optimizer.zero_grad()

                output = self.model(state_action)
                if has_nans:
                    output = output * mask
                loss = 0
                if self.regularize_ood:
                    loss = -self.ood_regularization_coeff * torch.mean(output)
                output = torch.sum(output, dim=1)

                if self.ranking_assumption == "bin_ranking":
                    output = output.view(gt_returns_shape[0], gt_returns_shape[1])
                    output = output.T
                    gt_returns = gt_returns.T

                loss += self.loss_func(output, gt_returns)
                if self.regularize_ood:
                    output_ood = self.model(s_a_ood)
                    loss += self.ood_regularization_coeff * torch.mean(output_ood)

                if self.l2_lambda > 0:
                    l2_reg = 0
                    for param in self.model.parameters():
                        l2_reg += torch.norm(param)
                    loss += self.l2_lambda * l2_reg

                loss.backward()
                self.optimizer.step()
                progress_bar.set_postfix(loss=loss.item())

                if iteration >= self.args.num_generations:
                    break

            if iteration >= self.args.num_generations:
                break
        progress_bar.close()

        return self.model.to("cpu"), loss.item()
