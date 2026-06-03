import numpy as np
import torch


class RewardFunction:
    def __init__(self):
        pass

    def reward_fn(self, state, action, new_state):
        raise NotImplementedError


class RewardFunctionNN_hungry_thirsty(RewardFunction):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def reward_fn(self, state, action, new_state):
        sa = torch.tensor(
            [state["position"][0], state["position"][1], state["hungry"], state["thirsty"], action],
            dtype=torch.float32,
        )
        sa = sa.unsqueeze(0)
        reward = self.model(sa)
        return reward.item()


class RewardFunctionNN_lunarlander(RewardFunction):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def reward_fn(self, state, action, new_state):
        sa = torch.tensor([*state, action], dtype=torch.float32)
        sa = sa.unsqueeze(0)
        reward = self.model(sa)
        return reward.item()


class RewardFunctionContinuousActions(RewardFunction):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def reward_fn(self, state, action, new_state):
        sa = torch.tensor([*state, *action], dtype=torch.float32)
        sa = sa.unsqueeze(0)
        reward = self.model(sa)
        return reward.item()


class RewardFunctionSimple(RewardFunction):
    def __init__(self, weights):
        super().__init__()
        self.weights = np.array(weights)

    def reward_fn(self, state, action, new_state):
        hungry = state["hungry"]
        thirsty = state["thirsty"]

        hungry_thirsty = int(hungry and thirsty)
        hungry_not_thirsty = int(hungry and not thirsty)
        not_hungry_thirsty = int(not hungry and thirsty)
        not_hungry_not_thirsty = int(not hungry and not thirsty)

        state = np.array([hungry_thirsty, hungry_not_thirsty, not_hungry_thirsty, not_hungry_not_thirsty]).T
        reward = np.dot(self.weights, state)

        return reward


def get_reward_functionNN(model, env_name):
    if env_name == "lunarlander":
        return RewardFunctionNN_lunarlander(model).reward_fn
    elif env_name == "hungrythirsty":
        return RewardFunctionNN_hungry_thirsty(model).reward_fn
    elif env_name in ["reacher", "inverteddoublependulum", "invertedpendulum", "halfcheetah", "ant", "hopper"]:
        return RewardFunctionContinuousActions(model).reward_fn
    else:
        raise ValueError(f"Unknown environment: {env_name}")
