from collections import deque
from copy import deepcopy

import numpy as np
import torch
from dm_control import suite
from gymnasium.wrappers import FlattenObservation
import shimmy
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from tqdm import tqdm


class Trainer:
    def __init__(self, reward_fn, pref_freq, batch_sz_reward, budget, intrinsic_reward_fn, args):
        self.reward_fn = reward_fn
        self.intrinsic_reward_fn = intrinsic_reward_fn
        self.pref_freq = pref_freq
        self.batch_sz_reward = batch_sz_reward
        self.budget = budget
        self.args = args

        env_dm = suite.load(args.env_name, args.task_name)
        self.env = FlattenObservation(shimmy.DmControlCompatibilityV0(env_dm))

        self.agent = SAC("MlpPolicy", self.env)
        tmp_path = "/tmp/sb3_log/"
        new_logger = configure(tmp_path, ["stdout", "csv"])
        self.agent.set_logger(new_logger)

        self.replay_buffer = self.agent.replay_buffer

        self.reward_update_every = args.reward_update_every
        self.num_preferences = args.num_preferences
        self.max_update_steps = args.num_generations

        self.total_feedback = 0
        self.steps = 0
        self.episode = 0

        self.preferences = {}
        self.bin_structures = args.bins

    def train(self):
        progress_bar = tqdm(range(self.args.max_steps))
        running_avg_fitness = deque(maxlen=50)
        running_avg_reward = deque(maxlen=50)
        state, _ = self.env.reset()
        self.learning_performance = []
        self.trajectories = deque(maxlen=50)
        self.returns = deque(maxlen=50)
        self.fitnesses = deque(maxlen=50)
        self.predicted_rewards = []
        trajectory_episode = []
        fitness_episode = []
        total_reward = 0.0
        total_fitness = 0.0

        done = False
        episode_step = 0
        reward_update_idx = 0
        next_reward_update = self.args.reward_update_every[0]
        redistributed = False
        while True:
            with torch.no_grad():
                action, _ = self.agent.predict(state, deterministic=False)

            next_state, fitness, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if self.steps < self.args.unsupervised_steps:
                reward = self.intrinsic_reward_fn(state, action, next_state)
            else:
                if self.reward_fn is not None:
                    reward = self.reward_fn(state, action, next_state)
                else:
                    reward = fitness

            self.replay_buffer.add(state, next_state, action, reward, done, [info])

            trajectory_episode.append([*state, *action])
            fitness_episode.append(fitness)

            total_reward += reward
            total_fitness += fitness

            self.agent.num_timesteps += 1
            if self.agent.num_timesteps >= 0 and self.steps % 1 == 0:
                self.agent.train(gradient_steps=1, batch_size=256)

            state = next_state

            self.steps += 1
            episode_step += 1

            if done:
                state, _ = self.env.reset()
                done = False
                running_avg_fitness.append(total_fitness)
                running_avg_reward.append(total_reward)

                if self.total_feedback < self.budget:
                    self.trajectories.append(deepcopy(trajectory_episode))
                    self.returns.append(deepcopy(total_fitness))
                    self.fitnesses.append(deepcopy(fitness_episode))

                if self.episode % self.args.record_freq == 0:
                    scores = [
                        self.episode,
                        {
                            "avg_undiscounted_return": total_reward,
                            "avg_fitness": total_fitness,
                        },
                    ]
                    self.learning_performance.append(scores)
                self.episode += 1

                progress_bar.set_postfix(
                    reward=total_reward,
                    fitness=total_fitness,
                    avg_fitness=np.mean(running_avg_fitness),
                    avg_reward=np.mean(running_avg_reward),
                )
                total_fitness = 0.0
                total_reward = 0.0
                trajectory_episode = []
                fitness_episode = []
                progress_bar.update(episode_step)
                episode_step = 0

            if self.steps == next_reward_update or (
                len(self.preferences) == 0 and len(self.trajectories) >= self.num_preferences
            ):
                if self.total_feedback < 40:
                    self.args.bins = self.bin_structures["start"]
                    redistributed = False
                else:
                    if not redistributed:
                        self.redistribute_preferences()
                        redistributed = True
                    self.args.bins = self.bin_structures["end"]
                if self.total_feedback < self.budget:
                    remaining = self.budget - self.total_feedback
                    to_collect = self.num_preferences if remaining > self.num_preferences else remaining
                    if len(self.trajectories) > 0 and to_collect > 0:
                        collected = self.collect_subtrajectory_preferences(
                            to_collect, subtrajectory_length=self.args.subtrajectory_length
                        )
                        self.total_feedback += collected
                    if hasattr(self.reward_fn, "update"):
                        self.args.num_generations = int(self.max_update_steps * self.total_feedback / self.budget)
                        self.reward_fn.update(self.preferences, batch_size=self.batch_sz_reward, args=self.args)
                        running_avg_fitness.clear()
                        running_avg_reward.clear()
                    self.relabel_with_predictor()
                    reward_update_idx += 1
                    next_reward_update = (
                        self.args.reward_update_every[min(reward_update_idx, len(self.args.reward_update_every) - 1)]
                        + self.steps
                    )
                    print(f"Next reward update at {next_reward_update} steps")

            if self.steps == self.args.max_steps:
                break

        return self.learning_performance

    def collect_subtrajectory_preferences(self, num_preferences, batch_size=2048, subtrajectory_length=50):
        print("collecting subtrajectory preferences")
        bins = self.args.bins
        if len(self.trajectories) == 0:
            return 0

        num_preferences = min(num_preferences, len(self.trajectories))
        trajectory_indices = np.arange(len(self.trajectories))
        self.sort_trajectories_according_to_predicted_returns()
        percentile_indices = self.get_trajectory_percentiles()
        trajectory_indices = np.delete(trajectory_indices, percentile_indices)

        percentile_samples = np.random.choice(
            percentile_indices, size=min(len(percentile_indices), num_preferences // 3), replace=False
        )
        trajectory_samples = np.random.choice(
            trajectory_indices, size=num_preferences - len(percentile_samples), replace=False
        )
        trajectory_samples = np.concatenate([percentile_samples, trajectory_samples])

        for idx in trajectory_samples:
            r = np.random.uniform(0, 1)
            if r < 0.5:
                start = self.max_sum_subarray(self.predicted_rewards[idx], subtrajectory_length)
            else:
                start = np.random.randint(0, len(self.trajectories[idx]) - subtrajectory_length + 1)
            subtrajectory = self.trajectories[idx][start : start + subtrajectory_length]
            subtrajectory_return = sum(self.fitnesses[idx][start : start + subtrajectory_length])

            for i in range(len(bins) - 1):
                if bins[i] <= (subtrajectory_return) * 1000 / subtrajectory_length < bins[i + 1]:
                    if i not in self.preferences:
                        self.preferences[i] = {"state_action": [], "gt_returns": []}
                    self.preferences[i]["state_action"].append(deepcopy(subtrajectory))
                    self.preferences[i]["gt_returns"].append(deepcopy(subtrajectory_return))

        print(f"Collected {[(i, len(self.preferences[i]['state_action'])) for i in self.preferences.keys()]}")

        return num_preferences

    def max_sum_subarray(self, arr, length):
        start_index = 0
        max_sum = float("-inf")
        current_sum = sum(arr[:length])
        for i in range(length, len(arr)):
            current_sum += arr[i] - arr[i - length]
            if current_sum > max_sum:
                max_sum = current_sum
                start_index = i - length + 1
        return start_index

    def collect_preferences(self, num_preferences, batch_size=2048):
        print("collecting preferences")
        bins = self.args.bins
        if len(self.trajectories) == 0:
            return 0
        num_preferences = min(num_preferences, len(self.trajectories))

        trajectory_indices = np.arange(len(self.trajectories))
        self.sort_trajectories_according_to_predicted_returns()
        percentile_indices = self.get_trajectory_percentiles()
        trajectory_indices = np.delete(trajectory_indices, percentile_indices)

        percentile_samples = np.random.choice(
            percentile_indices, size=min(len(percentile_indices), num_preferences // 3), replace=False
        )
        trajectory_samples = np.random.choice(
            trajectory_indices, size=num_preferences - len(percentile_samples), replace=False
        )
        sampled_trajectory_indices = np.concatenate([percentile_samples, trajectory_samples])

        for idx in sampled_trajectory_indices:
            for i in range(len(bins) - 1):
                if bins[i] <= self.returns[idx] < bins[i + 1]:
                    if i not in self.preferences:
                        self.preferences[i] = {"state_action": [], "gt_returns": []}
                    self.preferences[i]["state_action"].append(deepcopy(self.trajectories[idx]))
                    self.preferences[i]["gt_returns"].append(deepcopy(self.returns[idx]))

        self.trajectories = [t for i, t in enumerate(self.trajectories) if i not in sampled_trajectory_indices]
        self.returns = [r for i, r in enumerate(self.returns) if i not in sampled_trajectory_indices]
        maxlen = 50
        self.trajectories = deque(self.trajectories[-maxlen:], maxlen=maxlen)
        self.returns = deque(self.returns[-maxlen:], maxlen=maxlen)
        print(f"Collected {[(i, len(self.preferences[i]['state_action'])) for i in self.preferences.keys()]}")
        return num_preferences

    def relabel_with_predictor(self, batch_size=2048):
        if self.reward_fn is None:
            return
        size = self.replay_buffer.buffer_size if self.replay_buffer.full else self.replay_buffer.pos
        if size == 0:
            return

        obs = self.replay_buffer.observations[:size]
        actions = self.replay_buffer.actions[:size]

        new_rewards = np.empty((size,), dtype=np.float32)

        for start in range(0, size, batch_size):
            end = min(start + batch_size, size)
            o_batch = obs[start:end]
            a_batch = actions[start:end]
            for i, (o, a) in enumerate(zip(o_batch, a_batch)):
                r = self.reward_fn(o.squeeze(), a.squeeze(), None)
                r = r - self.reward_fn.get_average_reward()
                new_rewards[start + i] = float(r)

        if self.replay_buffer.rewards.ndim == 2 and self.replay_buffer.rewards.shape[1] == 1:
            self.replay_buffer.rewards[:size, 0] = new_rewards
        else:
            self.replay_buffer.rewards[:size] = new_rewards

    def redistribute_preferences(self):
        merge_start_idx = 0
        merge_end_idx = 0
        new_preferences = {}
        for i in range(len(self.bin_structures["end"]) - 1):
            bin_start = self.bin_structures["end"][i]
            bin_end = self.bin_structures["end"][i + 1]
            while self.bin_structures["start"][merge_end_idx + 1] < bin_end:
                merge_end_idx += 1

            for j in range(merge_start_idx, merge_end_idx + 1):
                if j in self.preferences:
                    if i not in new_preferences:
                        new_preferences[i] = {"state_action": [], "gt_returns": []}
                    new_preferences[i]["state_action"].extend(self.preferences[j]["state_action"])
                    new_preferences[i]["gt_returns"].extend(self.preferences[j]["gt_returns"])

            merge_start_idx = merge_end_idx + 1
            merge_end_idx = merge_start_idx

        self.preferences = new_preferences

    def sort_trajectories_according_to_predicted_returns(self):
        if self.reward_fn is None:
            return
        traj_returns = []
        self.predicted_rewards = []
        for traj in self.trajectories:
            trajectory_rewards = []
            sa_pairs = np.array(traj)
            total_predicted_return = 0.0
            for sa in sa_pairs:
                state = sa[: self.env.observation_space.shape[0]]
                action = sa[self.env.observation_space.shape[0] :]
                try:
                    r = self.reward_fn(state, action, None)
                except TypeError:
                    sa_vec = np.concatenate([state, action], dtype=np.float32)
                    r = self.reward_fn(sa_vec)
                trajectory_rewards.append(float(r))
                total_predicted_return += float(r)
            traj_returns.append(total_predicted_return)
            self.predicted_rewards.append(trajectory_rewards)
        sorted_indices = np.argsort(traj_returns)[::-1]
        self.trajectories = [self.trajectories[i] for i in sorted_indices]
        self.returns = [self.returns[i] for i in sorted_indices]
        self.fitnesses = [self.fitnesses[i] for i in sorted_indices]
        self.predicted_rewards = [self.predicted_rewards[i] for i in sorted_indices]

    def get_trajectory_percentiles(self):
        if not self.trajectories:
            return None, None

        p95_index = int(0.70 * len(self.trajectories))
        percentile_indices = np.arange(p95_index, len(self.trajectories))

        return percentile_indices
