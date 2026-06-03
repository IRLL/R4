from collections import deque

import numpy as np
import tqdm
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure


class SACAgent:
    def __init__(self, env, hyperparameters, reward_fn=None, fitness_fn=None):
        self.env = env
        self.hyperparameters = hyperparameters
        self.reward_fn = reward_fn
        self.fitness_fn = fitness_fn

        self.model = SAC("MlpPolicy", env)
        tmp_path = "/tmp/sb3_log/"
        new_logger = configure(tmp_path, ["stdout", "csv"])
        self.model.set_logger(new_logger)

        self.buffer = self.model.replay_buffer
        self.learning_performance = []

    def train(self):
        progress_bar = tqdm.tqdm(range(self.hyperparameters["num_episodes"]))
        running_avg_fitness = deque(maxlen=50)
        running_avg_reward = deque(maxlen=50)
        max_fitness = float("-inf")

        for episode in progress_bar:
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            total_fitness = 0

            while not done:
                action, _ = self.model.predict(state, deterministic=False)
                next_state, fitness, terminated, truncated, info = self.env.step(action)

                if self.reward_fn is not None:
                    reward = self.reward_fn(state, action, next_state)
                else:
                    reward = fitness

                if self.fitness_fn is not None:
                    fitness = self.fitness_fn(state, action, next_state)

                done = terminated or truncated
                total_reward += reward
                total_fitness += fitness

                scaled_action = self.model.policy.scale_action(action)
                self.buffer.add(state, next_state, scaled_action, reward, done, [info])

                self.model.num_timesteps += 1
                if self.model.num_timesteps >= self.hyperparameters["learning_start"]:
                    self.model.train(gradient_steps=1, batch_size=self.model.batch_size)

                state = next_state

            running_avg_fitness.append(total_fitness)
            running_avg_reward.append(total_reward)
            avg_fitness = np.mean(running_avg_fitness)
            avg_reward = np.mean(running_avg_reward)
            max_fitness = max(max_fitness, avg_fitness)

            progress_bar.set_postfix(
                avg_fitness=avg_fitness,
                avg_reward=avg_reward,
                max_fitness=max_fitness,
            )

            if episode % self.hyperparameters["record_freq"] == 0:
                scores = [
                    episode,
                    {
                        "avg_undiscounted_return": total_reward,
                        "avg_fitness": total_fitness,
                    },
                ]
                self.learning_performance.append(scores)
        return self.learning_performance
