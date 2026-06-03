import random
import time
from collections import deque

import gym
import numpy as np
from tqdm import tqdm


class QLearning:
    """
    Q-Learning algorithm
    """

    def __init__(self, env, num_episodes, reward_fn=None, epsilon=0.15, fitness_fn=None):
        self.env = env
        self.num_episodes = num_episodes
        self.reward_fn = reward_fn
        self.epsilon = epsilon
        self.fitness_fn = fitness_fn
        self.learning_performance = []

        if "hungry-thirsty" in self.env.spec.id:
            self.Q_table = self.env.construct_q_table()
        else:
            if type(self.env.observation_space) is gym.spaces.discrete.Discrete:
                self.Q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])
            else:
                raise Exception("No target for constructing a Q table")
        self.count = 0

    def q_table_argmax(self, state):
        if type(self.Q_table) is np.ndarray:
            return np.argmax(self.Q_table[state, :])
        else:
            action_vals_dict = self.Q_table[self.env.hash_lookup(state)]
            action = max(action_vals_dict, key=action_vals_dict.get)
            return action

    def q_lookup(self, state, action):
        if type(self.Q_table) is np.ndarray:
            return self.Q_table[state, action]
        else:
            return self.Q_table[self.env.hash_lookup(state)][action]

    def q_update(self, state, action, update_value, alpha_lr):
        if type(self.Q_table) is np.ndarray:
            self.Q_table[state, action] = (1 - alpha_lr) * self.Q_table[state, action] + update_value
        else:
            self.Q_table[self.env.hash_lookup(state)][action] = (1 - alpha_lr) * self.Q_table[
                self.env.hash_lookup(state)
            ][action] + update_value

    def e_greedy_action_selection(self, state, epsilon):
        if random.random() < epsilon:
            action = self.env.get_random_action()
        else:
            action = self.q_table_argmax(state)
        new_state, reward, done, info = self.env.step(action)

        if self.reward_fn is not None:
            reward = self.reward_fn(state=state, action=action, new_state=new_state)

        if self.fitness_fn is not None:
            fitness = self.fitness_fn(state=state, action=action, new_state=new_state)

        info = {}
        info["fitness"] = fitness

        return action, new_state, reward, done, info

    def score_policy(self, gamma, epsilon=0, new_water_food_loc=False, render=False):
        state = self.env.reset(new_water_food_loc=new_water_food_loc)

        episode_sum_rewards = 0
        episode_return = 0
        episode_fitness = 0
        j = 0
        done = False
        traj = []

        while not done:
            if render:
                self.env.render()
                time.sleep(0.1)

            action, new_state, reward, done, info = self.e_greedy_action_selection(state=state, epsilon=epsilon)
            episode_fitness += info["fitness"]
            traj.append((state, action))
            state = new_state
            episode_sum_rewards += reward
            episode_return += (gamma**j) * reward
            j += 1
        self.count += 1

        return episode_sum_rewards, episode_return, episode_fitness, traj

    def record_performance(self, episode, gamma, epsilon, num_tests, new_water_food_loc=False):
        scores = [
            episode,
            {
                "all_discounted_return": [],
                "all_undiscounted_return": [],
                "all_fitness": [],
                "avg_discounted_return": [],
                "avg_undiscounted_return": [],
                "avg_fitness": [],
                "all_trajectory": [],
            },
        ]
        avg_episode_sum_rewards = []
        avg_episode_return = []
        avg_episode_fitness = []

        for _ in range(num_tests):
            episode_sum_rewards, episode_return, episode_fitness, traj = self.score_policy(
                gamma=gamma, epsilon=epsilon, new_water_food_loc=new_water_food_loc
            )
            avg_episode_sum_rewards.append(episode_sum_rewards)
            avg_episode_return.append(episode_return)
            avg_episode_fitness.append(episode_fitness)

            if num_tests == 1:
                scores[1]["all_trajectory"] = traj
                scores[1]["all_discounted_return"] = episode_return
                scores[1]["all_undiscounted_return"] = episode_sum_rewards
                scores[1]["all_fitness"] = episode_fitness
            else:
                scores[1]["all_trajectory"].append(traj)
                scores[1]["all_discounted_return"].append(episode_return)
                scores[1]["all_undiscounted_return"].append(episode_sum_rewards)
                scores[1]["all_fitness"].append(episode_fitness)

        scores[1]["avg_discounted_return"] = np.mean(avg_episode_return)
        scores[1]["avg_undiscounted_return"] = np.mean(avg_episode_sum_rewards)
        scores[1]["avg_fitness"] = np.mean(avg_episode_fitness)

        self.learning_performance.append(scores)

    def learn_1_episode(self, alpha_lr, gamma, epsilon, new_water_food_loc=False, record=True):
        state = self.env.reset(new_water_food_loc=new_water_food_loc)

        done = False
        fitness = 0
        step = 0
        episode_fitness = []

        while not done:
            step += 1
            if not state["hungry"]:
                fitness += 1

            action, new_state, reward, done, info = self.e_greedy_action_selection(state=state, epsilon=epsilon)

            best_next_action = self.q_table_argmax(new_state)
            new_state_q_val = self.q_lookup(state=new_state, action=best_next_action)

            update_value = alpha_lr * (reward + gamma * new_state_q_val)

            self.q_update(state=state, action=action, update_value=update_value, alpha_lr=alpha_lr)

            state = new_state
            if record:
                episode_fitness.append([step, [fitness]])

            if done and record:
                return episode_fitness
            elif done:
                return fitness

    def learn_n_episodes(self, alpha_lr, gamma, record_freq, epsilon, num_tests, new_water_food_loc):
        print(f"training for {self.num_episodes}")
        progress_bar = tqdm(range(self.num_episodes), desc="Episodes")
        fitness_running_avg = deque(maxlen=50)
        for i in progress_bar:
            fitness = self.learn_1_episode(
                alpha_lr=alpha_lr,
                gamma=gamma,
                epsilon=epsilon,
                new_water_food_loc=new_water_food_loc,
                record=False,
            )
            fitness_running_avg.append(fitness)
            progress_bar.set_postfix(avg_fitness=np.mean(fitness_running_avg))

            if i == self.num_episodes - 1:
                self.record_performance(
                    episode=i,
                    gamma=gamma,
                    epsilon=0,
                    num_tests=1000,
                    new_water_food_loc=new_water_food_loc,
                )
            elif i % record_freq == 0:
                self.record_performance(
                    episode=i,
                    gamma=gamma,
                    epsilon=epsilon,
                    num_tests=num_tests,
                    new_water_food_loc=new_water_food_loc,
                )
        return self.learning_performance


def create_q_learning_agent(env, hyper_params, new_water_food_loc=False, train=True, reward_fn=None, fitness_fn=None):
    assert "alpha_lr" in hyper_params.keys()
    assert "gamma" in hyper_params.keys()
    assert "epsilon" in hyper_params.keys()
    assert "num_episodes" in hyper_params.keys()
    assert "record_freq" in hyper_params.keys()
    assert "num_episodes" in hyper_params.keys()

    alpha_lr = hyper_params["alpha_lr"]
    epsilon = hyper_params["epsilon"]
    gamma = hyper_params["gamma"]
    num_tests = 1
    record_freq = hyper_params["record_freq"]
    num_episodes = hyper_params["num_episodes"]

    alg = QLearning(env, num_episodes=num_episodes, epsilon=epsilon, reward_fn=reward_fn, fitness_fn=fitness_fn)
    if train:
        if num_episodes == 1:
            return alg, alg.learn_1_episode(alpha_lr=alpha_lr, gamma=gamma, epsilon=epsilon)
        else:
            alg.learn_n_episodes(
                alpha_lr=alpha_lr,
                gamma=gamma,
                record_freq=record_freq,
                epsilon=epsilon,
                num_tests=num_tests,
                new_water_food_loc=new_water_food_loc,
            )

            return alg, alg.learning_performance
    else:
        return alg, None
