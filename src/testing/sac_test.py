import gymnasium

from algos.sac import SACAgent
from utils import write_pickle


class SAC_Test:
    def __init__(
        self,
        reward_fn,
        fitness_fn,
        env_seeds_to_test,
        num_episodes,
        learning_start,
        record_freq,
        learning_performance_dir="../checkpoints/learning_performance",
        run_guid="run_guid",
        try_num=0,
        env_name="Reacher-v5",
        config=None,
    ):
        self.reward_fn = reward_fn
        self.fitness_fn = fitness_fn
        self.env_seeds_to_test = env_seeds_to_test
        self.num_episodes = num_episodes
        self.learning_start = learning_start
        self.record_freq = record_freq
        self.learning_performance_dir = learning_performance_dir
        self.run_guid = run_guid
        self.try_num = try_num
        self.env = gymnasium.make(env_name)
        self.config = config

    def test(self):
        env = self.env

        print("Testing")
        hyperparameters = {
            "num_episodes": self.num_episodes,
            "learning_start": self.learning_start,
            "record_freq": self.record_freq,
        }

        agent = SACAgent(env, hyperparameters, reward_fn=self.reward_fn, fitness_fn=self.fitness_fn)
        learning_performance = agent.train()
        self.save_learning_performance(learning_performance, 0)

    def save_learning_performance(self, learning_performance, seed):
        fitness_over_time = [x[1]["avg_fitness"] for x in learning_performance]
        undiscounted_return_over_time = [x[1]["avg_undiscounted_return"] for x in learning_performance]

        write_pickle(
            f"{self.learning_performance_dir}/{self.run_guid}_fitness_over_time_{seed}_try{self.try_num}.pkl",
            fitness_over_time,
        )
        write_pickle(
            f"{self.learning_performance_dir}/{self.run_guid}_undiscounted_return_over_time_{seed}_try{self.try_num}.pkl",
            undiscounted_return_over_time,
        )
