# Ranked Return Regression for RL (R4)

This repository contains the code release for the paper "Reward Learning through Ranking Mean Squared Error".

Paper: https://arxiv.org/abs/2601.09236

## Abstract

Reward design remains a significant bottleneck in applying reinforcement learning (RL) to real-world problems. A popular alternative is reward learning, where reward functions are inferred from human feedback rather than manually specified. Recent work has proposed learning reward functions from human feedback in the form of ratings, rather than traditional binary preferences, enabling richer and potentially less cognitively demanding supervision. Building on this paradigm, we introduce a new rating-based RL method, Ranked Return Regression for RL (R4). At its core, R4 employs a novel ranking mean squared error (rMSE) loss, which treats teacher-provided ratings as ordinal targets. Our approach learns from a dataset of trajectory-rating pairs, where each trajectory is labeled with a discrete rating (e.g., "bad," "neutral," "good"). At each training step, we sample a set of trajectories, predict their returns, and rank them using a differentiable sorting operator (soft ranks). We then optimize a mean squared error loss between the resulting soft ranks and the teacher's ratings. Unlike prior rating-based approaches, R4 offers formal guarantees: its solution set is provably minimal and complete under mild assumptions. Empirically, using simulated human feedback, we demonstrate that R4 consistently matches or outperforms existing rating and preference-based RL methods on robotic locomotion benchmarks from OpenAI Gym and the DeepMind Control Suite, while requiring significantly less feedback.

## Repository layout

- src/main.py: Offline rMSE reward learning entry point
- src/main_online.py: Online training with preference-based reward updates
- src/optimization.py: Shared optimization logic (offline and online)
- src/lossFunctions: rMSE losses (fast_soft_sort and OT variants)
- src/configs: Environment configurations for offline and online runs

## Setup

Create a Python 3.8.20 virtual environment from the repo root and install the project dependencies:

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r src/requirements.txt
```

Install `fast_soft_sort` in editable mode:

```bash
git clone https://github.com/google-research/fast-soft-sort.git
cd fast-soft-sort
pip install -e .
```

Return to this repository before running the offline or online entry points.


## Offline reward learning (rMSE)

Run the offline rMSE training from the repo root:

- python src/main.py --env_name hungrythirsty --num_files_to_read 1 --num_trajectories 1000

Common options:

- --loss_func ranking_mse (fast_soft_sort) or --loss_func ranking_mse_ot (OT-based)
- --ranking_assumption bin_ranking or full_ranking
- --model mini|medium|large|large2

Artifacts are saved under ../checkpoints/ours by default.

## Online reward learning

Run the online preference-based training from the repo root:

- python src/main_online.py --env_name walker_walk --task_name walk

Common options:

- --reward_model_arch mini|medium|large|large2
- --budget 100 --num_preferences 10 --num_generations 2000
- --subtrajectory_length 50

Artifacts are saved under ../checkpoints/online by default.

## Human Data

Data collected in our human study is available at [this link](https://drive.google.com/drive/folders/1kSoj8-OARPYaOCjqwNeCMZbCN75tiXKR?usp=sharing).

### Dataset Overview

We release the dataset of human ratings collected in the Reacher and Hopper environments to aid future research on reward learning from human rated feedback, particularly in settings involving noisy, subjective, and heterogeneous human supervision. Apart from the details already provided in previous sections, this section describes the dataset structure and usage.

Rating Scale and Heterogeneity: Ratings are absolute, discrete evaluations assigned independently by each annotator. As described in Section B.2, users were allowed to (1) choose their own rating scale, (2) extend the scale if they encountered behavior outside their chosen range, and (3) skip trajectories when uncertain. As a result, rating scales are annotator-specific and not globally normalized. The released dataset preserves all raw ratings without aggregation or rescaling. This design allows researchers to explicitly account for inter-rater variability, inconsistency, and noise, and to develop algorithms better suited to handling these challenges.

Dataset Statistics: For each environment and each rater, the dataset contains (1) a set of trajectories, (2) rating labels corresponding to each trajectory, and (3) the undiscounted environment returns associated with each trajectory. For Reacher, users employed an average of 6.22 rating classes, with a standard deviation of 2.30. Similarly, for Hopper, users employed an average of 6.57 rating classes, with a standard deviation of 2.26. The environment returns of the collected trajectories also exhibit substantial variability: for Reacher, average environment returns are −13.27 ± 9.02, ranging from −45.03 to −2.86, while for Hopper, average environment returns are 500.54 ± 239.97, ranging from 5.67 to 825.51.

File Structure: The ratings for each environment are stored in the corresponding folder named <Env name> Human. Each folder contains three pickle files per user: `pX labels.pkl`, `pX returns.pkl`, and `pX state action pairs.pkl`, where `X` denotes the user ID. The file `pX state action pairs.pkl` contains the trajectories shown to the user, while `pX labels.pkl` and `pX returns.pkl` contain the corresponding user-provided ratings and undiscounted environment returns, respectively.

Intended Use and Limitations: The dataset is intended for research on reward learning from human provided ratings, and robustness to noisy human supervision. Because ratings are subjective and use annotator-specific scales, direct comparison of raw rating values across annotators may be inappropriate without normalization or modeling assumptions. If using this provided dataset, researchers should account for this heterogeneity when designing learning algorithms or evaluation procedures.

## Citation

If you use this code, please cite the paper:

```
@inproceedings{kharyal2026rmse,
  title     = {Reward Learning through Ranking Mean Squared Error},
  author = {Kharyal, Chaitanya and Muslimani, Calarina and Taylor, Matthew E.}
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026},
}
```
