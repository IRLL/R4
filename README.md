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

## Citation

If you use this code, please cite the paper:

- Chaitanya Kharyal, Calarina Muslimani, Matthew E. Taylor. "Reward Learning through Ranking Mean Squared Error." ICML 2026.
