# DRL-SnowDrone-Project
Comparative study of DRL algorithms (DQN, Double DQN, REINFORCE, PPO, A2C) on LunarLander-v2 and custom Snow Drone Rescue environment
# DRL-SnowDrone-Project 🚀

A comprehensive comparative study of Deep Reinforcement Learning algorithms applied to both benchmark and custom environments.

## 📋 Project Overview

This repository contains the final project for **DA 346: Deep Reinforcement Learning**, featuring:
- **Part 1**: Comparative analysis of 5 DRL algorithms on OpenAI Gym's LunarLander-v2 environment
- **Part 2**: Design and implementation of a custom Snow Drone Rescue environment with DQN


## 🧠 Algorithms Studied

| Algorithm | Type | Policy | Key Features |
|-----------|------|--------|--------------|
| **DQN** | Value-based | Off-policy | Experience replay, target network |
| **Double DQN** | Value-based | Off-policy | Reduced overestimation bias |
| **REINFORCE** | Policy-based | On-policy | Monte Carlo policy gradient |
| **PPO** | Policy-based | On-policy | Clipped surrogate objective, GAE |
| **A2C** | Actor-Critic | On-policy | Advantage estimation |

## 📊 Key Findings

### LunarLander-v2 Results:
- **DQN** solved in **461 episodes** (avg score: 200.87)
- **Double DQN** solved in **798 episodes** (avg score: 202.32)
- **REINFORCE** failed to converge (avg score: -565.68)
- **PPO** showed gradual improvement but didn't solve
- **A2C** achieved only 58.00 avg score over 8000 episodes

### Snow Drone Rescue Environment:
- Custom 2D navigation task with battery management
- Partial observability and stochastic elements
- Successfully trained with DQN
- Demonstrated emergent behaviors: obstacle avoidance, energy-aware navigation

## 🛠️ Technologies Used

- **Python 3.8+**
- **PyTorch** (for DRL implementations)
- **Gymnasium** (for LunarLander-v2)
- **PyGame** (for custom environment visualization)
- **Jupyter Notebook** (for analysis and visualization)

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch gymnasium pygame numpy matplotlib
