# Project Setup and Environment Registration

This project provides custom Spider environments for Gymnasium and Gymnasium-Robotics.  
Unlike the previous approach, environments are now **automatically loaded from the local `env/` folder**, so manual copying into Gymnasium or Gymnasium-Robotics directories is no longer required.
---

## Demo Videos
https://github.com/user-attachments/assets/891cee29-d6a0-475c-9df1-bce40d891837

---

## 1. Key Update

Previously, environments had to be manually added to Gymnasium’s internal directories.  
Now, the project uses a **local registration system (`env/register.py`)**, which dynamically registers all environments directly from the project.

✔ No manual file copying  
✔ No modification of site-packages  
✔ Cleaner and portable setup  

---

## 2. Project Directory Structure

Ensure your project root has the following layout:

```
Project/
|── env
│   ├── assets
│   |   └── spider.xml
│   ├── __init__.py
│   ├── register.py
│   ├── spider_v0.py
│   └── spider_maze_v0.py
├── Videos/
│   ├── demo.mp4
│   ├── ddpg-spider-step-0-to-step-2000.mp4
│   ├── SAC-spid-step-0-to-step-2000.mp4
│   ├── ppo-spid-step-0-to-step-2000.mp4
│   ├── ppo-spider-step-0-to-step-1000.mp4
│   └── ppo-spider-step-0-to-step-2000.mp4
│   └── ppo-spiderMaze1-step-0-to-step-2000.mp4
├── main.py
├── spiderm.zip
├── train_spider.py
├── test_spider.py
├── train_spiderMaze.py
└── test_spiderMaze.py
```
---

## Environment Files

- **`env/assets/spider.xml`**  
  MuJoCo model file for the Spider robot.

- **`env/spider_v0.py`**  
  Core Spider locomotion environment.

- **`env/spider_maze_v0.py`**  
  Maze-based navigation environment for the Spider.

- **`env/register.py`**  
  Automatically registers all custom environments with Gymnasium.

---

## Setup Instructions

### 1. Install Dependencies

Make sure you have the required packages installed:

```bash
pip install -r requirements.txt
````

---

### 3. Using the Environments

#### Spider Environment

```python
import gymnasium as gym
from env import register

register.register_envs()
env = gym.make("Spider-v0")
obs, _ = env.reset()

print("Spider-v0 environment loaded successfully")
```

---

#### Spider Maze Environment

```python
import gymnasium as gym
from env import register

register.register_envs()
env = gym.make("SpiderMaze_UMaze-v0")
obs, _ = env.reset()

print("SpiderMaze_UMaze-v0 environment loaded successfully")
```

---

## Training and Testing

* **`train_spider.py`** → Train Spider locomotion agent
* **`test_spider.py`** → Evaluate trained Spider agent
* **`train_spiderMaze.py`** → Train maze navigation agent
* **`test_spiderMaze.py`** → Evaluate maze agent

---

## Videos

The `Videos/` folder contains rollout recordings for different algorithms:

* PPO
* SAC
* DDPG
* Spider locomotion and maze navigation demos

---

## Pretrained Model

* **`spiderm.zip`**
  Contains a pretrained Spider agent for quick testing.

---

## Verification

Run a quick test to confirm everything works:

```python
import gymnasium as gym
import env.register

env = gym.make("Spider-v0")
obs, _ = env.reset()

print("Environment loaded successfully!")
```

---

## 📝 Notes

* No need to modify Gymnasium or Gymnasium-Robotics source code.
* No need to copy `spider.xml` into site-packages.
* Everything is self-contained within the project.
* Always ensure `env.register` is imported before calling `gym.make()`.

