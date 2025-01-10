"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import gym
import numpy as np
import collections
import pickle
import d4rl_atari  # 与 d4rl 对应的 Atari 扩展库

# 你可以根据需要来调整以下环境列表
# 例如: "breakout-expert-v0", "breakout-medium-v0", "breakout-mixed-v0", "breakout-full-v0"
# 以及其他已经在 d4rl_atari 中支持的环境
env_name = "breakout-medium-v4"
env = gym.make(env_name,stack=True)
dataset = env.get_dataset()

# 打印键信息
print(f"Keys in dataset: {list(dataset.keys())}")

# 检查每个键的长度
for k in dataset.keys():
    print(f"{k}: {len(dataset[k])}")

# 轨迹收集
data_ = collections.defaultdict(list)
paths = []

N = len(dataset["observations"])
for i in range(N):
    # 收集 step 数据
    for k in ["observations", "actions", "rewards", "terminals"]:
        data_[k].append(dataset[k][i])

    # 判断是否是轨迹结束
    done_bool = bool(dataset["terminals"][i])
    if done_bool or i == N - 1:  # 如果到达轨迹终点或数据集的最后一帧
        episode_data = {k: np.array(v) for k, v in data_.items()}
        paths.append(episode_data)
        data_ = collections.defaultdict(list)

# 打印轨迹统计信息
print(f"Number of trajectories collected: {len(paths)}")
if paths:
    returns = np.array([np.sum(p["rewards"]) for p in paths])
    print(f"Trajectory returns: mean = {np.mean(returns)}, max = {np.max(returns)}, min = {np.min(returns)}")

# 保存到文件
with open(f"{env_name}.pkl", "wb") as f:
    pickle.dump(paths, f)
print(f"Saved {len(paths)} trajectories to {env_name}.pkl")