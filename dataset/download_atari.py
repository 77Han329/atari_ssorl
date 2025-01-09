import gym
import numpy as np
import collections
import pickle
import d4rl_atari

env_names = [
    "pong-mixed-v4",
    "breakout-medium-v4",
    "breakout-expert-v4",
    # 如果还有其他，比如 "pong-mixed-v0", ...
    # 你也可以写在这里
]

for env_name in env_names:
    print(f"\n===== Processing {env_name} =====")
    env = gym.make(env_name)
    dataset = env.get_dataset()
    
    N = dataset["rewards"].shape[0]
    print(f"Total steps in dataset: {N}")
    
    data_ = collections.defaultdict(list)
    paths = []
    episode_step = 0

    for i in range(N):
        done_bool = bool(dataset["terminals"][i])
        
        if "next_observations" not in dataset:
            next_obs = (
                dataset["observations"][i+1] if (i < N - 1) else dataset["observations"][i]
            )
        else:
            next_obs = dataset["next_observations"][i]
            
        data_["observations"].append(dataset["observations"][i])
        data_["actions"].append(dataset["actions"][i])
        data_["rewards"].append(dataset["rewards"][i])
        data_["terminals"].append(dataset["terminals"][i])
        data_["next_observations"].append(next_obs) 
        
        if done_bool:
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            paths.append(episode_data)

            # 重置临时缓存
            data_ = collections.defaultdict(list)
            episode_step = 0
        else:
            episode_step += 1
            
    returns = np.array([np.sum(p["rewards"]) for p in paths])
    num_samples = np.sum([p["rewards"].shape[0] for p in paths])
    print(f"Number of samples collected: {num_samples}")
    print(
        f"Trajectory returns: mean = {returns.mean():.2f}, "
        f"std = {returns.std():.2f}, max = {returns.max():.2f}, min = {returns.min():.2f}"
    )

    # 存到本地 pkl 文件
    out_name = f"{env_name}.pkl"
    with open(out_name, "wb") as f:
        pickle.dump(paths, f)
    print(f"Saved {env_name} offline data to {out_name}")