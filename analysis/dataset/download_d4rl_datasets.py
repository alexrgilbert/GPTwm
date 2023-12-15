import gym
import numpy as np

import collections
import pickle

import d4rl

def download_dataset(name):
    env = gym.make(name)
    dataset = env.get_dataset()

    N = dataset["rewards"].shape[0]
    data_ = collections.defaultdict(list)

    use_timeouts = False
    if "timeouts" in dataset:
        use_timeouts = True

    episode_step = 0
    paths = []
    for i in range(N):
        done_bool = bool(dataset["terminals"][i])
        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == 1000 - 1
        for k in [
            "observations",
            "timeouts",
            "actions",
            "rewards",
            "terminals",
        ]:
            data_[k].append(dataset[k][i])
        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            paths.append(episode_data)
            data_ = collections.defaultdict(list)
        episode_step += 1

    returns = np.array([np.sum(p["rewards"]) for p in paths])
    num_samples = np.sum([p["rewards"].shape[0] for p in paths])
    print(f"Number of samples collected: {num_samples}")
    print(
        f"Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}"
    )

    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(paths, f)


datasets = []

# for env_name in ["halfcheetah", "hopper", "walker2d"]:
#     for dataset_type in ["medium"]: #, "medium-replay", "expert", "medium-expert", "random"]:
#         name = f"{env_name}-{dataset_type}-v2"
#         download_dataset(name)

# download_dataset('maze2d-open-v0')
# download_dataset('maze2d-umaze-v1')
# download_dataset('maze2d-open-dense-v0')
# download_dataset('maze2d-umaze-dense-v1')

download_dataset('cartpole-expert-v0')