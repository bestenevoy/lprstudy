import re, statistics
path = r"EduSim/Experiment_logs/KESassist15/Experiment_2/1.txt"
rewards = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        m = re.search(r"episode_reward:\s*([-+0-9.]+)", line)
        if m:
            rewards.append(float(m.group(1)))
print("count", len(rewards))
print("mean", sum(rewards)/len(rewards))