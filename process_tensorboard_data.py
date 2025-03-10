import matplotlib.pyplot as plt
from tbparse import SummaryReader

log_dir = "logs/0310-1642-HHHPPH-test-42-100000"
reader = SummaryReader(log_dir)
df = reader.scalars
rewards = df[df["tag"] == "Reward (Episode)"]
smoothed_rewards = rewards.value.rolling(200).max()
plt.plot(smoothed_rewards)
plt.savefig("test")
