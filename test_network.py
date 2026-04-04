"""Test network and PPO agent."""
import numpy as np
import torch

from junqi.env.junqi_env import JunqiEnv
from junqi.model.network import JunqiNetwork
from junqi.training.ppo import PPOAgent, RolloutBuffer

# Create network
net = JunqiNetwork(hidden_dim=64, num_res_blocks=4)
print(f"Network parameters: {net.count_parameters():,}")

# Test forward pass
dummy_state = torch.randn(2, 62, 12, 5)
dummy_mask = torch.ones(2, 3600)
policy, value, belief = net(dummy_state, dummy_mask)
print(f"Policy shape: {policy.shape}")  # (2, 3600)
print(f"Value shape: {value.shape}")    # (2, 1)
print(f"Belief shape: {belief.shape}")  # (2, 12, 12, 5)

# Test PPO agent
agent = PPOAgent(net, lr=1e-3, device='cpu')

# Collect a small buffer
env = JunqiEnv(max_turns=50)
obs, info = env.reset(seed=42)
buf = RolloutBuffer()

for step in range(20):
    mask = env.get_action_mask()
    action, log_prob, value = agent.select_action(obs, mask)
    obs, reward, done, trunc, info = env.step(action)
    buf.add(obs, action, mask, log_prob, reward, value, done)
    if done:
        break

print(f"\nBuffer size: {len(buf)}")

# PPO update
stats = agent.update([buf])
print(f"PPO update stats:")
for k, v in stats.items():
    print(f"  {k}: {v:.4f}")

print("\nAll tests passed!")
