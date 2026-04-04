"""Quick smoke test for the full environment."""
import numpy as np

from junqi.env.junqi_env import JunqiEnv

env = JunqiEnv(max_turns=500)
obs, info = env.reset(template_idx_p0=0, template_idx_p1=1, seed=42)
print(f"Obs shape: {obs.shape}")
print(f"Action mask sum: {info['action_mask'].sum():.0f} legal actions")
print(f"Belief entropy: {info['belief_entropy']:.2f}")

# Play a few random moves
for step in range(10):
    mask = env.get_action_mask()
    legal = np.where(mask > 0)[0]
    action = int(np.random.choice(legal))
    obs, reward, done, trunc, info = env.step(action)
    print(f"Step {step+1}: P{info['current_player']} action={action}, reward={reward:.4f}, done={done}")
    if done:
        break

print(f"\nFinal state: {env.game}")
print("env OK!")
