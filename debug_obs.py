from model.env_wrapper import MontrealSnowEnv
import numpy as np

# 1. Initialize and Reset
env = MontrealSnowEnv()
obs, infos = env.reset(seed=42)

# 2. Pick the first agent to examine
agent = env.possible_agents[0]
agent_obs = obs[agent]
obs_space = env.observation_space(agent)

print(f"\n=== DEBUGGING SPACES FOR: {agent} ===")

# 3. The Ultimate Gymnasium Truth Test
is_valid = obs_space.contains(agent_obs)
if is_valid:
    print("✅ obs_space.contains(obs) PASSED! (The arrays perfectly match the spaces)")
else:
    print("❌ obs_space.contains(obs) FAILED! (There is a mismatch below)")

print("\n=== TENSOR BREAKDOWN ===")

# 4. Check the main graph observations
for key, matrix in agent_obs["observation"].items():
    expected_space = obs_space['observation'][key]
    
    shape_match = "✅" if matrix.shape == expected_space.shape else "❌"
    type_match = "✅" if matrix.dtype == expected_space.dtype else "❌"
    
    print(f"{key.upper()}:")
    print(f"  {shape_match} Shape: {matrix.shape} (Expected: {expected_space.shape})")
    print(f"  {type_match} Dtype: {matrix.dtype} (Expected: {expected_space.dtype})")
    
    # We use np.min/np.max to safely check arrays even if they are empty
    min_val = np.min(matrix) if matrix.size > 0 else 0
    max_val = np.max(matrix) if matrix.size > 0 else 0
    print(f"  ℹ️ Values: Min = {min_val:.1f}, Max = {max_val:.1f}")
    print("-" * 40)

# 5. Check the Action Mask
mask = agent_obs["action_mask"]
expected_mask = obs_space['action_mask']
shape_match = "✅" if mask.shape == expected_mask.shape else "❌"
type_match = "✅" if mask.dtype == expected_mask.dtype else "❌"

print(f"ACTION_MASK:")
print(f"  {shape_match} Shape: {mask.shape} (Expected: {expected_mask.shape})")
print(f"  {type_match} Dtype: {mask.dtype} (Expected: {expected_mask.dtype})")
print(f"  ℹ️ Values: Min = {mask.min()}, Max = {mask.max()}")
print("=======================================\n")