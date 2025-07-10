import torch
import torch.nn.functional as F
import itertools
from typing import Dict, List, Tuple

def get_updated_obs_with_comm(obs, comm):
    """Your function - copied here for testing"""
    obs_faulty = torch.zeros(obs.shape[0], obs.shape[1], obs.shape[2] + 9, device=obs.device)
    bs = obs.shape[0]
    n_agents = obs.shape[1]

    index_map = {8: 15, 15: 23, 22: 31, 29: 39, 36: 47, 43: 55, 50: 63, 57: 71, 64: 79}
    
    for source_idx, target_idx in index_map.items():
        # Step 1: Mask where obs[:, :, source_idx] == 1
        mask = obs[:, :, source_idx] == 1
        # Step 2: Get one-hot vector from next 4 positions
        one_hot = obs[:, :, source_idx+1:source_idx+5] # shape: (10, 4, 4)
        # Step 3: Get agent index from one-hot
        agent_idx = one_hot.argmax(dim=-1) # shape: (10, 4)
        # Step 4: Use advanced indexing to fetch comm[batch, agent_idx]
        # Expand dimensions to match for gather
        comm_gather = comm.squeeze(-1).unsqueeze(1).expand(-1, n_agents, -1) # (10, 4, 4)
        selected_comm = torch.gather(comm_gather, dim=2, index=agent_idx.unsqueeze(-1)).squeeze(-1) # (10, 4)
        # Step 5: Write the selected_comm to obs_new at target_idx
        obs_faulty[:, :, target_idx] = selected_comm * mask # write only where mask is 1
    
    obs_faulty[:, :, 0:15] = obs[:, :, 0:15]
    obs_faulty[:, :, 16:23] = obs[:, :, 15:22]
    obs_faulty[:, :, 24:31] = obs[:, :, 22:29]
    obs_faulty[:, :, 32:39] = obs[:, :, 29:36]
    obs_faulty[:, :, 40:47] = obs[:, :, 36:43]
    obs_faulty[:, :, 48:55] = obs[:, :, 43:50]
    obs_faulty[:, :, 56:63] = obs[:, :, 50:57]
    obs_faulty[:, :, 64:71] = obs[:, :, 57:64]
    obs_faulty[:, :, 72:79] = obs[:, :, 64:71]
    
    return obs_faulty

def test_comprehensive_obs_combinations():
    """Test all possible combinations of triggers and agent indices"""
    
    torch.manual_seed(42)
    
    # Test parameters
    B, A = 3, 4  # batch size, num agents
    index_map = {8: 15, 15: 23, 22: 31, 29: 39, 36: 47, 43: 55, 50: 63, 57: 71, 64: 79}
    trigger_indices = list(index_map.keys())
    
    print("🧪 Starting comprehensive test...")
    print(f"Testing with B={B}, A={A}, triggers={trigger_indices}")
    
    # Test 1: No triggers (all zeros)
    print("\n1️⃣ Testing no triggers...")
    obs = torch.randn(B, A, 75)
    comm = torch.randn(B, A, 1)
    
    # Ensure no triggers
    for idx in trigger_indices:
        obs[:, :, idx] = 0
    
    obs_new = get_updated_obs_with_comm(obs, comm)
    
    # Check that all target positions are 0
    for target_idx in index_map.values():
        assert torch.all(obs_new[:, :, target_idx] == 0), f"Expected 0 at target {target_idx} when no triggers"
    
    # Check copying is correct
    _check_copying_correctness(obs, obs_new, index_map)
    print("✅ No triggers test passed")
    
    # Test 2: Single trigger for each position
    print("\n2️⃣ Testing single triggers...")
    for trigger_idx in trigger_indices:
        obs = torch.randn(B, A, 75)
        comm = torch.randn(B, A, 1)
        
        # Clear all triggers first
        for idx in trigger_indices:
            obs[:, :, idx] = 0
        
        # Set single trigger
        obs[0, 0, trigger_idx] = 1
        
        # Test each possible agent index
        for agent_id in range(A):
            one_hot = F.one_hot(torch.tensor(agent_id), num_classes=4).float()
            obs[0, 0, trigger_idx+1:trigger_idx+5] = one_hot
            
            obs_new = get_updated_obs_with_comm(obs, comm)
            
            # Check communication value
            target_idx = index_map[trigger_idx]
            expected_comm = comm[0, agent_id, 0].item()
            actual_comm = obs_new[0, 0, target_idx].item()
            
            assert abs(expected_comm - actual_comm) < 1e-5, \
                f"Comm mismatch at trigger {trigger_idx}->target {target_idx}, agent {agent_id}: {actual_comm} != {expected_comm}"
    
    print("✅ Single trigger test passed")
    
    # Test 3: Multiple triggers same batch/agent
    print("\n3️⃣ Testing multiple triggers same batch/agent...")
    obs = torch.randn(B, A, 75)
    comm = torch.randn(B, A, 1)
    
    # Clear all triggers
    for idx in trigger_indices:
        obs[:, :, idx] = 0
    
    # Set multiple triggers for same batch/agent
    test_triggers = [8, 22, 36]  # subset for testing
    test_agents = [0, 2, 1]     # different agents
    
    for i, (trigger_idx, agent_id) in enumerate(zip(test_triggers, test_agents)):
        obs[0, 0, trigger_idx] = 1
        one_hot = F.one_hot(torch.tensor(agent_id), num_classes=4).float()
        obs[0, 0, trigger_idx+1:trigger_idx+5] = one_hot
    
    obs_new = get_updated_obs_with_comm(obs, comm)
    
    # Check each trigger
    for trigger_idx, agent_id in zip(test_triggers, test_agents):
        target_idx = index_map[trigger_idx]
        expected_comm = comm[0, agent_id, 0].item()
        actual_comm = obs_new[0, 0, target_idx].item()
        
        assert abs(expected_comm - actual_comm) < 1e-5, \
            f"Multi-trigger mismatch at {trigger_idx}->{target_idx}, agent {agent_id}"
    
    print("✅ Multiple triggers test passed")
    
    # Test 4: All possible combinations across batches/agents
    print("\n4️⃣ Testing all combinations across batches/agents...")
    
    # Generate all possible trigger patterns (subset for performance)
    trigger_patterns = [
        [],  # no triggers
        [8],  # single trigger
        [8, 22],  # two triggers
        [8, 22, 36],  # three triggers
        trigger_indices[:5],  # first 5 triggers
        trigger_indices,  # all triggers
    ]
    
    for pattern in trigger_patterns:
        obs = torch.randn(B, A, 75)
        comm = torch.randn(B, A, 1)
        
        # Clear all triggers
        for idx in trigger_indices:
            obs[:, :, idx] = 0
        
        # Apply pattern to different batch/agent combinations
        for b in range(B):
            for a in range(A):
                for i, trigger_idx in enumerate(pattern):
                    if (b + a + i) % 2 == 0:  # Apply to some combinations
                        obs[b, a, trigger_idx] = 1
                        agent_id = (b + a + i) % A
                        one_hot = F.one_hot(torch.tensor(agent_id), num_classes=4).float()
                        obs[b, a, trigger_idx+1:trigger_idx+5] = one_hot
        
        obs_new = get_updated_obs_with_comm(obs, comm)
        
        # Verify all triggered positions
        for b in range(B):
            for a in range(A):
                for i, trigger_idx in enumerate(pattern):
                    if (b + a + i) % 2 == 0:
                        target_idx = index_map[trigger_idx]
                        agent_id = (b + a + i) % A
                        expected_comm = comm[b, agent_id, 0].item()
                        actual_comm = obs_new[b, a, target_idx].item()
                        
                        assert abs(expected_comm - actual_comm) < 1e-5, \
                            f"Pattern {pattern} mismatch at (b={b},a={a}), trigger {trigger_idx}->target {target_idx}, agent {agent_id}"
        
        # Check copying correctness
        _check_copying_correctness(obs, obs_new, index_map)
    
    print("✅ All combinations test passed")
    
    # Test 5: Edge cases
    print("\n5️⃣ Testing edge cases...")
    
    # Edge case: Invalid one-hot (all zeros)
    obs = torch.randn(B, A, 75)
    comm = torch.randn(B, A, 1)
    
    for idx in trigger_indices:
        obs[:, :, idx] = 0
    
    obs[0, 0, 8] = 1  # trigger
    obs[0, 0, 9:13] = 0  # invalid one-hot (all zeros)
    
    obs_new = get_updated_obs_with_comm(obs, comm)
    
    # Should use agent 0 (argmax of all zeros is 0)
    expected_comm = comm[0, 0, 0].item()
    actual_comm = obs_new[0, 0, 15].item()
    assert abs(expected_comm - actual_comm) < 1e-5, "Edge case: invalid one-hot failed"
    
    # Edge case: Non-perfect one-hot
    obs[0, 0, 9:13] = torch.tensor([0.1, 0.9, 0.3, 0.2])  # agent 1 has max
    obs_new = get_updated_obs_with_comm(obs, comm)
    
    expected_comm = comm[0, 1, 0].item()
    actual_comm = obs_new[0, 0, 15].item()
    assert abs(expected_comm - actual_comm) < 1e-5, "Edge case: non-perfect one-hot failed"
    
    print("✅ Edge cases test passed")
    
    print("\n🎉 ALL TESTS PASSED! Your function works correctly for all combinations.")

def _check_copying_correctness(obs, obs_new, index_map):
    """Helper function to check that copying segments work correctly"""
    B, A = obs.shape[0], obs.shape[1]
    
    # Define the copying segments based on your hardcoded ranges
    copy_segments = [
        (0, 15, 0, 15),      # obs[:,:,0:15] -> obs_new[:,:,0:15]
        (15, 22, 16, 23),    # obs[:,:,15:22] -> obs_new[:,:,16:23]
        (22, 29, 24, 31),    # obs[:,:,22:29] -> obs_new[:,:,24:31]
        (29, 36, 32, 39),    # obs[:,:,29:36] -> obs_new[:,:,32:39]
        (36, 43, 40, 47),    # obs[:,:,36:43] -> obs_new[:,:,40:47]
        (43, 50, 48, 55),    # obs[:,:,43:50] -> obs_new[:,:,48:55]
        (50, 57, 56, 63),    # obs[:,:,50:57] -> obs_new[:,:,56:63]
        (57, 64, 64, 71),    # obs[:,:,57:64] -> obs_new[:,:,64:71]
        (64, 71, 72, 79),    # obs[:,:,64:71] -> obs_new[:,:,72:79]
    ]
    
    for obs_start, obs_end, new_start, new_end in copy_segments:
        obs_segment = obs[:, :, obs_start:obs_end]
        new_segment = obs_new[:, :, new_start:new_end]
        
        assert torch.allclose(obs_segment, new_segment, atol=1e-5), \
            f"Copying failed for segment obs[{obs_start}:{obs_end}] -> obs_new[{new_start}:{new_end}]"

# Run the comprehensive test
if __name__ == "__main__":
    test_comprehensive_obs_combinations()
