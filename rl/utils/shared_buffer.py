"""
Thread-Safe Shared Replay Buffer for Multi-GPU RL Training

Uses preallocated shared memory tensors and multiprocessing locks
to enable concurrent access from multiple worker processes.
"""

import torch
import numpy as np
import multiprocessing as mp
from typing import Dict, List, Optional


class SharedReplayBuffer:
    """
    Thread-safe replay buffer using shared memory for multi-GPU training.

    Uses preallocated torch tensors with .share_memory_() to avoid
    serialization overhead. Implements a circular buffer with atomic
    operations protected by multiprocessing.Lock.

    Key features:
    - Zero-copy sharing between processes
    - Lock-based synchronization for push/sample
    - Circular buffer with automatic wraparound
    - Minimal lock hold time for maximum throughput
    """

    def __init__(self, capacity: int, state_dim: int):
        """
        Initialize shared replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state vectors
        """
        self.capacity = capacity
        self.state_dim = state_dim

        # Preallocate shared memory tensors
        # Using .share_memory_() makes tensors accessible across processes
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self.states.share_memory_()

        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.actions.share_memory_()

        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.rewards.share_memory_()

        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self.next_states.share_memory_()

        self.dones = torch.zeros(capacity, dtype=torch.bool)
        self.dones.share_memory_()

        # Synchronization primitives (shared across processes)
        self.lock = mp.Lock()
        self.write_idx = mp.Value('i', 0)  # Current write index
        self.size = mp.Value('i', 0)       # Current buffer size

        print(f"✅ SharedReplayBuffer initialized")
        print(f"   Capacity: {capacity:,} transitions")
        print(f"   State dim: {state_dim}")
        print(f"   Memory usage: ~{self._estimate_memory_mb():.1f} MB")

    def _estimate_memory_mb(self) -> float:
        """Estimate memory usage in MB."""
        # states: capacity × state_dim × 4 bytes (float32)
        # next_states: capacity × state_dim × 4 bytes
        # actions: capacity × 8 bytes (int64)
        # rewards: capacity × 4 bytes (float32)
        # dones: capacity × 1 byte (bool)
        states_mb = (self.capacity * self.state_dim * 4) / (1024 * 1024)
        next_states_mb = (self.capacity * self.state_dim * 4) / (1024 * 1024)
        actions_mb = (self.capacity * 8) / (1024 * 1024)
        rewards_mb = (self.capacity * 4) / (1024 * 1024)
        dones_mb = (self.capacity * 1) / (1024 * 1024)

        return states_mb + next_states_mb + actions_mb + rewards_mb + dones_mb

    def push(self, state: torch.Tensor, action: int, reward: float,
             next_state: torch.Tensor, done: bool):
        """
        Add a transition to the buffer (thread-safe).

        Args:
            state: Current state tensor (state_dim,)
            action: Action taken
            reward: Reward received
            next_state: Next state tensor (state_dim,)
            done: Whether episode ended
        """
        # Ensure tensors are on CPU for copying to shared memory
        if state.device.type == 'cuda':
            state = state.cpu()
        if next_state.device.type == 'cuda':
            next_state = next_state.cpu()

        # Acquire lock for thread-safe access
        with self.lock:
            # Get current write position
            idx = self.write_idx.value

            # Copy data to shared memory (detach to avoid gradient tracking)
            self.states[idx] = state.detach()
            self.actions[idx] = action
            self.rewards[idx] = reward
            self.next_states[idx] = next_state.detach()
            self.dones[idx] = done

            # Update write index (circular buffer)
            self.write_idx.value = (idx + 1) % self.capacity

            # Update size (saturate at capacity)
            self.size.value = min(self.size.value + 1, self.capacity)

    def sample(self, batch_size: int, device: str = 'cuda:0') -> Dict[str, torch.Tensor]:
        """
        Sample a random batch from the buffer (thread-safe).

        Args:
            batch_size: Number of transitions to sample
            device: Device to move tensors to

        Returns:
            Dictionary with keys: states, actions, rewards, next_states, dones
        """
        with self.lock:
            current_size = self.size.value

            # Don't sample more than available
            actual_batch_size = min(batch_size, current_size)

            if actual_batch_size == 0:
                # Return empty batch
                return {
                    'states': torch.empty((0, self.state_dim), device=device),
                    'actions': torch.empty(0, dtype=torch.long, device=device),
                    'rewards': torch.empty(0, device=device),
                    'next_states': torch.empty((0, self.state_dim), device=device),
                    'dones': torch.empty(0, dtype=torch.bool, device=device)
                }

            # Random sampling without replacement
            indices = torch.randperm(current_size)[:actual_batch_size]

            # Gather samples (still under lock to ensure consistency)
            batch = {
                'states': self.states[indices].clone().to(device),
                'actions': self.actions[indices].clone().to(device),
                'rewards': self.rewards[indices].clone().to(device),
                'next_states': self.next_states[indices].clone().to(device),
                'dones': self.dones[indices].clone().to(device)
            }

        return batch

    def __len__(self) -> int:
        """Return current size of buffer."""
        with self.lock:
            return self.size.value

    def is_ready(self, min_size: int) -> bool:
        """
        Check if buffer has enough samples for training.

        Args:
            min_size: Minimum number of samples required

        Returns:
            True if buffer size >= min_size
        """
        return len(self) >= min_size

    def get_stats(self) -> Dict:
        """Get statistics about buffer contents."""
        with self.lock:
            current_size = self.size.value

            if current_size == 0:
                return {
                    'size': 0,
                    'capacity': self.capacity,
                    'capacity_used': 0.0,
                    'avg_reward': 0.0,
                    'std_reward': 0.0
                }

            # Sample rewards for statistics (without full lock)
            rewards_sample = self.rewards[:current_size].clone()

        # Compute stats outside lock
        return {
            'size': current_size,
            'capacity': self.capacity,
            'capacity_used': current_size / self.capacity,
            'avg_reward': float(rewards_sample.mean()),
            'std_reward': float(rewards_sample.std()),
            'min_reward': float(rewards_sample.min()),
            'max_reward': float(rewards_sample.max())
        }

    def clear(self):
        """Clear all transitions from buffer."""
        with self.lock:
            self.write_idx.value = 0
            self.size.value = 0
