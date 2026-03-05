"""
Shared Weights Manager for Multi-GPU RL Training

Handles synchronization of Q-network weights between main training
process and worker processes collecting episodes.
"""

import torch
import torch.nn as nn
import multiprocessing as mp
from typing import Dict, OrderedDict
from collections import OrderedDict as ODict


class SharedWeights:
    """
    Manages shared Q-network weights between main and worker processes.

    The main process (training) periodically updates weights after
    gradient descent steps. Worker processes periodically copy the
    latest weights for action selection.

    Key features:
    - Version tracking to detect updates
    - Thread-safe access with locks
    - Efficient state_dict sharing
    - CPU-side operations to avoid CUDA context issues
    """

    def __init__(self, manager: mp.Manager):
        """
        Initialize shared weights manager.

        Args:
            manager: multiprocessing.Manager instance
        """
        self.manager = manager

        # Shared dictionary to store weights
        self.weights_dict = manager.dict()

        # Lock for thread-safe access
        self.lock = manager.Lock()

        # Version counter to track updates
        self.version = manager.Value('i', 0)

        print(f"✅ SharedWeights initialized")

    def update(self, q_network: nn.Module):
        """
        Update shared weights from Q-network (called by main process).

        Args:
            q_network: Q-network to copy weights from
        """
        with self.lock:
            # Move state dict to CPU for sharing
            # (avoids CUDA context issues across processes)
            state_dict = {}
            for key, value in q_network.state_dict().items():
                # Detach and move to CPU
                state_dict[key] = value.detach().cpu()

            # Store in shared dict
            self.weights_dict['q_network'] = state_dict

            # Increment version
            self.version.value += 1

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get current Q-network weights (called by workers).

        Returns:
            State dict with weights on CPU
        """
        with self.lock:
            if 'q_network' not in self.weights_dict:
                return None

            # Create a regular dict from shared dict
            # (shared dict can have serialization issues)
            state_dict = dict(self.weights_dict['q_network'])

        return state_dict

    def get_version(self) -> int:
        """
        Get current version number.

        Workers can check version to see if weights have been updated
        since last sync.

        Returns:
            Current version number
        """
        return self.version.value

    def is_initialized(self) -> bool:
        """
        Check if weights have been initialized.

        Returns:
            True if weights are available
        """
        with self.lock:
            return 'q_network' in self.weights_dict


class WorkerWeightSyncer:
    """
    Helper class for workers to manage weight synchronization.

    Tracks local version and efficiently syncs when new weights are available.
    """

    def __init__(self, shared_weights: SharedWeights, q_network: nn.Module,
                 sync_frequency: int = 500):
        """
        Initialize worker weight syncer.

        Args:
            shared_weights: SharedWeights instance
            q_network: Local Q-network to update
            sync_frequency: How often to check for updates (in steps)
        """
        self.shared_weights = shared_weights
        self.q_network = q_network
        self.sync_frequency = sync_frequency

        # Track local version and steps
        self.local_version = 0
        self.steps_since_sync = 0

    def maybe_sync(self, force: bool = False) -> bool:
        """
        Sync weights if new version is available or frequency reached.

        Args:
            force: Force sync regardless of frequency

        Returns:
            True if weights were synced
        """
        self.steps_since_sync += 1

        # Check if it's time to sync
        if not force and self.steps_since_sync < self.sync_frequency:
            return False

        # Check if new version is available
        remote_version = self.shared_weights.get_version()

        if remote_version <= self.local_version and not force:
            # No new weights available
            self.steps_since_sync = 0
            return False

        # Get new weights
        state_dict = self.shared_weights.get_state_dict()

        if state_dict is None:
            return False

        # Load into local Q-network
        # Move to network's device
        device = next(self.q_network.parameters()).device
        for key, value in state_dict.items():
            state_dict[key] = value.to(device)

        self.q_network.load_state_dict(state_dict)

        # Update tracking
        self.local_version = remote_version
        self.steps_since_sync = 0

        return True

    def get_sync_info(self) -> Dict:
        """
        Get synchronization status information.

        Returns:
            Dictionary with sync status
        """
        return {
            'local_version': self.local_version,
            'remote_version': self.shared_weights.get_version(),
            'steps_since_sync': self.steps_since_sync,
            'is_stale': self.shared_weights.get_version() > self.local_version
        }
