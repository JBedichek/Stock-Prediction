"""
Reinforcement Learning Components for Stock Trading

This module contains the core RL neural network components:
- PredictorFeatureExtractor: Wraps existing price predictor to extract RL features
- StockDQN: Dueling DQN network for Q-value estimation
- TradingAgent: Complete RL agent combining feature extraction and Q-learning
- ReplayBuffer: Experience replay buffer for stable training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
from typing import Dict, Tuple, Optional, List
import numpy as np


class PredictorFeatureExtractor(nn.Module):
    """
    Wraps existing t_Dist_Pred model to extract rich features for RL.

    Features extracted:
    - Transformer activations (604 dims): Learned temporal patterns
    - Probability distributions (1280 dims): Full uncertainty quantification
    - Distribution entropy (4 dims): Confidence per horizon
    - Expected returns (4 dims): Weighted predictions
    - Fundamentals (27 dims): Time-varying metrics

    Total: 1919 dimensions
    """

    def __init__(self, predictor_checkpoint_path: str, bin_edges_path: str = 'adaptive_bin_edges.pt'):
        """
        Initialize feature extractor.

        Args:
            predictor_checkpoint_path: Path to trained price predictor checkpoint
            bin_edges_path: Path to adaptive bin edges for expected return calculation
        """
        super().__init__()

        # Load pretrained predictor
        print(f"Loading predictor from: {predictor_checkpoint_path}")
        self.predictor = torch.load(predictor_checkpoint_path)
        self.predictor.eval()  # Start in eval mode

        # Load bin edges for expected return calculation
        print(f"Loading bin edges from: {bin_edges_path}")
        self.bin_edges = torch.load(bin_edges_path)

        # Freeze predictor initially (will unfreeze later for joint training)
        for param in self.predictor.parameters():
            param.requires_grad = False

        print(f"✅ Feature extractor initialized")
        print(f"   Predictor frozen: {not any(p.requires_grad for p in self.predictor.parameters())}")

    def extract_features(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Extract RL features from input data using the price predictor.

        Args:
            input_features: Input tensor of shape (batch, seq_len, feature_dim)
                           Contains price sequences and summary features

        Returns:
            Feature tensor of shape (batch, 1919):
            - t_act: 604 dims (transformer activations)
            - prob_dist: 1280 dims (320 bins × 4 horizons, flattened)
            - entropy: 4 dims (confidence per horizon)
            - expected_returns: 4 dims (weighted predictions)
            - fundamentals: 27 dims (from input)
        """
        batch_size = input_features.shape[0]
        device = input_features.device

        # Ensure bin_edges is on same device
        if self.bin_edges.device != device:
            self.bin_edges = self.bin_edges.to(device)

        # Split input into price sequence and summary
        # Assuming first 218 dims are price sequence, rest are summary
        price_seq = input_features[:, :, :218]
        summary = input_features[:, :, 218:]

        # Use existing forward_with_t_act() method from models.py:319
        # Returns: (predictions, transformer_activations)
        with torch.set_grad_enabled(self.predictor.training):
            pred, t_act = self.predictor.forward_with_t_act(price_seq, summary)

        # Convert logits to probabilities
        # pred shape: (batch, num_bins=320, num_horizons=4)
        prob_dist = F.softmax(pred, dim=1)

        # Compute entropy per horizon (confidence measure)
        # Low entropy = confident, high entropy = uncertain
        entropy = -(prob_dist * torch.log(prob_dist + 1e-10)).sum(dim=1)  # (batch, 4)

        # Compute expected returns (weighted average of bin centers)
        bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2  # (320,)
        bin_centers = bin_centers.to(device).unsqueeze(0).unsqueeze(2)  # (1, 320, 1)
        expected_returns = (prob_dist * bin_centers).sum(dim=1)  # (batch, 4)

        # Extract fundamentals from input (assuming they're at positions 768:795 in summary)
        # This matches the structure from dataset_processing.py
        fundamentals = input_features[:, 0, 768:795]  # (batch, 27)

        # Concatenate all features
        features = torch.cat([
            t_act,                      # 604 dims
            prob_dist.flatten(1),       # 1280 dims (320×4)
            entropy,                    # 4 dims
            expected_returns,           # 4 dims
            fundamentals                # 27 dims
        ], dim=1)  # Total: 1919 dims

        return features

    def unfreeze_predictor(self):
        """Unfreeze predictor for joint training."""
        for param in self.predictor.parameters():
            param.requires_grad = True
        self.predictor.train()
        print("🔥 Predictor unfrozen for joint training")

    def freeze_predictor(self):
        """Freeze predictor (disable gradient updates)."""
        for param in self.predictor.parameters():
            param.requires_grad = False
        self.predictor.eval()
        print("❄️  Predictor frozen")


class StockDQN(nn.Module):
    """
    Dueling DQN network for per-stock Q-value estimation.

    Architecture:
    - Shared feature extractor (2 layers)
    - Dueling streams:
      - Value stream: V(s) - value of being in state s
      - Advantage stream: A(s,a) - advantage of taking action a in state s
    - Combined: Q(s,a) = V(s) + [A(s,a) - mean(A)]

    Dueling architecture is better for comparing action values,
    which is critical for our buy/sell/hold decisions.
    """

    def __init__(self, state_dim: int = 1920, hidden_dim: int = 1024, action_dim: int = 5):
        """
        Initialize DQN.

        Args:
            state_dim: Input state dimension (1920 = 1919 predictor features + 1 portfolio context placeholder)
            hidden_dim: Hidden layer dimension
            action_dim: Number of actions (5 = HOLD, BUY_SMALL, BUY_MEDIUM, BUY_LARGE, SELL)
        """
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

        # Advantage stream: A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, action_dim)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Args:
            state: Input state tensor of shape (batch, state_dim)

        Returns:
            Q-values of shape (batch, action_dim)
        """
        # Shared feature extraction
        features = self.shared(state)

        # Value and advantage streams
        value = self.value_stream(features)  # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, action_dim)

        # Dueling aggregation: Q(s,a) = V(s) + [A(s,a) - mean(A(s,a'))]
        # Subtracting mean helps identifiability of V and A
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


class TradingAgent(nn.Module):
    """
    Complete RL trading agent.

    Combines:
    - Feature extraction from price predictor
    - Q-network for action selection
    - Target network for stable training
    """

    def __init__(self, predictor_checkpoint_path: str,
                 state_dim: int = 1920,
                 hidden_dim: int = 1024,
                 action_dim: int = 5):
        """
        Initialize trading agent.

        Args:
            predictor_checkpoint_path: Path to price predictor checkpoint
            state_dim: State dimension (predictor features + portfolio context)
            hidden_dim: Hidden dimension for Q-network
            action_dim: Number of actions
        """
        super().__init__()

        # Feature extractor (wraps predictor)
        self.feature_extractor = PredictorFeatureExtractor(predictor_checkpoint_path)

        # Q-network (main network for action selection)
        self.q_network = StockDQN(state_dim, hidden_dim, action_dim)

        # Target network (for stable Q-learning targets)
        self.target_network = StockDQN(state_dim, hidden_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Target network is always in eval mode
        self.target_network.eval()
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.action_dim = action_dim

        print(f"✅ TradingAgent initialized")
        print(f"   State dim: {state_dim}")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   Action dim: {action_dim}")

    def select_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: State tensor of shape (state_dim,)
            epsilon: Exploration rate (0 = greedy, 1 = random)

        Returns:
            Selected action index (0-4)
        """
        if random.random() < epsilon:
            # Explore: random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploit: use Q-network
            with torch.no_grad():
                q_values = self.q_network(state.unsqueeze(0))
                return q_values.argmax(dim=1).item()

    def select_actions_batch(self, states: Dict[str, torch.Tensor], epsilon: float = 0.0) -> Dict[str, int]:
        """
        Select actions for multiple stocks in batch.

        Args:
            states: Dictionary mapping ticker -> state tensor
            epsilon: Exploration rate

        Returns:
            Dictionary mapping ticker -> action index
        """
        actions = {}

        # Stack all states for batch inference
        tickers = list(states.keys())
        state_tensors = torch.stack([states[ticker] for ticker in tickers])

        # Get Q-values for all stocks at once
        with torch.no_grad():
            q_values_batch = self.q_network(state_tensors)  # (num_stocks, action_dim)

        # Select actions (epsilon-greedy)
        for i, ticker in enumerate(tickers):
            if random.random() < epsilon:
                actions[ticker] = random.randint(0, self.action_dim - 1)
            else:
                actions[ticker] = q_values_batch[i].argmax().item()

        return actions

    def update_target_network(self, tau: float = 0.005):
        """
        Soft update of target network: θ_target = τ*θ_q + (1-τ)*θ_target

        Args:
            tau: Soft update coefficient (0 = no update, 1 = hard update)
        """
        for target_param, q_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * q_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """
    Experience replay buffer for DQN.

    Stores transitions (s, a, r, s', done) and samples random minibatches
    for training. This decorrelates experiences and stabilizes learning.
    """

    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state: torch.Tensor, action: int, reward: float,
             next_state: torch.Tensor, done: bool, ticker: str = None,
             **kwargs):
        """
        Add a transition to the buffer.

        Args:
            state: Current state tensor
            action: Action taken
            reward: Reward received
            next_state: Next state tensor
            done: Whether episode ended
            ticker: Stock ticker (for debugging)
            **kwargs: Additional info to store
        """
        transition = {
            'state': state.cpu() if isinstance(state, torch.Tensor) else state,
            'action': action,
            'reward': reward,
            'next_state': next_state.cpu() if isinstance(next_state, torch.Tensor) else next_state,
            'done': done,
            'ticker': ticker,
            **kwargs
        }
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Dict]:
        """
        Sample random minibatch from buffer.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            List of transition dictionaries
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)

    def clear(self):
        """Clear all transitions from buffer."""
        self.buffer.clear()

    def get_stats(self) -> Dict:
        """Get statistics about the buffer contents."""
        if len(self.buffer) == 0:
            return {'size': 0, 'avg_reward': 0.0, 'capacity_used': 0.0}

        rewards = [t['reward'] for t in self.buffer]

        return {
            'size': len(self.buffer),
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'capacity_used': len(self.buffer) / self.capacity
        }


def compute_dqn_loss(agent: TradingAgent,
                     batch: List[Dict],
                     gamma: float = 0.99,
                     device: str = 'cuda') -> torch.Tensor:
    """
    Compute Double DQN loss for a batch of transitions.

    Double DQN reduces overestimation by:
    1. Selecting best action using main Q-network
    2. Evaluating action using target network

    Args:
        agent: TradingAgent instance
        batch: List of transition dictionaries
        gamma: Discount factor
        device: Device to compute on

    Returns:
        Loss tensor (scalar)
    """
    # Extract batch components
    states = torch.stack([t['state'] for t in batch]).to(device)
    actions = torch.tensor([t['action'] for t in batch], dtype=torch.long).unsqueeze(1).to(device)
    rewards = torch.tensor([t['reward'] for t in batch], dtype=torch.float32).to(device)
    next_states = torch.stack([t['next_state'] for t in batch]).to(device)
    dones = torch.tensor([t['done'] for t in batch], dtype=torch.float32).to(device)

    # Current Q-values: Q(s, a)
    current_q_values = agent.q_network(states).gather(1, actions).squeeze(1)

    # Target Q-values (Double DQN)
    with torch.no_grad():
        # Select best next action using main Q-network
        next_actions = agent.q_network(next_states).argmax(1, keepdim=True)

        # Evaluate selected action using target network
        next_q_values = agent.target_network(next_states).gather(1, next_actions).squeeze(1)

        # Compute target: r + γ * Q_target(s', argmax_a Q(s', a))
        target_q_values = rewards + gamma * next_q_values * (1 - dones)

    # Compute loss (MSE between current Q and target Q)
    loss = F.mse_loss(current_q_values, target_q_values)

    return loss


# Action definitions (for reference)
ACTIONS = {
    0: 'HOLD',
    1: 'BUY_SMALL',
    2: 'BUY_MEDIUM',
    3: 'BUY_LARGE',
    4: 'SELL'
}

# Inverse mapping
ACTION_NAMES_TO_IDS = {v: k for k, v in ACTIONS.items()}
