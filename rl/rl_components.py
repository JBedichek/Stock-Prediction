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

    def __init__(self, predictor_checkpoint_path: str, bin_edges_path: str = './data/adaptive_bin_edges.pt'):
        """
        Initialize feature extractor.

        Args:
            predictor_checkpoint_path: Path to trained price predictor checkpoint
            bin_edges_path: Path to adaptive bin edges for expected return calculation
        """
        super().__init__()

        # Load pretrained predictor
        print(f"Loading predictor from: {predictor_checkpoint_path}")
        checkpoint = torch.load(predictor_checkpoint_path, map_location='cpu')

        # Handle both direct model saves and checkpoint dicts
        if isinstance(checkpoint, dict):
            # Checkpoint dict format - need to reconstruct model
            if 'model' in checkpoint:
                self.predictor = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                # Need to reconstruct the model
                from training.train_new_format import SimpleTransformerPredictor

                config = checkpoint.get('config', {})
                state_dict = checkpoint['model_state_dict']

                # Infer input_dim from state dict
                if '_orig_mod.input_proj.0.weight' in state_dict:
                    input_dim = state_dict['_orig_mod.input_proj.0.weight'].shape[1]
                elif 'input_proj.0.weight' in state_dict:
                    input_dim = state_dict['input_proj.0.weight'].shape[1]
                else:
                    raise KeyError(f"Could not find input_proj layer in state_dict. Keys: {list(state_dict.keys())[:5]}")

                # Infer pred_mode from output size
                pred_head_keys = [k for k in state_dict.keys() if 'pred_head' in k and 'weight' in k]
                if pred_head_keys:
                    final_pred_key = sorted(pred_head_keys)[-1]
                    final_output_size = state_dict[final_pred_key].shape[0]
                    if final_output_size == 400:
                        pred_mode = 'classification'
                    elif final_output_size == 4:
                        pred_mode = 'regression'
                    else:
                        pred_mode = config.get('pred_mode', 'classification')
                else:
                    pred_mode = config.get('pred_mode', 'classification')

                # Create model
                self.predictor = SimpleTransformerPredictor(
                    input_dim=input_dim,
                    hidden_dim=config.get('hidden_dim', 1024),
                    num_layers=config.get('num_layers', 10),
                    num_heads=config.get('num_heads', 16),
                    dropout=config.get('dropout', 0.15),
                    num_pred_days=4,
                    pred_mode=pred_mode
                )

                # Strip '_orig_mod.' prefix if present (from torch.compile)
                if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
                    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

                # Load weights
                self.predictor.load_state_dict(state_dict, strict=False)
            else:
                raise ValueError("Checkpoint does not contain 'model' or 'model_state_dict'")
        else:
            # Direct model save
            self.predictor = checkpoint

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

        Returns:
            Feature tensor of shape (batch, 1944):
            - transformer_output: 1024 dims (mean-pooled transformer activations)
            - pred_logits: 400 dims (100 bins × 4 horizons) for classification
            - confidence: 4 dims (confidence per horizon)
            - expected_returns: 4 dims (weighted predictions)
            - raw_predictions: 4 dims (argmax bin per horizon)
            - entropy: 4 dims (distribution entropy per horizon)
            - input_stats: 4 dims (mean, std, min, max of input)
        """
        batch_size, seq_len, current_feature_dim = input_features.shape
        device = input_features.device

        # Pad or truncate to match model's expected input_dim
        model_input_dim = self.predictor.input_dim

        if current_feature_dim != model_input_dim:
            if current_feature_dim < model_input_dim:
                # Pad with zeros
                padding = torch.zeros(batch_size, seq_len, model_input_dim - current_feature_dim, device=device)
                input_features = torch.cat([input_features, padding], dim=2)
            else:
                # Truncate
                input_features = input_features[:, :, :model_input_dim]

        # Extract transformer activations using hook
        transformer_output = None

        def hook(module, input, output):
            nonlocal transformer_output
            transformer_output = output.mean(dim=1)  # Mean pool: (batch, hidden_dim)

        handle = self.predictor.transformer.register_forward_hook(hook)

        # Forward pass through predictor
        with torch.set_grad_enabled(self.predictor.training):
            pred, confidence = self.predictor(input_features)
            # pred: (batch, num_bins, num_pred_days) for classification
            # confidence: (batch, num_pred_days)

        handle.remove()

        # Extract features based on prediction mode
        if hasattr(self.predictor, 'pred_mode') and self.predictor.pred_mode == 'classification':
            # Flatten prediction logits
            pred_logits = pred.flatten(1)  # (batch, num_bins * num_pred_days)

            # Convert logits to probabilities
            prob_dist = F.softmax(pred, dim=1)  # (batch, num_bins, num_pred_days)

            # Compute entropy per horizon (uncertainty measure)
            entropy = -(prob_dist * torch.log(prob_dist + 1e-10)).sum(dim=1)  # (batch, 4)

            # Compute expected returns if bin_edges available
            if self.bin_edges is not None:
                if self.bin_edges.device != device:
                    self.bin_edges = self.bin_edges.to(device)

                bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
                bin_centers = bin_centers.unsqueeze(0).unsqueeze(2)  # (1, num_bins, 1)
                expected_returns = (prob_dist * bin_centers).sum(dim=1)  # (batch, 4)
            else:
                expected_returns = torch.zeros(batch_size, 4, device=device)

            # Get raw predictions (argmax bin index)
            raw_predictions = pred.argmax(dim=1).float()  # (batch, 4)

        else:
            # Regression mode
            pred_logits = pred  # (batch, 4)
            # Pad to match classification size
            pred_logits = F.pad(pred_logits, (0, 400 - 4))
            entropy = torch.zeros(batch_size, 4, device=device)
            expected_returns = pred  # Use regression predictions as expected returns
            raw_predictions = pred

        # Input statistics (simple features from input)
        input_stats = torch.stack([
            input_features.mean(),
            input_features.std(),
            input_features.min(),
            input_features.max()
        ]).unsqueeze(0).expand(batch_size, -1)  # (batch, 4)

        # Concatenate all features
        features = torch.cat([
            transformer_output,     # 1024 dims (hidden_dim)
            pred_logits,           # 400 dims (or padded regression)
            confidence,            # 4 dims
            expected_returns,      # 4 dims
            raw_predictions,       # 4 dims
            entropy,               # 4 dims
            input_stats            # 4 dims
        ], dim=1)  # Total: 1444 dims

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

    def __init__(self, state_dim: int = 1469, hidden_dim: int = 1024, action_dim: int = 5):
        """
        Initialize DQN.

        Args:
            state_dim: Input state dimension (1444 predictor features + 25 portfolio context = 1469)
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
                 state_dim: int = 1469,
                 hidden_dim: int = 1024,
                 action_dim: int = 5):
        """
        Initialize trading agent.

        Args:
            predictor_checkpoint_path: Path to price predictor checkpoint
            state_dim: State dimension (1444 predictor features + 25 portfolio context = 1469)
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
    # Ensure Q-network is in training mode
    agent.q_network.train()
    agent.target_network.eval()

    # Extract batch components
    # Detach states from any existing computation graph (they come from replay buffer)
    # But don't call requires_grad_(False) - let the Q-network handle gradient flow
    states = torch.stack([t['state'].detach() for t in batch]).to(device)
    actions = torch.tensor([t['action'] for t in batch], dtype=torch.long).unsqueeze(1).to(device)
    rewards = torch.tensor([t['reward'] for t in batch], dtype=torch.float32).to(device)
    next_states = torch.stack([t['next_state'].detach() for t in batch]).to(device)
    dones = torch.tensor([t['done'] for t in batch], dtype=torch.float32).to(device)

    # Current Q-values: Q(s, a)
    # This should have grad_fn because q_network parameters require grad
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


# ============================================================================
# ACTOR-CRITIC COMPONENTS (New Architecture)
# ============================================================================

class ActorNetwork(nn.Module):
    """
    Actor network for policy-based action selection.

    Maps state → probability distribution over actions.
    Used in actor-critic framework for learning optimal policy.
    """

    def __init__(self, state_dim: int = 5881, hidden_dim: int = 1024, action_dim: int = 5):
        """
        Initialize actor network.

        Args:
            state_dim: Input state dimension (4 stocks × 1469 + 5 position encoding = 5881)
            hidden_dim: Hidden layer dimension
            action_dim: Number of actions (5 = cash + 4 stocks)
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

        # Policy head: outputs action logits
        self.policy_head = nn.Sequential(
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
            Action logits of shape (batch, action_dim)
        """
        features = self.shared(state)
        logits = self.policy_head(features)
        return logits

    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get action probabilities (softmax of logits).

        Args:
            state: Input state tensor of shape (batch, state_dim)

        Returns:
            Action probabilities of shape (batch, action_dim)
        """
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        return probs

    def sample_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy distribution.

        Args:
            state: Input state tensor of shape (state_dim,)

        Returns:
            Tuple of (action_id, log_prob, entropy)
        """
        logits = self.forward(state.unsqueeze(0))  # (1, action_dim)
        probs = F.softmax(logits, dim=-1)

        # Sample from categorical distribution
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action.item(), log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Critic network for value estimation.

    Similar to StockDQN but optimized for actor-critic framework.
    Maps state → Q-values for all actions.
    """

    def __init__(self, state_dim: int = 5881, hidden_dim: int = 2048, action_dim: int = 5):
        """
        Initialize critic network.

        Args:
            state_dim: Input state dimension (4 stocks × 1469 + 5 position encoding = 5881)
            hidden_dim: Hidden layer dimension
            action_dim: Number of actions (5 = cash + 4 stocks)
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
            #nn.Linear(512, 512),
            #nn.LayerNorm(512),
            #nn.GELU(),
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
        features = self.shared(state)

        # Value and advantage streams
        value = self.value_stream(features)  # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, action_dim)

        # Dueling aggregation: Q(s,a) = V(s) + [A(s,a) - mean(A)]
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


class ActorCriticAgent(nn.Module):
    """
    Complete actor-critic trading agent with Twin Critics.

    Combines:
    - Feature extraction from price predictor
    - Actor network for policy learning
    - Twin critic networks for value estimation (prevents overestimation)
    - Twin target critics for stable training

    Twin Critics (TD3/SAC approach):
    - Two independent Q-networks (Q1, Q2)
    - Use min(Q1, Q2) for target computation
    - Eliminates positive feedback loop in Q-value updates
    """

    def __init__(self, predictor_checkpoint_path: str,
                 state_dim: int = 5881,
                 hidden_dim: int = 1024,
                 action_dim: int = 5):
        """
        Initialize actor-critic agent.

        Args:
            predictor_checkpoint_path: Path to price predictor checkpoint
            state_dim: State dimension (4 stocks × 1469 + 5 position encoding = 5881)
            hidden_dim: Hidden dimension for networks
            action_dim: Number of actions (5 = cash + 4 stocks)
        """
        super().__init__()

        # Feature extractor (wraps predictor)
        self.feature_extractor = PredictorFeatureExtractor(predictor_checkpoint_path)

        # Actor network (policy)
        self.actor = ActorNetwork(state_dim, hidden_dim, action_dim)

        # Twin critic networks (Q1, Q2) - prevents overestimation
        self.critic1 = CriticNetwork(state_dim, hidden_dim, action_dim)
        self.critic2 = CriticNetwork(state_dim, hidden_dim, action_dim)

        # Twin target critics (for stable Q-learning)
        self.target_critic1 = CriticNetwork(state_dim, hidden_dim, action_dim)
        self.target_critic2 = CriticNetwork(state_dim, hidden_dim, action_dim)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Target networks are always in eval mode
        self.target_critic1.eval()
        self.target_critic2.eval()
        for param in self.target_critic1.parameters():
            param.requires_grad = False
        for param in self.target_critic2.parameters():
            param.requires_grad = False

        self.action_dim = action_dim

        print(f"✅ ActorCriticAgent initialized (Twin Critics)")
        print(f"   State dim: {state_dim}")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   Action dim: {action_dim}")
        print(f"   Using clipped double Q-learning (TD3-style)")

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select action using actor policy.

        Args:
            state: State tensor of shape (state_dim,)
            deterministic: If True, select argmax action; if False, sample from distribution

        Returns:
            Tuple of (action_id, log_prob, entropy)
        """
        if deterministic:
            # Greedy action selection (for evaluation)
            with torch.no_grad():
                probs = self.actor.get_action_probs(state.unsqueeze(0))
                action = probs.argmax(dim=-1).item()
                return action, None, None
        else:
            # Sample from policy (for training)
            return self.actor.sample_action(state)

    def select_actions_batch(self, states: Dict[str, torch.Tensor], deterministic: bool = False) -> Dict[str, Tuple]:
        """
        Select actions for multiple stocks in batch.

        Args:
            states: Dictionary mapping ticker -> state tensor
            deterministic: If True, use greedy selection

        Returns:
            Dictionary mapping ticker -> (action, log_prob, entropy)
        """
        results = {}

        # Stack all states for batch inference
        tickers = list(states.keys())
        state_tensors = torch.stack([states[ticker] for ticker in tickers])

        if deterministic:
            # Greedy selection
            with torch.no_grad():
                probs = self.actor.get_action_probs(state_tensors)
                actions = probs.argmax(dim=-1)

            for i, ticker in enumerate(tickers):
                results[ticker] = (actions[i].item(), None, None)
        else:
            # Sample from policy
            logits = self.actor(state_tensors)
            probs = F.softmax(logits, dim=-1)

            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            entropies = dist.entropy()

            for i, ticker in enumerate(tickers):
                results[ticker] = (actions[i].item(), log_probs[i], entropies[i])

        return results

    def get_q_values(self, state: torch.Tensor, use_min: bool = False) -> torch.Tensor:
        """
        Get Q-values from critics.

        Args:
            state: State tensor
            use_min: If True, return min(Q1, Q2); if False, return Q1

        Returns:
            Q-values for all actions
        """
        q1 = self.critic1(state)
        if use_min:
            q2 = self.critic2(state)
            return torch.min(q1, q2)
        return q1

    def select_action_reduced(
        self,
        top_4_stocks: List[Tuple[str, int]],
        states: Dict[str, torch.Tensor],
        current_position: Optional[str],
        epsilon: float = 0.1,
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Select action using reduced action space (5 discrete actions).

        Args:
            top_4_stocks: List of (ticker, horizon_idx) for top 4 stocks
            states: Dict mapping ticker -> state tensor
            current_position: Currently held stock ticker (or None if in cash)
            epsilon: Exploration rate for epsilon-greedy
            deterministic: If True, use greedy (no epsilon)

        Returns:
            Tuple of (action, log_prob, entropy, trades)
        """
        from rl.reduced_action_space import (
            create_global_state,
            select_action_epsilon_greedy,
            decode_action_to_trades
        )

        # Create global state representation
        global_state = create_global_state(
            top_4_stocks, states, current_position, device=next(self.actor.parameters()).device
        )

        # Get action from actor
        logits = self.actor(global_state.unsqueeze(0)).squeeze(0)

        # Epsilon-greedy selection
        action, log_prob, entropy = select_action_epsilon_greedy(
            logits, epsilon=epsilon, training=not deterministic
        )

        # Decode action to trades
        trades = decode_action_to_trades(action, top_4_stocks, current_position)

        return action, log_prob, entropy, trades

    def select_actions_reduced_batch(
        self,
        top_4_stocks_list: List[List[Tuple[str, int]]],
        states_list: List[Dict[str, torch.Tensor]],
        positions_list: List[Optional[str]],
        epsilon: float = 0.1,
        deterministic: bool = False
    ) -> List[Tuple[int, torch.Tensor, torch.Tensor, List[Dict]]]:
        """
        Select actions for multiple environments in batch (vectorized).

        Args:
            top_4_stocks_list: List of N top_4_stocks
            states_list: List of N state dicts
            positions_list: List of N current positions
            epsilon: Exploration rate
            deterministic: If True, use greedy

        Returns:
            List of N (action, log_prob, entropy, trades) tuples
        """
        from rl.reduced_action_space import (
            create_global_state,
            select_action_epsilon_greedy,
            decode_action_to_trades
        )

        num_envs = len(top_4_stocks_list)
        device = next(self.actor.parameters()).device

        # Create global states for all environments
        global_states = []
        for i in range(num_envs):
            try:
                global_state = create_global_state(
                    top_4_stocks_list[i], states_list[i], positions_list[i], device=device
                )
                global_states.append(global_state)
            except (KeyError, IndexError, ValueError):
                # Fallback to zero state if construction fails
                global_states.append(torch.zeros(5881, device=device))

        # Batch forward pass (N, 5881) -> (N, 5)
        global_states_batch = torch.stack(global_states)
        logits_batch = self.actor(global_states_batch)

        # Select actions for each environment
        results = []
        for i in range(num_envs):
            logits = logits_batch[i]

            # Epsilon-greedy selection
            action, log_prob, entropy = select_action_epsilon_greedy(
                logits, epsilon=epsilon, training=not deterministic
            )

            # Decode action to trades
            try:
                trades = decode_action_to_trades(action, top_4_stocks_list[i], positions_list[i])
            except (KeyError, IndexError, ValueError):
                trades = []

            results.append((action, log_prob, entropy, trades))

        return results

    def update_target_critics(self, tau: float = 0.005):
        """
        Soft update of both target critics: θ_target = τ*θ + (1-τ)*θ_target

        Args:
            tau: Soft update coefficient (0 = no update, 1 = hard update)
        """
        # Update target critic 1
        for target_param, critic_param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * critic_param.data + (1.0 - tau) * target_param.data)

        # Update target critic 2
        for target_param, critic_param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * critic_param.data + (1.0 - tau) * target_param.data)


def compute_critic_loss_monte_carlo(agent: ActorCriticAgent,
                                   batch: List[Dict],
                                   device: str = 'cuda') -> torch.Tensor:
    """
    Compute critic loss using Monte Carlo returns (for offline pretraining).

    Instead of bootstrapping (TD learning), uses actual cumulative returns
    from complete episodes. This provides zero-bias targets.

    Q(s,a) should predict: G_t = r_t + γr_{t+1} + γ²r_{t+2} + ... + γ^T r_T

    Args:
        agent: ActorCriticAgent instance
        batch: List of transition dictionaries (must have 'mc_return' field)
        device: Device to compute on

    Returns:
        Loss tensor (scalar)
    """
    agent.critic.train()

    # Extract batch components
    states = torch.stack([t['state'].detach() for t in batch]).to(device)
    actions = torch.tensor([t['action'] for t in batch], dtype=torch.long).unsqueeze(1).to(device)
    mc_returns = torch.tensor([t['mc_return'] for t in batch], dtype=torch.float32).to(device)

    # Current Q-values: Q(s, a)
    current_q_values = agent.critic(states).gather(1, actions).squeeze(1)

    # Target is the actual Monte Carlo return (no bootstrapping!)
    target_q_values = mc_returns

    # MSE loss
    loss = F.mse_loss(current_q_values, target_q_values)

    return loss


def compute_critic_loss(agent: ActorCriticAgent,
                       batch: List[Dict],
                       gamma: float = 0.99,
                       device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute twin critic losses (Clipped Double Q-Learning from TD3).

    Twin Critics approach:
    1. Train both Q1 and Q2 independently
    2. For targets, use min(Q1_target, Q2_target) to prevent overestimation
    3. This eliminates the positive feedback loop in Q-value updates

    Args:
        agent: ActorCriticAgent instance
        batch: List of transition dictionaries
        gamma: Discount factor
        device: Device to compute on

    Returns:
        Tuple of (loss1, loss2) for both critics
    """
    agent.critic1.train()
    agent.critic2.train()
    agent.target_critic1.eval()
    agent.target_critic2.eval()

    # Extract batch components
    states = torch.stack([t['state'].detach() for t in batch]).to(device)
    actions = torch.tensor([t['action'] for t in batch], dtype=torch.long).unsqueeze(1).to(device)
    rewards = torch.tensor([t['reward'] for t in batch], dtype=torch.float32).to(device)
    next_states = torch.stack([t['next_state'].detach() for t in batch]).to(device)
    dones = torch.tensor([t['done'] for t in batch], dtype=torch.float32).to(device)

    # Current Q-values from both critics: Q1(s, a), Q2(s, a)
    current_q1_values = agent.critic1(states).gather(1, actions).squeeze(1)
    current_q2_values = agent.critic2(states).gather(1, actions).squeeze(1)

    # Target Q-values (Clipped Double Q-Learning)
    with torch.no_grad():
        # Select best next action using main critic1
        next_actions = agent.critic1(next_states).argmax(1, keepdim=True)

        # Evaluate using BOTH target critics and take MINIMUM (prevents overestimation)
        next_q1_values = agent.target_critic1(next_states).gather(1, next_actions).squeeze(1)
        next_q2_values = agent.target_critic2(next_states).gather(1, next_actions).squeeze(1)
        next_q_values = torch.min(next_q1_values, next_q2_values)  # KEY: Take minimum!

        # Compute target: r + γ * min(Q1_target, Q2_target)
        target_q_values = rewards + gamma * next_q_values * (1 - dones)

    # MSE loss for both critics (same target, prevents divergence)
    loss1 = F.mse_loss(current_q1_values, target_q_values)
    loss2 = F.mse_loss(current_q2_values, target_q_values)

    return loss1, loss2


def compute_actor_loss(agent: ActorCriticAgent,
                      batch: List[Dict],
                      device: str = 'cuda',
                      entropy_coef: float = 0.01,
                      action_diversity_coef: float = 0.0) -> Tuple[torch.Tensor, Dict]:
    """
    Compute actor loss for a batch of transitions.

    Uses policy gradient with Q-value as advantage:
    L = -E[log π(a|s) * Q(s,a)] - β*H(π)

    Args:
        agent: ActorCriticAgent instance
        batch: List of transition dictionaries
        device: Device to compute on
        entropy_coef: Coefficient for entropy regularization

    Returns:
        Tuple of (loss, info_dict)
    """
    agent.actor.train()
    agent.critic1.eval()  # Don't update critics when training actor
    agent.critic2.eval()

    # Extract batch components
    states = torch.stack([t['state'].detach() for t in batch]).to(device)
    actions = torch.tensor([t['action'] for t in batch], dtype=torch.long).to(device)

    # Get current policy distribution
    logits = agent.actor(states)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)

    # Log probabilities of actions taken
    log_probs = dist.log_prob(actions)

    # Entropy for exploration
    entropy = dist.entropy()

    # Q-values as advantages (detach to prevent critic gradient flow)
    # Use min(Q1, Q2) for conservative advantage estimates
    with torch.no_grad():
        q1_values = agent.critic1(states)
        q2_values = agent.critic2(states)
        q_values = torch.min(q1_values, q2_values)  # Conservative Q-values

        action_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Use policy-weighted baseline: V(s) = E_{a~π}[Q(s,a)]
        # This is the correct baseline for actor-critic (reduces variance more than uniform mean)
        baseline = (probs * q_values).sum(dim=1)
        advantages = action_q_values - baseline

    # Policy gradient loss: maximize advantage weighted by log_prob
    # Negative because we minimize loss
    policy_loss = -(log_probs * advantages).mean()

    # Entropy regularization (encourage exploration)
    entropy_loss = -entropy.mean()

    # Action diversity regularization (discourage always choosing action 0/HOLD)
    # Compute action distribution in batch (for logging)
    action_diversity_loss_batch = 0.0
    if action_diversity_coef > 0:
        # Method 1: Penalize imbalanced BATCH actions (lagging indicator)
        action_counts = torch.bincount(actions, minlength=agent.action_dim).float()
        action_probs_empirical = action_counts / len(actions)
        uniform_dist = torch.ones(agent.action_dim, device=device) / agent.action_dim
        action_probs_empirical = action_probs_empirical + 1e-8
        kl_div_batch = (action_probs_empirical * (torch.log(action_probs_empirical) - torch.log(uniform_dist))).sum()
        action_diversity_loss_batch = kl_div_batch

        # Method 2: Penalize policy probabilities that concentrate on one action (leading indicator)
        # This directly affects the policy output, not just the batch
        # Compute average probability mass on action 0 (HOLD) across the batch
        probs_action_0 = probs[:, 0]  # Probability of action 0 for each state
        # Penalize if action 0 has high probability on average
        # Target: action 0 should have prob ~0.2 (uniform over 5 actions)
        target_prob = 1.0 / agent.action_dim
        action0_penalty = torch.relu(probs_action_0.mean() - target_prob * 1.5)  # Penalty if prob > 0.3

        # Combine both methods
        action_diversity_loss = action_diversity_loss_batch + action0_penalty * 10.0  # Scale up the direct penalty

    else:
        action_diversity_loss = 0.0
        probs_action_0 = probs[:, 0]  # Still compute for logging

    # Total loss
    loss = policy_loss + entropy_coef * entropy_loss + action_diversity_coef * action_diversity_loss

    # Info for logging
    info = {
        'policy_loss': policy_loss.item(),
        'entropy': entropy.mean().item(),
        'action_diversity_loss': action_diversity_loss.item() if isinstance(action_diversity_loss, torch.Tensor) else action_diversity_loss,
        'action0_prob': probs_action_0.mean().item(),
        'avg_advantage': advantages.mean().item(),
        'std_advantage': advantages.std().item(),
        'avg_q_value': action_q_values.mean().item(),
        'avg_baseline': baseline.mean().item(),
        'avg_log_prob': log_probs.mean().item()
    }

    return loss, info


# Action definitions (for reference)
ACTIONS = {
    0: 'HOLD',
    1: 'BUY_SMALL',
    2: 'BUY_MEDIUM',
    3: 'BUY_LARGE',
    4: 'SELL'
}

# New simplified actions (for actor-critic)
ACTIONS_AC = {
    0: 'HOLD',
    1: 'BUY',
    2: 'SELL'
}

# Inverse mapping
ACTION_NAMES_TO_IDS = {v: k for k, v in ACTIONS.items()}
ACTION_NAMES_TO_IDS_AC = {v: k for k, v in ACTIONS_AC.items()}
