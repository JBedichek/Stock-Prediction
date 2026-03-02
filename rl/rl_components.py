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
    Experience replay buffer for DQN with n-step returns support.

    Stores transitions (s, a, r, s', done) and samples random minibatches
    for training. This decorrelates experiences and stabilizes learning.

    Supports n-step returns for improved credit assignment:
    - Uses sum of discounted rewards over n steps
    - Bootstraps from value estimate n steps in the future
    - Common n values: 3-5
    """

    def __init__(self, capacity: int = 100000, n_step: int = 1, gamma: float = 0.99,
                 use_per: bool = False, per_alpha: float = 0.6, per_beta: float = 0.4,
                 per_beta_increment: float = 0.001, per_epsilon: float = 1e-6):
        """
        Initialize replay buffer with optional Prioritized Experience Replay (PER).

        Args:
            capacity: Maximum number of transitions to store
            n_step: Number of steps for n-step returns (1 = standard TD)
            gamma: Discount factor for n-step computation
            use_per: Whether to use Prioritized Experience Replay
            per_alpha: Prioritization exponent (0=uniform, 1=fully prioritized)
            per_beta: Importance sampling exponent (0=no correction, 1=full correction)
            per_beta_increment: Increment for beta per sample (anneal to 1.0)
            per_epsilon: Small constant to ensure non-zero priorities
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma

        # Prioritized Experience Replay (PER)
        self.use_per = use_per
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.per_beta_increment = per_beta_increment
        self.per_epsilon = per_epsilon
        self.priorities = deque(maxlen=capacity)  # Store priorities for each transition
        self.max_priority = 1.0  # Track maximum priority seen

        # Temporary buffer for computing n-step returns
        # Stores recent transitions until we have n steps to compute return
        self.n_step_buffer = deque(maxlen=n_step)

    def push(self, state: torch.Tensor, action: int, reward: float,
             next_state: torch.Tensor, done: bool, ticker: str = None,
             **kwargs):
        """
        Add a transition to the buffer with n-step return computation.

        For n-step > 1:
        - Accumulates transitions in temporary buffer
        - Computes n-step return when buffer is full or episode ends
        - Stores transition with n-step reward and n-step-ahead next_state

        Args:
            state: Current state tensor
            action: Action taken
            reward: Reward received
            next_state: Next state tensor
            done: Whether episode ended
            ticker: Stock ticker (for debugging)
            **kwargs: Additional info to store
        """
        # Store transition in temporary n-step buffer
        transition = {
            'state': state.cpu() if isinstance(state, torch.Tensor) else state,
            'action': action,
            'reward': reward,
            'next_state': next_state.cpu() if isinstance(next_state, torch.Tensor) else next_state,
            'done': done,
            'ticker': ticker,
            **kwargs
        }
        self.n_step_buffer.append(transition)

        # If n-step buffer is full OR episode ended, compute n-step return and add to main buffer
        if len(self.n_step_buffer) == self.n_step or done:
            # Compute n-step return: R_t = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1}
            n_step_reward = 0.0
            n_step_done = False

            for i, trans in enumerate(self.n_step_buffer):
                n_step_reward += (self.gamma ** i) * trans['reward']
                # Episode ends if any intermediate step has done=True
                if trans['done']:
                    n_step_done = True

            # Create n-step transition
            # - state: first state in n-step buffer
            # - action: first action in n-step buffer
            # - reward: n-step accumulated reward
            # - next_state: last next_state in n-step buffer (n steps ahead)
            # - done: True if episode ended within n steps
            first_trans = self.n_step_buffer[0]
            last_trans = self.n_step_buffer[-1]

            n_step_transition = {
                'state': first_trans['state'],
                'action': first_trans['action'],
                'reward': n_step_reward,
                'next_state': last_trans['next_state'],
                'done': n_step_done,
                'ticker': first_trans['ticker'],
                # Also store original single-step info for compatibility
                'portfolio_value': first_trans.get('portfolio_value', None)
            }

            # Copy over any additional kwargs (e.g., mc_return for Monte Carlo training)
            for key in first_trans:
                if key not in n_step_transition:
                    n_step_transition[key] = first_trans[key]

            self.buffer.append(n_step_transition)

            # Prioritized Experience Replay: assign initial priority (max priority)
            # New transitions get max priority to ensure they're sampled at least once
            if self.use_per:
                self.priorities.append(self.max_priority)

            # If episode ended, clear n-step buffer to start fresh for next episode
            if done:
                self.n_step_buffer.clear()

    def sample(self, batch_size: int) -> List[Dict]:
        """
        Sample minibatch from buffer (uniform or priority-based).

        For PER:
        - Samples based on priority: P(i) = p_i^α / Σ p_k^α
        - Returns importance sampling weights: w_i = (1/N * 1/P(i))^β
        - Anneals beta towards 1.0 over time

        Args:
            batch_size: Number of transitions to sample

        Returns:
            List of transition dictionaries (includes 'weight' and 'index' for PER)
        """
        batch_size = min(batch_size, len(self.buffer))

        if not self.use_per:
            # Uniform random sampling
            return random.sample(self.buffer, batch_size)

        # Prioritized sampling
        priorities_array = np.array(self.priorities, dtype=np.float32)

        # Validate priorities (check for NaN/inf)
        num_nonfinite = np.sum(~np.isfinite(priorities_array))
        if num_nonfinite > 0:
            # Only warn once every 100 occurrences to avoid spam
            if not hasattr(self, '_nan_warning_count'):
                self._nan_warning_count = 0
            self._nan_warning_count += 1
            if self._nan_warning_count % 100 == 1:
                print(f"⚠️ WARNING: Found {num_nonfinite}/{len(priorities_array)} non-finite priorities! (shown every 100 occurrences)")
            priorities_array = np.where(np.isfinite(priorities_array), priorities_array, self.per_epsilon)

        # Compute sampling probabilities: P(i) = p_i^α / Σ p_k^α
        priorities_alpha = priorities_array ** self.per_alpha
        sum_priorities = priorities_alpha.sum()

        # Additional safety check
        if not np.isfinite(sum_priorities) or sum_priorities == 0:
            print(f"WARNING: Sum of priorities is {sum_priorities}! Falling back to uniform sampling.")
            probabilities = np.ones(len(priorities_array)) / len(priorities_array)
        else:
            probabilities = priorities_alpha / sum_priorities

        # Sample indices based on probabilities
        # Use replace=True when batch size is close to buffer size (avoids numerical issues)
        use_replacement = (batch_size > len(self.buffer) * 0.9)
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probabilities, replace=use_replacement)

        # Compute importance sampling weights: w_i = (1/N * 1/P(i))^β
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.per_beta)
        # Normalize weights by max weight for stability
        weights = weights / weights.max()

        # Anneal beta towards 1.0
        self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)

        # Return transitions with weights and indices
        batch = []
        for idx, weight in zip(indices, weights):
            transition = self.buffer[idx].copy()
            transition['weight'] = weight
            transition['index'] = idx
            batch.append(transition)

        return batch

    def update_priorities(self, indices: List[int], td_errors: np.ndarray,
                           use_curriculum: bool = True, curriculum_percentiles: Tuple[float, float] = (0.3, 0.95)):
        """
        Update priorities for sampled transitions based on TD errors with curriculum learning.

        Curriculum learning focuses on medium-difficulty samples:
        - Too easy (small TD error): Already learned, low priority
        - Medium difficulty: Informative, high priority
        - Too hard / outliers (huge TD error): Likely noise, moderate priority

        Args:
            indices: List of transition indices that were sampled
            td_errors: TD errors for the sampled transitions (|r + γV(s') - V(s)|)
            use_curriculum: Whether to apply curriculum learning (clip extreme TD errors)
            curriculum_percentiles: (lower, upper) percentiles for clipping TD errors
        """
        if not self.use_per:
            return

        # Convert to numpy for easier computation
        td_errors_array = np.abs(td_errors)

        # Validate TD errors (check for NaN/inf)
        if not np.all(np.isfinite(td_errors_array)):
            print(f"WARNING: Found {np.sum(~np.isfinite(td_errors_array))} non-finite TD errors! Replacing with max finite value.")
            finite_mask = np.isfinite(td_errors_array)
            if np.any(finite_mask):
                max_finite = np.max(td_errors_array[finite_mask])
                td_errors_array = np.where(finite_mask, td_errors_array, max_finite)
            else:
                # All TD errors are non-finite, use epsilon
                td_errors_array = np.full_like(td_errors_array, self.per_epsilon)

        if use_curriculum and len(self.buffer) > 100:
            # Compute percentiles of all TD errors in buffer for curriculum learning
            all_priorities = np.array(list(self.priorities), dtype=np.float32)
            lower_bound = np.percentile(all_priorities, curriculum_percentiles[0] * 100)
            upper_bound = np.percentile(all_priorities, curriculum_percentiles[1] * 100)

            # Clip TD errors to focus on medium-difficulty samples
            # This prevents outliers from dominating and ensures we learn from informative samples
            td_errors_clipped = np.clip(td_errors_array, lower_bound, upper_bound)
        else:
            td_errors_clipped = td_errors_array

        for idx, td_error in zip(indices, td_errors_clipped):
            # Priority = |TD error| + ε
            priority = abs(td_error) + self.per_epsilon
            self.priorities[idx] = priority

            # Update max priority
            if priority > self.max_priority:
                self.max_priority = priority

    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)

    def clear(self):
        """Clear all transitions from buffer."""
        self.buffer.clear()
        if self.use_per:
            self.priorities.clear()
            self.max_priority = 1.0

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

    def add_hindsight_experiences(self, episode_transitions: List[Dict],
                                   hindsight_ratio: float = 0.5):
        """
        Add hindsight experiences from failed episodes (HER for trading).

        Hindsight Experience Replay helps the agent learn from failures by
        relabeling goals. For trading:
        - Losing trades → Learn that HOLD would have been better (avoid loss)
        - Early exits → Learn from actual future returns (maximize gains)

        Args:
            episode_transitions: List of transitions from a completed episode
            hindsight_ratio: Fraction of negative-reward transitions to relabel
        """
        if len(episode_transitions) == 0:
            return

        for trans in episode_transitions:
            reward = trans['reward']

            # HER Strategy 1: Failed trades (negative reward)
            # Create synthetic experience where HOLD would have been better
            if reward < -0.01 and random.random() < hindsight_ratio:
                # Create hindsight transition: HOLD instead of the taken action
                hindsight_trans = trans.copy()
                hindsight_trans['action'] = 0  # Action 0 = HOLD
                # Reward for HOLD = 0 (avoided the loss)
                hindsight_trans['reward'] = 0.0
                # Add to buffer directly (bypass n-step computation)
                self.buffer.append(hindsight_trans)

                if self.use_per:
                    # Give high priority to hindsight experiences (important for learning)
                    self.priorities.append(self.max_priority * 1.5)
                    self.max_priority = max(self.max_priority, self.max_priority * 1.5)

        # HER Strategy 2: Consecutive transitions with improving returns
        # If we took action A at step t and got reward R1, but staying in the trade
        # at step t+1 gave reward R2 > R1, create hindsight experience with R2
        for i in range(len(episode_transitions) - 1):
            current_trans = episode_transitions[i]
            next_trans = episode_transitions[i + 1]

            current_reward = current_trans['reward']
            next_reward = next_trans['reward']

            # If the next step had better reward, create hindsight with that reward
            if next_reward > current_reward and current_reward > 0 and random.random() < hindsight_ratio * 0.5:
                hindsight_trans = current_trans.copy()
                # Keep the same action, but use the better future reward
                hindsight_trans['reward'] = next_reward
                # Use the next_state from the future step
                hindsight_trans['next_state'] = next_trans['next_state']

                self.buffer.append(hindsight_trans)

                if self.use_per:
                    # Medium priority for these experiences
                    self.priorities.append(self.max_priority)



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

    def __init__(self, state_dim: int = 11761, hidden_dim: int = 1024, action_dim: int = 9):
        """
        Initialize actor network WITH SHORT SELLING.

        Args:
            state_dim: Input state dimension (8 stocks × 1469 + 9 position encoding = 11761)
                      [4 long stocks + 4 short stocks + position encoding]
            hidden_dim: Hidden layer dimension
            action_dim: Number of actions (9 = hold + 4 longs + 4 shorts)
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

    def __init__(self, state_dim: int = 11761, hidden_dim: int = 2048, action_dim: int = 9):
        """
        Initialize critic network WITH SHORT SELLING.

        Args:
            state_dim: Input state dimension (8 stocks × 1469 + 9 position encoding = 11761)
                      [4 long stocks + 4 short stocks + position encoding]
            hidden_dim: Hidden layer dimension
            action_dim: Number of actions (9 = hold + 4 longs + 4 shorts)
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


class DistributionalCritic(nn.Module):
    """
    Distributional Critic Network - outputs Q-value distributions (mean, std).

    Instead of point estimates, models full distribution of returns.
    This enables risk-aware decision making and better uncertainty quantification.

    Outputs: [Q_mean, Q_std] for each action
    Benefits:
    - Risk-aware action selection (use confidence bounds)
    - Better exploration (high uncertainty → explore)
    - Robust to noise (models uncertainty explicitly)
    """

    def __init__(self, state_dim: int = 11761, hidden_dim: int = 2048, action_dim: int = 9):
        """
        Initialize distributional critic network.

        Args:
            state_dim: Input state dimension
            hidden_dim: Hidden layer dimension
            action_dim: Number of actions
        """
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Shared feature extractor (same as regular critic)
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

        # Mean head: V(s) + A(s,a) dueling architecture
        self.value_stream_mean = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

        self.advantage_stream_mean = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, action_dim)
        )

        # Std head: models uncertainty (always positive)
        self.value_stream_std = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Softplus()  # Ensure positive
        )

        self.advantage_stream_std = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, action_dim),
            nn.Softplus()  # Ensure positive
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through network.

        Args:
            state: Input state tensor of shape (batch, state_dim)

        Returns:
            Tuple of (q_mean, q_std) each of shape (batch, action_dim)
        """
        features = self.shared(state)

        # Mean Q-values (dueling)
        value_mean = self.value_stream_mean(features)
        advantage_mean = self.advantage_stream_mean(features)
        q_mean = value_mean + (advantage_mean - advantage_mean.mean(dim=1, keepdim=True))

        # Std Q-values (dueling) - represents uncertainty
        value_std = self.value_stream_std(features)
        advantage_std = self.advantage_stream_std(features)
        q_std = value_std + (advantage_std - advantage_std.mean(dim=1, keepdim=True)) + 1e-6  # Small epsilon

        return q_mean, q_std

    def sample_q(self, state: torch.Tensor, num_samples: int = 10) -> torch.Tensor:
        """
        Sample from Q-distribution using reparameterization trick.

        Args:
            state: Input state
            num_samples: Number of samples to draw

        Returns:
            Q-samples of shape (num_samples, batch, action_dim)
        """
        q_mean, q_std = self.forward(state)
        # Sample Q ~ N(mean, std) using reparameterization
        epsilon = torch.randn(num_samples, *q_mean.shape, device=q_mean.device)
        q_samples = q_mean.unsqueeze(0) + epsilon * q_std.unsqueeze(0)
        return q_samples

    def get_conservative_q(self, state: torch.Tensor, confidence: float = 0.9) -> torch.Tensor:
        """
        Get lower confidence bound for risk-averse action selection.

        Args:
            state: Input state
            confidence: Confidence level (0.9 = 90% confidence)

        Returns:
            Conservative Q-values (lower bound)
        """
        q_mean, q_std = self.forward(state)
        z_score = 1.28 if confidence == 0.9 else 1.96  # 90% or 95%
        return q_mean - z_score * q_std

    def get_optimistic_q(self, state: torch.Tensor, confidence: float = 0.9) -> torch.Tensor:
        """
        Get upper confidence bound for exploration.

        Args:
            state: Input state
            confidence: Confidence level

        Returns:
            Optimistic Q-values (upper bound)
        """
        q_mean, q_std = self.forward(state)
        z_score = 1.28 if confidence == 0.9 else 1.96
        return q_mean + z_score * q_std

    def get_risk_metrics(self, state: torch.Tensor, action: int) -> Dict[str, float]:
        """
        Compute risk metrics for a specific action.

        Args:
            state: Input state
            action: Action index

        Returns:
            Dictionary of risk metrics
        """
        q_mean, q_std = self.forward(state)

        return {
            'expected_return': q_mean[0, action].item(),
            'uncertainty': q_std[0, action].item(),
            'sharpe_proxy': (q_mean[0, action] / (q_std[0, action] + 1e-6)).item(),
            'lcb_90': (q_mean[0, action] - 1.28 * q_std[0, action]).item(),
            'ucb_90': (q_mean[0, action] + 1.28 * q_std[0, action]).item()
        }


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
        bottom_4_stocks: List[Tuple[str, int]],
        states: Dict[str, torch.Tensor],
        current_position: Optional[str],
        current_is_short: bool = False,
        epsilon: float = 0.1,
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor, torch.Tensor, List[Dict], Optional[str], bool]:
        """
        Select action using reduced action space (9 discrete actions) WITH SHORT SELLING.

        Args:
            top_4_stocks: List of (ticker, horizon_idx) for top 4 stocks to LONG
            bottom_4_stocks: List of (ticker, horizon_idx) for bottom 4 stocks to SHORT
            states: Dict mapping ticker -> state tensor
            current_position: Currently held stock ticker (or None if in cash)
            current_is_short: True if current position is a short, False if long
            epsilon: Exploration rate for epsilon-greedy
            deterministic: If True, use greedy (no epsilon)

        Returns:
            Tuple of (action, log_prob, entropy, trades, new_position, new_is_short)
        """
        from rl.reduced_action_space import (
            create_global_state,
            select_action_epsilon_greedy,
            decode_action_to_trades
        )

        # Create global state representation
        global_state = create_global_state(
            top_4_stocks, bottom_4_stocks, states, current_position,
            is_short=current_is_short, device=next(self.actor.parameters()).device
        )

        # Get action from actor
        logits = self.actor(global_state.unsqueeze(0)).squeeze(0)

        # Epsilon-greedy selection
        action, log_prob, entropy = select_action_epsilon_greedy(
            logits, epsilon=epsilon, training=not deterministic
        )

        # Decode action to trades (returns trades, new_position, new_is_short)
        trades, new_position, new_is_short = decode_action_to_trades(
            action, top_4_stocks, bottom_4_stocks, current_position, current_is_short
        )

        return action, log_prob, entropy, trades, new_position, new_is_short

    def select_actions_reduced_batch(
        self,
        top_4_stocks_list: List[List[Tuple[str, int]]],
        bottom_4_stocks_list: List[List[Tuple[str, int]]],
        states_list: List[Dict[str, torch.Tensor]],
        positions_list: List[Optional[str]],
        is_short_list: List[bool],
        epsilon: float = 0.1,
        deterministic: bool = False
    ) -> List[Tuple[int, torch.Tensor, torch.Tensor, List[Dict], Optional[str], bool]]:
        """
        Select actions for multiple environments in batch (vectorized) WITH SHORT SELLING.

        Args:
            top_4_stocks_list: List of N top_4_stocks (for LONG)
            bottom_4_stocks_list: List of N bottom_4_stocks (for SHORT)
            states_list: List of N state dicts
            positions_list: List of N current positions
            is_short_list: List of N bools indicating if position is short
            epsilon: Exploration rate
            deterministic: If True, use greedy

        Returns:
            List of N (action, log_prob, entropy, trades, new_position, new_is_short) tuples
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
                    top_4_stocks_list[i], bottom_4_stocks_list[i], states_list[i],
                    positions_list[i], is_short=is_short_list[i], device=device
                )
                global_states.append(global_state)
            except (KeyError, IndexError, ValueError):
                # Fallback to zero state if construction fails
                global_states.append(torch.zeros(11761, device=device))

        # Batch forward pass (N, 11761) -> (N, 9)
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

            # Decode action to trades (returns trades, new_position, new_is_short)
            try:
                trades, new_position, new_is_short = decode_action_to_trades(
                    action, top_4_stocks_list[i], bottom_4_stocks_list[i],
                    positions_list[i], is_short_list[i]
                )
            except (KeyError, IndexError, ValueError):
                trades = []
                new_position = positions_list[i]
                new_is_short = is_short_list[i]

            results.append((action, log_prob, entropy, trades, new_position, new_is_short))

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
    Compute twin critic loss using Monte Carlo returns (for offline pretraining).

    Instead of bootstrapping (TD learning), uses actual cumulative returns
    from complete episodes. This provides zero-bias targets.

    Q(s,a) should predict: G_t = r_t + γr_{t+1} + γ²r_{t+2} + ... + γ^T r_T

    Trains both critics independently with the same MC returns target.

    Args:
        agent: ActorCriticAgent instance (with twin critics)
        batch: List of transition dictionaries (must have 'mc_return' field)
        device: Device to compute on

    Returns:
        Loss tensor (scalar) - sum of both critic losses
    """
    agent.critic1.train()
    agent.critic2.train()

    # Extract batch components
    states = torch.stack([t['state'].detach() for t in batch]).to(device)
    actions = torch.tensor([t['action'] for t in batch], dtype=torch.long).unsqueeze(1).to(device)
    mc_returns = torch.tensor([t['mc_return'] for t in batch], dtype=torch.float32).to(device)

    # Current Q-values from both critics: Q1(s, a) and Q2(s, a)
    current_q1_values = agent.critic1(states).gather(1, actions).squeeze(1)
    current_q2_values = agent.critic2(states).gather(1, actions).squeeze(1)

    # Target is the actual Monte Carlo return (no bootstrapping!)
    target_q_values = mc_returns

    # MSE loss for both critics
    loss1 = F.mse_loss(current_q1_values, target_q_values)
    loss2 = F.mse_loss(current_q2_values, target_q_values)

    # Total loss (sum of both)
    loss = loss1 + loss2

    return loss


def compute_critic_loss(agent: ActorCriticAgent,
                       batch: List[Dict],
                       gamma: float = 0.99,
                       device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Compute twin critic losses (Clipped Double Q-Learning from TD3) with PER support.

    Twin Critics approach:
    1. Train both Q1 and Q2 independently
    2. For targets, use min(Q1_target, Q2_target) to prevent overestimation
    3. This eliminates the positive feedback loop in Q-value updates

    PER support:
    - Uses importance sampling weights to correct for biased sampling
    - Returns TD errors for priority updates

    Args:
        agent: ActorCriticAgent instance
        batch: List of transition dictionaries (may include 'weight' for PER)
        gamma: Discount factor
        device: Device to compute on

    Returns:
        Tuple of (loss1, loss2, td_errors) for both critics and priority updates
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

    # Extract PER weights if available (default to 1.0 for uniform sampling)
    weights = torch.tensor([t.get('weight', 1.0) for t in batch], dtype=torch.float32).to(device)

    # DEBUG: Check for NaN in input data (only warn once to avoid spam)
    if not hasattr(compute_critic_loss, '_input_nan_warned'):
        compute_critic_loss._input_nan_warned = False
    if not compute_critic_loss._input_nan_warned:
        if torch.any(~torch.isfinite(states)):
            print(f"🔴 DEBUG: States contain {torch.sum(~torch.isfinite(states))} non-finite values!")
            compute_critic_loss._input_nan_warned = True
        if torch.any(~torch.isfinite(next_states)):
            print(f"🔴 DEBUG: Next states contain {torch.sum(~torch.isfinite(next_states))} non-finite values!")
            compute_critic_loss._input_nan_warned = True
        if torch.any(~torch.isfinite(rewards)):
            print(f"🔴 DEBUG: Rewards contain {torch.sum(~torch.isfinite(rewards))} non-finite values!")
            compute_critic_loss._input_nan_warned = True

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

    # Compute TD errors for PER priority updates
    # Use Q1 for TD error (could also use min or mean of Q1 and Q2)
    td_errors = (target_q_values - current_q1_values).detach().cpu().numpy()

    # DEBUG: Check for NaN in Q-values and TD errors
    if not np.all(np.isfinite(td_errors)):
        num_nan = np.sum(~np.isfinite(td_errors))
        print(f"\n🔴 DEBUG: {num_nan}/{len(td_errors)} TD errors are non-finite!")
        print(f"   Current Q1 range: [{current_q1_values.min():.2f}, {current_q1_values.max():.2f}]")
        print(f"   Target Q range: [{target_q_values.min():.2f}, {target_q_values.max():.2f}]")
        print(f"   Rewards range: [{rewards.min():.4f}, {rewards.max():.4f}]")
        print(f"   Next Q range: [{next_q_values.min():.2f}, {next_q_values.max():.2f}]")
        # Check for NaN sources
        if torch.any(~torch.isfinite(current_q1_values)):
            print(f"   ⚠️ Current Q1 has {torch.sum(~torch.isfinite(current_q1_values))} non-finite values")
        if torch.any(~torch.isfinite(target_q_values)):
            print(f"   ⚠️ Target Q has {torch.sum(~torch.isfinite(target_q_values))} non-finite values")
        if torch.any(~torch.isfinite(rewards)):
            print(f"   ⚠️ Rewards has {torch.sum(~torch.isfinite(rewards))} non-finite values")
        if torch.any(~torch.isfinite(next_q_values)):
            print(f"   ⚠️ Next Q has {torch.sum(~torch.isfinite(next_q_values))} non-finite values")

    # Weighted MSE loss for both critics (apply importance sampling weights)
    # weights are already normalized in the buffer
    loss1 = (weights * F.mse_loss(current_q1_values, target_q_values, reduction='none')).mean()
    loss2 = (weights * F.mse_loss(current_q2_values, target_q_values, reduction='none')).mean()

    return loss1, loss2, td_errors


def compute_actor_loss(agent: ActorCriticAgent,
                      batch: List[Dict],
                      device: str = 'cuda',
                      entropy_coef: float = 0.01,
                      action_diversity_coef: float = 0.0,
                      gamma: float = 0.99) -> Tuple[torch.Tensor, Dict]:
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

    # Compute GAE-style advantages using n-step returns
    # For n-step returns: A(s,a) = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n V(s_{t+n}) - V(s_t)
    # This provides better bias-variance tradeoff than 1-step TD
    with torch.no_grad():
        # Get Q-values from both critics
        q1_values = agent.critic1(states)
        q2_values = agent.critic2(states)
        q_values = torch.min(q1_values, q2_values)  # Conservative Q-values

        # Extract V(s) from dueling architecture
        # V(s) = E_{a~π}[Q(s,a)] under current policy
        baseline = (probs * q_values).sum(dim=1)

        # Get Q-values for actions taken (for logging)
        action_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get next states and rewards for GAE computation
        next_states = torch.stack([t['next_state'].detach() for t in batch]).to(device)
        rewards = torch.tensor([t['reward'] for t in batch], dtype=torch.float32).to(device)
        dones = torch.tensor([t['done'] for t in batch], dtype=torch.float32).to(device)

        # Compute V(s') for next states
        q1_next = agent.critic1(next_states)
        q2_next = agent.critic2(next_states)
        q_next = torch.min(q1_next, q2_next)
        # Recompute probs for next states to get V(s')
        next_logits = agent.actor(next_states)
        next_probs = F.softmax(next_logits, dim=-1)
        next_baseline = (next_probs * q_next).sum(dim=1)

        # GAE-style advantage: A = r + γV(s') - V(s)
        # For n-step returns, the reward already includes multi-step accumulation
        # and next_state is n steps ahead, so this naturally extends to n-step GAE
        advantages = rewards + gamma * next_baseline * (1 - dones) - baseline

        # Normalize advantages for more stable gradients (prevent large advantage values from dominating)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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


def compute_cql_loss(agent: ActorCriticAgent,
                     batch: List[Dict],
                     device: str = 'cuda',
                     gamma: float = 0.99,
                     tau: float = 0.005,
                     cql_alpha: float = 1.0) -> Tuple[torch.Tensor, Dict]:
    """
    Compute Conservative Q-Learning (CQL) loss for critic networks.

    CQL prevents Q-value overestimation by penalizing high Q-values on unseen actions.
    This is especially important in noisy environments where overestimation is common.

    Loss = TD_loss + α * CQL_penalty
    where CQL_penalty = E[log Σ exp(Q(s,a'))] - E[Q(s,a)]

    Args:
        agent: ActorCriticAgent instance
        batch: List of transition dictionaries
        device: Device to compute on
        gamma: Discount factor
        tau: Temperature for CQL penalty
        cql_alpha: Weight of CQL penalty

    Returns:
        Tuple of (loss, info_dict)
    """
    agent.critic1.train()
    agent.critic2.train()

    # Extract batch components
    states = torch.stack([t['state'] for t in batch]).to(device)
    actions = torch.tensor([t['action'] for t in batch], dtype=torch.long).to(device)
    rewards = torch.tensor([t['reward'] for t in batch], dtype=torch.float32).to(device)
    next_states = torch.stack([t['next_state'] for t in batch]).to(device)
    dones = torch.tensor([t['done'] for t in batch], dtype=torch.float32).to(device)

    # Current Q-values for both critics
    q1_values = agent.critic1(states)
    q2_values = agent.critic2(states)
    q1_selected = q1_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    q2_selected = q2_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Target Q-values (using target networks)
    with torch.no_grad():
        # Get next Q-values from target critics
        target_q1 = agent.target_critic1(next_states)
        target_q2 = agent.target_critic2(next_states)
        target_q = torch.min(target_q1, target_q2)  # Clipped double Q-learning

        # Compute target
        max_target_q = target_q.max(dim=1)[0]
        target = rewards + gamma * (1 - dones) * max_target_q

    # Standard TD loss
    td_loss1 = F.mse_loss(q1_selected, target)
    td_loss2 = F.mse_loss(q2_selected, target)
    td_loss = td_loss1 + td_loss2

    # CQL penalty: penalize Q-values for all actions
    # This prevents overestimation on unseen actions
    q1_logsumexp = torch.logsumexp(q1_values / tau, dim=1)
    q2_logsumexp = torch.logsumexp(q2_values / tau, dim=1)

    cql_penalty1 = q1_logsumexp.mean() - q1_selected.mean()
    cql_penalty2 = q2_logsumexp.mean() - q2_selected.mean()
    cql_penalty = cql_penalty1 + cql_penalty2

    # Total loss
    loss = td_loss + cql_alpha * cql_penalty

    # Info for logging
    info = {
        'td_loss': td_loss.item(),
        'cql_penalty': cql_penalty.item(),
        'q1_mean': q1_selected.mean().item(),
        'q2_mean': q2_selected.mean().item(),
        'target_mean': target.mean().item()
    }

    return loss, info


def compute_distributional_critic_loss(critic: DistributionalCritic,
                                       target_critic: DistributionalCritic,
                                       batch: List[Dict],
                                       device: str = 'cuda',
                                       gamma: float = 0.99) -> Tuple[torch.Tensor, Dict]:
    """
    Compute distributional critic loss (models Q-distribution, not just mean).

    Loss = NLL(Q(s,a) | target_distribution)

    The critic outputs (mean, std) for Q-values. We train it to match
    the distribution of actual returns, not just the mean.

    Args:
        critic: Distributional critic network
        target_critic: Target critic network
        batch: List of transition dictionaries
        device: Device to compute on
        gamma: Discount factor

    Returns:
        Tuple of (loss, info_dict)
    """
    critic.train()
    target_critic.eval()

    # Extract batch components
    states = torch.stack([t['state'] for t in batch]).to(device)
    actions = torch.tensor([t['action'] for t in batch], dtype=torch.long).to(device)
    rewards = torch.tensor([t['reward'] for t in batch], dtype=torch.float32).to(device)
    next_states = torch.stack([t['next_state'] for t in batch]).to(device)
    dones = torch.tensor([t['done'] for t in batch], dtype=torch.float32).to(device)

    # Current Q-distribution
    q_mean, q_std = critic(states)
    q_mean_selected = q_mean.gather(1, actions.unsqueeze(1)).squeeze(1)
    q_std_selected = q_std.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Target Q-distribution
    with torch.no_grad():
        target_q_mean, target_q_std = target_critic(next_states)
        # Use conservative estimate (mean - 0.5*std) for target
        conservative_target_q = target_q_mean - 0.5 * target_q_std
        max_target_q_mean = conservative_target_q.max(dim=1)[0]

        # Target mean
        target_mean = rewards + gamma * (1 - dones) * max_target_q_mean

        # Target std: uncertainty should propagate
        max_idx = conservative_target_q.argmax(dim=1)
        target_std_next = target_q_std.gather(1, max_idx.unsqueeze(1)).squeeze(1)
        target_std = gamma * (1 - dones) * target_std_next + 0.01  # Small base uncertainty

    # Negative Log-Likelihood loss (Gaussian assumption)
    # NLL = 0.5 * ((pred - target)^2 / std^2) + log(std)
    nll_mean = 0.5 * ((q_mean_selected - target_mean)**2 / (q_std_selected**2 + 1e-6))
    nll_std = torch.log(q_std_selected + 1e-6)
    nll_loss = (nll_mean + nll_std).mean()

    # Regularization: prevent std from collapsing or exploding
    std_regularization = torch.relu(0.01 - q_std_selected).mean() + torch.relu(q_std_selected - 10.0).mean()

    loss = nll_loss + 0.1 * std_regularization

    # Info for logging
    info = {
        'nll_loss': nll_loss.item(),
        'q_mean': q_mean_selected.mean().item(),
        'q_std': q_std_selected.mean().item(),
        'target_mean': target_mean.mean().item(),
        'target_std': target_std.mean().item(),
        'std_reg': std_regularization.item()
    }

    return loss, info


def augment_state(state: torch.Tensor, noise_level: float = 0.01, device: str = 'cuda') -> torch.Tensor:
    """
    Apply data augmentation to state by adding small noise.

    This makes the policy robust to state uncertainty and prevents overfitting.

    Args:
        state: Input state tensor (can be on any device)
        noise_level: Standard deviation of Gaussian noise
        device: Target device for computation

    Returns:
        Augmented state on the target device
    """
    if noise_level > 0:
        # Move state to target device if not already there
        state = state.to(device)
        # Create noise on same device as state
        noise = torch.randn_like(state) * noise_level
        return state + noise
    return state.to(device)


def augment_action_probs(action_probs: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply temperature scaling to action probabilities.

    Higher temperature → more exploration (smoother distribution)
    Lower temperature → more exploitation (sharper distribution)

    Args:
        action_probs: Action probability distribution
        temperature: Temperature parameter (default 1.0 = no change)

    Returns:
        Temperature-scaled probabilities
    """
    if temperature != 1.0:
        # Apply temperature scaling
        logits = torch.log(action_probs + 1e-8)
        scaled_logits = logits / temperature
        return F.softmax(scaled_logits, dim=-1)
    return action_probs


def analyze_return_risk_relationship(buffer: ReplayBuffer, num_samples: int = 10000) -> Dict:
    """
    Analyze relationship between predicted returns and realized risk.

    This helps understand:
    - Do high predicted returns lead to high risk?
    - What's the downside risk for each prediction bin?
    - Optimal risk/return tradeoff

    Args:
        buffer: Replay buffer with transitions
        num_samples: Number of samples to analyze

    Returns:
        Dictionary with analysis results
    """
    if len(buffer.buffer) < num_samples:
        num_samples = len(buffer.buffer)

    transitions = buffer.sample(num_samples)

    # Extract predicted returns (from state embeddings)
    pred_returns_1d = [t['state'][1428].item() if t['state'].numel() > 1428 else 0 for t in transitions]
    pred_returns_3d = [t['state'][1429].item() if t['state'].numel() > 1429 else 0 for t in transitions]
    pred_returns_5d = [t['state'][1430].item() if t['state'].numel() > 1430 else 0 for t in transitions]
    pred_returns_10d = [t['state'][1431].item() if t['state'].numel() > 1431 else 0 for t in transitions]

    # Actual outcomes
    actual_returns = [t['reward'] for t in transitions]

    results = {}

    # Analyze each horizon
    for horizon, pred in [('1d', pred_returns_1d), ('3d', pred_returns_3d),
                          ('5d', pred_returns_5d), ('10d', pred_returns_10d)]:

        # Bin by predicted return (quintiles)
        pred_array = np.array(pred)
        returns_array = np.array(actual_returns)

        bins = np.percentile(pred_array, [0, 20, 40, 60, 80, 100])

        for i in range(len(bins)-1):
            mask = (pred_array >= bins[i]) & (pred_array < bins[i+1])
            if mask.sum() == 0:
                continue

            bin_returns = returns_array[mask]

            results[f'{horizon}_quintile_{i+1}'] = {
                'pred_return_range': (float(bins[i]), float(bins[i+1])),
                'actual_mean_return': float(np.mean(bin_returns)),
                'actual_std_return': float(np.std(bin_returns)),
                'sharpe_ratio': float(np.mean(bin_returns) / (np.std(bin_returns) + 1e-8)),
                'downside_mean': float(np.mean([r for r in bin_returns if r < 0])) if any(r < 0 for r in bin_returns) else 0.0,
                'prob_loss': float(sum(1 for r in bin_returns if r < 0) / len(bin_returns)),
                'max_loss': float(np.min(bin_returns)),
                'max_gain': float(np.max(bin_returns)),
                'num_samples': int(mask.sum())
            }

    return results


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


# ============================================================================
# TRAINING LOOP INTEGRATION HELPERS (for noisy stock data)
# ============================================================================

def train_step_with_augmentation(agent: ActorCriticAgent,
                                   buffer: ReplayBuffer,
                                   batch_size: int = 256,
                                   gamma: float = 0.99,
                                   device: str = 'cuda',
                                   use_cql: bool = False,
                                   cql_alpha: float = 1.0,
                                   augment_states: bool = True,
                                   noise_level: float = 0.01) -> Dict:
    """
    Single training step with data augmentation for robustness to noisy data.

    Integrates:
    - Data augmentation (state noise)
    - Conservative Q-Learning (CQL) optional
    - Curriculum learning via PER

    Args:
        agent: ActorCriticAgent instance
        buffer: ReplayBuffer with experiences
        batch_size: Batch size for training
        gamma: Discount factor
        device: Device to train on
        use_cql: Whether to use Conservative Q-Learning loss
        cql_alpha: CQL penalty coefficient
        augment_states: Whether to apply state augmentation
        noise_level: Noise level for state augmentation

    Returns:
        Dictionary with training metrics
    """
    if len(buffer) < batch_size:
        return {'critic_loss': 0.0, 'actor_loss': 0.0}

    # Sample batch (with PER if enabled)
    batch = buffer.sample(batch_size)

    # Apply data augmentation to states
    if augment_states:
        for trans in batch:
            trans['state'] = augment_state(trans['state'], noise_level=noise_level, device=device)
            trans['next_state'] = augment_state(trans['next_state'], noise_level=noise_level, device=device)

    # Compute critic loss (with CQL if enabled)
    if use_cql:
        critic_loss, loss_info = compute_cql_loss(
            agent=agent,
            batch=batch,
            device=device,
            gamma=gamma,
            cql_alpha=cql_alpha
        )
    else:
        critic_loss, loss_info = compute_critic_loss(
            agent=agent,
            batch=batch,
            device=device,
            gamma=gamma
        )

    # Update PER priorities with curriculum learning
    if buffer.use_per and 'td_errors' in loss_info:
        indices = [t['index'] for t in batch if 'index' in t]
        if len(indices) > 0:
            buffer.update_priorities(
                indices=indices,
                td_errors=loss_info['td_errors'],
                use_curriculum=True,  # Focus on medium-difficulty samples
                curriculum_percentiles=(0.3, 0.95)
            )

    return {
        'critic_loss': critic_loss.item(),
        'actor_loss': 0.0,  # No actor updates in this simple version
        **loss_info
    }


def post_episode_processing(buffer: ReplayBuffer,
                              episode_transitions: List[Dict],
                              use_her: bool = True,
                              hindsight_ratio: float = 0.5) -> Dict:
    """
    Post-episode processing with Hindsight Experience Replay.

    Call this after each episode completes to add hindsight experiences
    from failed trades and early exits.

    Args:
        buffer: ReplayBuffer to add hindsight experiences to
        episode_transitions: List of transitions from the completed episode
        use_her: Whether to use Hindsight Experience Replay
        hindsight_ratio: Fraction of negative-reward transitions to relabel

    Returns:
        Dictionary with statistics
    """
    stats = {
        'episode_length': len(episode_transitions),
        'total_reward': sum(t['reward'] for t in episode_transitions),
        'negative_rewards': sum(1 for t in episode_transitions if t['reward'] < 0),
        'hindsight_added': 0
    }

    if use_her and len(episode_transitions) > 0:
        # Count buffer size before HER
        buffer_size_before = len(buffer)

        # Add hindsight experiences
        buffer.add_hindsight_experiences(
            episode_transitions=episode_transitions,
            hindsight_ratio=hindsight_ratio
        )

        # Count hindsight experiences added
        stats['hindsight_added'] = len(buffer) - buffer_size_before

    return stats


def create_distributional_agent(predictor_checkpoint_path: str,
                                  state_dim: int = 11761,
                                  hidden_dim: int = 1024,
                                  action_dim: int = 9) -> nn.Module:
    """
    Create agent with distributional critics for uncertainty-aware trading.

    This is an alternative to ActorCriticAgent that uses DistributionalCritic
    instead of standard CriticNetwork.

    Args:
        predictor_checkpoint_path: Path to price predictor checkpoint
        state_dim: State dimension
        hidden_dim: Hidden dimension for networks
        action_dim: Number of actions

    Returns:
        Agent with distributional critics

    Example usage:
        agent = create_distributional_agent(
            predictor_checkpoint_path='./checkpoints/best_model.pt',
            state_dim=11761,
            action_dim=9
        )

        # Use conservative Q-values for action selection (risk-averse)
        q_values = agent.critic1.get_conservative_q(state, confidence=0.9)
        action = q_values.argmax(dim=-1)
    """
    # Create agent structure
    class DistributionalActorCriticAgent(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_extractor = PredictorFeatureExtractor(predictor_checkpoint_path)
            self.actor = ActorNetwork(state_dim, hidden_dim, action_dim)

            # Distributional critics instead of standard critics
            self.critic1 = DistributionalCritic(state_dim, hidden_dim, action_dim)
            self.critic2 = DistributionalCritic(state_dim, hidden_dim, action_dim)

            # Distributional target critics
            self.target_critic1 = DistributionalCritic(state_dim, hidden_dim, action_dim)
            self.target_critic2 = DistributionalCritic(state_dim, hidden_dim, action_dim)

            self.target_critic1.load_state_dict(self.critic1.state_dict())
            self.target_critic2.load_state_dict(self.critic2.state_dict())

            self.target_critic1.eval()
            self.target_critic2.eval()
            for param in self.target_critic1.parameters():
                param.requires_grad = False
            for param in self.target_critic2.parameters():
                param.requires_grad = False

        def update_target_critics(self, tau=0.005):
            for target_param, critic_param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                target_param.data.copy_(tau * critic_param.data + (1.0 - tau) * target_param.data)
            for target_param, critic_param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                target_param.data.copy_(tau * critic_param.data + (1.0 - tau) * target_param.data)

    agent = DistributionalActorCriticAgent()
    print(f"✅ Distributional Actor-Critic Agent Created")
    print(f"   State dim: {state_dim}")
    print(f"   Action dim: {action_dim}")
    print(f"   Critics: Distributional (mean + std)")
    return agent


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
INTEGRATION GUIDE: How to use all new features for noisy stock data

1. RISK-AWARE REWARDS (already in environment)
   - Automatically penalizes volatility and drawdowns
   - No code changes needed in training loop

2. DATA AUGMENTATION
   # In training loop:
   metrics = train_step_with_augmentation(
       agent=agent,
       buffer=buffer,
       batch_size=256,
       augment_states=True,
       noise_level=0.01  # 1% noise
   )

3. HINDSIGHT EXPERIENCE REPLAY
   # After each episode:
   episode_stats = post_episode_processing(
       buffer=buffer,
       episode_transitions=episode_transitions,
       use_her=True,
       hindsight_ratio=0.5
   )

4. CONSERVATIVE Q-LEARNING
   # In training step:
   metrics = train_step_with_augmentation(
       agent=agent,
       buffer=buffer,
       use_cql=True,
       cql_alpha=1.0  # Penalty strength
   )

5. CURRICULUM LEARNING (automatic in PER)
   # When creating buffer:
   buffer = ReplayBuffer(
       capacity=100000,
       use_per=True,
       per_alpha=0.6,
       per_beta=0.4
   )
   # Curriculum learning is automatically applied in update_priorities()

6. DISTRIBUTIONAL CRITICS
   # Create agent with distributional critics:
   agent = create_distributional_agent(
       predictor_checkpoint_path='./checkpoints/best_model.pt'
   )

   # Use conservative Q-values (risk-averse):
   q_conservative = agent.critic1.get_conservative_q(state, confidence=0.9)
   action = q_conservative.argmax()

   # Or use distributional loss:
   loss, info = compute_distributional_critic_loss(
       critic=agent.critic1,
       target_critic=agent.target_critic1,
       batch=batch
   )

7. RETURN/RISK ANALYSIS
   # Periodically analyze buffer:
   if episode % 100 == 0:
       analysis = analyze_return_risk_relationship(buffer, num_samples=10000)
       print(f"Quintile 5 (highest pred): Sharpe={analysis['1d_quintile_5']['sharpe_ratio']:.2f}")

COMPLETE TRAINING LOOP EXAMPLE:

    # Setup
    buffer = ReplayBuffer(capacity=100000, use_per=True, n_step=3)
    agent = create_distributional_agent(predictor_checkpoint_path='./checkpoints/best.pt')

    for episode in range(num_episodes):
        # Run episode
        states = env.reset()
        episode_transitions = []
        done = False

        while not done:
            # Select action
            action = agent.critic1.get_conservative_q(state, confidence=0.9).argmax()

            # Step environment (risk-aware rewards automatically applied)
            next_states, reward, done, info = env.step({ticker: action})

            # Store transition
            transition = {
                'state': state, 'action': action, 'reward': reward,
                'next_state': next_state, 'done': done
            }
            episode_transitions.append(transition)
            buffer.push(**transition)

        # Post-episode: Add hindsight experiences
        post_episode_processing(buffer, episode_transitions, use_her=True)

        # Training: Data augmentation + CQL + Curriculum learning
        metrics = train_step_with_augmentation(
            agent=agent, buffer=buffer,
            augment_states=True, use_cql=True, cql_alpha=1.0
        )

        # Periodic analysis
        if episode % 100 == 0:
            analysis = analyze_return_risk_relationship(buffer)
            print(f"Risk/Return Analysis: {analysis}")
"""
