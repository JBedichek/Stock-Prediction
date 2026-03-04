#!/usr/bin/env python3
"""
Lightweight Attention Networks for RL Trading.

Set-based architecture with attention over stocks, designed for:
- Sample efficiency (critical for RL)
- Permutation invariance (stock order doesn't matter)
- Explicit modeling of stock relationships

Architecture philosophy:
- NOT a full transformer (overkill, sample-inefficient)
- Lightweight attention to compare stocks
- Much smaller than MLP, more expressive for portfolio tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional


class StockAttentionEncoder(nn.Module):
    """
    Encodes a set of stocks with lightweight self-attention.

    Architecture:
    1. Shared encoder processes each stock independently
    2. Self-attention allows stocks to compare with each other
    3. Pooling aggregates to fixed-size representation

    This is permutation-invariant: stock order doesn't matter.
    """

    def __init__(self, stock_dim: int = 1469, hidden_dim: int = 256, num_heads: int = 4):
        """
        Args:
            stock_dim: Dimension of each stock's features (1469 from predictor)
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads (keep small for sample efficiency)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Encode each stock independently (shared weights across stocks)
        self.stock_encoder = nn.Sequential(
            nn.Linear(stock_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Multi-head attention (lightweight)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, stock_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            stock_features: (batch, num_stocks, stock_dim)

        Returns:
            Tuple of:
            - aggregated: (batch, hidden_dim) - pooled representation
            - attended: (batch, num_stocks, hidden_dim) - per-stock attended features
        """
        batch_size, num_stocks, _ = stock_features.shape

        # Encode each stock independently
        encoded = self.stock_encoder(stock_features)  # (batch, num_stocks, hidden)

        # Multi-head self-attention
        Q = self.query(encoded).reshape(batch_size, num_stocks, self.num_heads, self.head_dim)
        K = self.key(encoded).reshape(batch_size, num_stocks, self.num_heads, self.head_dim)
        V = self.value(encoded).reshape(batch_size, num_stocks, self.num_heads, self.head_dim)

        # Transpose for attention: (batch, num_heads, num_stocks, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended = torch.matmul(attention_weights, V)  # (batch, num_heads, num_stocks, head_dim)

        # Concatenate heads
        attended = attended.transpose(1, 2).reshape(batch_size, num_stocks, self.hidden_dim)

        # Output projection
        attended = self.out_proj(attended)  # (batch, num_stocks, hidden)

        # Aggregate stocks (mean pooling)
        aggregated = attended.mean(dim=1)  # (batch, hidden)

        return aggregated, attended


class AttentionCritic(nn.Module):
    """
    Critic network with lightweight attention over stocks.

    Architecture:
    1. Split state into stocks (8 × 1469) and position (9 dims)
    2. Attend over stocks to capture relationships
    3. Combine with position encoding
    4. Predict Q-values for each action

    Benefits over MLP:
    - Explicitly models "which stock is better than which"
    - Permutation invariant
    - More sample-efficient for portfolio tasks
    """

    def __init__(self, state_dim: int = 11761, hidden_dim: int = 256, action_dim: int = 9):
        """
        Args:
            state_dim: Full state dimension (8*1469 + 9 = 11761)
            hidden_dim: Hidden dimension for attention
            action_dim: Number of actions
        """
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Constants
        self.num_stocks = 8
        self.stock_dim = 1469
        self.position_dim = 9

        # Stock attention encoder
        self.stock_encoder = StockAttentionEncoder(
            stock_dim=self.stock_dim,
            hidden_dim=hidden_dim,
            num_heads=4
        )

        # Position encoder
        self.position_encoder = nn.Sequential(
            nn.Linear(self.position_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Dueling architecture: separate value and advantage streams
        # Value stream (how good is the overall state?)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

        # Advantage stream (how good is each action relative to others?)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, 11761) - 8 stocks × 1469 + 9 position

        Returns:
            Q-values: (batch, action_dim)
        """
        batch_size = state.shape[0]

        # Split state into stocks and position
        # First 8*1469 dims are stocks, last 9 are position
        stocks = state[:, :self.num_stocks * self.stock_dim].reshape(
            batch_size, self.num_stocks, self.stock_dim
        )
        position = state[:, -self.position_dim:]

        # Encode stocks with attention
        stock_summary, _ = self.stock_encoder(stocks)  # (batch, hidden)

        # Encode position
        position_features = self.position_encoder(position)  # (batch, hidden)

        # Combine
        combined = torch.cat([stock_summary, position_features], dim=-1)  # (batch, 2*hidden)

        # Dueling Q-values: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        value = self.value_stream(combined)  # (batch, 1)
        advantages = self.advantage_stream(combined)  # (batch, action_dim)

        # Combine with mean-subtraction trick (stabilizes learning)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values


class AttentionActor(nn.Module):
    """
    Actor network with lightweight attention over stocks.

    Outputs action logits (policy) for discrete actions.
    """

    def __init__(self, state_dim: int = 11761, hidden_dim: int = 256, action_dim: int = 9):
        """
        Args:
            state_dim: Full state dimension (8*1469 + 9 = 11761)
            hidden_dim: Hidden dimension for attention
            action_dim: Number of actions
        """
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Constants
        self.num_stocks = 8
        self.stock_dim = 1469
        self.position_dim = 9

        # Stock attention encoder
        self.stock_encoder = StockAttentionEncoder(
            stock_dim=self.stock_dim,
            hidden_dim=hidden_dim,
            num_heads=4
        )

        # Position encoder
        self.position_encoder = nn.Sequential(
            nn.Linear(self.position_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, 11761) - 8 stocks × 1469 + 9 position

        Returns:
            Action logits: (batch, action_dim)
        """
        batch_size = state.shape[0]

        # Split state into stocks and position
        stocks = state[:, :self.num_stocks * self.stock_dim].reshape(
            batch_size, self.num_stocks, self.stock_dim
        )
        position = state[:, -self.position_dim:]

        # Encode stocks with attention
        stock_summary, _ = self.stock_encoder(stocks)  # (batch, hidden)

        # Encode position
        position_features = self.position_encoder(position)  # (batch, hidden)

        # Combine
        combined = torch.cat([stock_summary, position_features], dim=-1)  # (batch, 2*hidden)

        # Policy logits
        logits = self.policy_head(combined)

        return logits


class AttentionActorCriticAgent(nn.Module):
    """
    Complete actor-critic agent with lightweight attention networks.

    Drop-in replacement for ActorCriticAgent, but with attention instead of MLP.
    """

    def __init__(self,
                 predictor_checkpoint_path: str,
                 state_dim: int = 11761,
                 hidden_dim: int = 256,
                 action_dim: int = 9):
        """
        Args:
            predictor_checkpoint_path: Path to price predictor checkpoint
            state_dim: State dimension (8 stocks × 1469 + 9 position = 11761)
            hidden_dim: Hidden dimension for attention networks
            action_dim: Number of actions (9 = HOLD + 4 long + 4 short)
        """
        super().__init__()

        # Feature extractor (wraps predictor)
        from rl.rl_components import PredictorFeatureExtractor
        self.feature_extractor = PredictorFeatureExtractor(predictor_checkpoint_path)

        # Attention-based actor and critics
        self.actor = AttentionActor(state_dim, hidden_dim, action_dim)

        # Twin critics (TD3-style)
        self.critic1 = AttentionCritic(state_dim, hidden_dim, action_dim)
        self.critic2 = AttentionCritic(state_dim, hidden_dim, action_dim)

        # Target critics
        self.target_critic1 = AttentionCritic(state_dim, hidden_dim, action_dim)
        self.target_critic2 = AttentionCritic(state_dim, hidden_dim, action_dim)

        # Initialize targets with same weights
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Freeze targets
        self.target_critic1.eval()
        self.target_critic2.eval()
        for param in self.target_critic1.parameters():
            param.requires_grad = False
        for param in self.target_critic2.parameters():
            param.requires_grad = False

        self.action_dim = action_dim

        print(f"✅ AttentionActorCriticAgent initialized")
        print(f"   Architecture: Lightweight attention (NOT full transformer)")
        print(f"   State dim: {state_dim}")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   Action dim: {action_dim}")
        print(f"   Parameters: ~{sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select action using actor network.

        Args:
            state: State tensor
            deterministic: If True, select argmax action; if False, sample from policy

        Returns:
            Tuple of (action_id, log_prob, entropy)
        """
        logits = self.actor(state.unsqueeze(0))
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if deterministic:
            action = probs.argmax(dim=-1).item()
        else:
            action = dist.sample().item()

        log_prob = dist.log_prob(torch.tensor(action, device=state.device))
        entropy = dist.entropy()

        return action, log_prob, entropy

    def update_target_critics(self, tau: float = 0.005):
        """Soft update of target critics."""
        for target_param, critic_param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * critic_param.data + (1.0 - tau) * target_param.data)
        for target_param, critic_param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * critic_param.data + (1.0 - tau) * target_param.data)

    # Add compatibility methods from ActorCriticAgent
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

    def select_actions_reduced_batch(self, top_4_stocks_list, bottom_4_stocks_list,
                                       states_list, positions_list, is_short_list,
                                       epsilon=0.0, deterministic=False):
        """
        Batch action selection (compatibility with training loop).

        This method exists for backward compatibility with the training code.
        """
        from rl.reduced_action_space import create_global_state, decode_action_to_trades

        results = []

        for i in range(len(top_4_stocks_list)):
            if len(states_list[i]) == 0:
                results.append((0, None, None, [], None, False))
                continue

            # Create global state
            global_state = create_global_state(
                top_4_stocks_list[i],
                bottom_4_stocks_list[i],
                states_list[i],
                positions_list[i],
                is_short=is_short_list[i],
                device=next(self.parameters()).device
            )

            # Select action
            if not deterministic and np.random.random() < epsilon:
                # Epsilon-greedy exploration
                action = np.random.randint(0, self.action_dim)
                log_prob = torch.tensor(0.0)
                entropy = torch.tensor(0.0)
            else:
                action, log_prob, entropy = self.select_action(global_state, deterministic=deterministic)

            # Decode to trades
            trades, new_position, new_is_short = decode_action_to_trades(
                action,
                top_4_stocks_list[i],
                bottom_4_stocks_list[i],
                positions_list[i],
                is_short_list[i]
            )

            results.append((action, log_prob, entropy, trades, new_position, new_is_short))

        return results


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    """Test attention networks."""
    print("Testing Attention Networks")
    print("="*80)

    # Test critic
    critic = AttentionCritic(state_dim=11761, hidden_dim=256, action_dim=9)
    print(f"\nAttentionCritic:")
    print(f"  Parameters: {count_parameters(critic):,}")

    # Test forward pass
    batch_size = 4
    state = torch.randn(batch_size, 11761)
    q_values = critic(state)
    print(f"  Input shape: {state.shape}")
    print(f"  Output shape: {q_values.shape}")
    assert q_values.shape == (batch_size, 9), "Wrong output shape!"
    print(f"  ✅ Forward pass successful")

    # Test actor
    actor = AttentionActor(state_dim=11761, hidden_dim=256, action_dim=9)
    print(f"\nAttentionActor:")
    print(f"  Parameters: {count_parameters(actor):,}")

    logits = actor(state)
    print(f"  Input shape: {state.shape}")
    print(f"  Output shape: {logits.shape}")
    assert logits.shape == (batch_size, 9), "Wrong output shape!"
    print(f"  ✅ Forward pass successful")

    # Compare to MLP
    print(f"\nComparison to MLP:")
    print(f"  MLP Critic (1024 hidden):  ~12M parameters")
    print(f"  Attention Critic (256 hidden): ~{count_parameters(critic)/1e6:.1f}M parameters")
    print(f"  Reduction: ~{12 / (count_parameters(critic)/1e6):.1f}x smaller")

    print(f"\n✅ All tests passed!")
    print("="*80)
