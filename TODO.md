# TODO

### 1. Paperspace remote GPU development workflow

1. Instance stop/start functionality
    - Reason: Eliminate charges during inactive development periods while preserving the environment state.
2. Zero-Reconfiguration Persistence: Persistent file system for code, SSH keys, and configurations
    - Reason: Resume development immediately after restarts without repetitive/tedious environment rebuilding.
3. Consistent Remote Access: Persistent IP for SSH connections
    - Reason: Maintain the same connection endpoint across instance restarts, avoiding SSH configuration updates.
4. Hybrid Development Workflow (with VSCode)
    - Reason: Develop locally when I don't need a GPU runtime and develop or use the VSCode debugger/IDE remotely when I do need a GPU runtime (pushing/pulling to GitHub to sync changes).

### 2. Train/evaluate DreamerV3 on some non-trivial environment for N steps, saving results

### 3. Implement Dream-&-Ponder

1. Wrap actor w/ PonderNet wrapper
- Update config w/ new/renamed args/vals
- DON'T do anything to "imagination" algo... I.e., DON'T DO THIS (just sample _more_ trajectories-`cfg.algo.per_rank_batch_size * cfg.algo.per_rank_sequence_length`)

```python
"""Pseudocode for getting branching trajectories"""

# [[(p_ocurring, prior, action), (p_ocurring, prior, action), ...], ...]
trajectory: list[list[tuple]] = [[]] * cfg.algo.horizon

def populate_trajectory_for_branch(from_i_onward, p_occuring, prior, action):
    # Populate for earliest-most level of the branch
    trajectory[from_i_onward].append(p_occuring, prior, action)
    # Stop if end of horizon
    if cfg.horizon >= from_i_onward:
        return
    # Else branch further
    actions, p_halts = actor(prior)
    imagined_priors = imagination([prior] * n_actions, actions)
    for p_halt, imagined_prior, action in zip(p_halts, imagined_priors, actions):
        populate_trajectory_for_branch(
            from_i_onward=from_i_onward + 1,
            p_occuring=p_occuring * p_halt,
            prior=imagined_prior,
            action=action,
        )
   
actions, p_halts = actor(prior)
imagined_priors = imagination([prior] * n_actions, actions)
for p_halt, imagined_prior, action in zip(p_halts, imagined_priors, actions):
    populate_trajectory_for_branch(
        from_i_onward=1,
        p_occuring=p_halt,
        prior=imagined_prior,
        action=action,
    )
```

- Do this instead:

```python
# In this line of code, we sample an action for all 1024 (64*16) latent observations in train step
actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)
# Here, we set this as the first imagined action for all 1024 imagined trajectories
# TODO: The actor, instead of returning [1, per_rank_batch_size*per_rank_sequence_length, action_dim],
# will return a tuple of actions, halt probs ([1, max_halts, ", "], [1, max_halts, 1])
# TODO: What is the first dimension here?
# TODO: Maintain this max_halts dimension (imagining trajectories for all halts), all the way up until loss calc
# TODO: After which, we calculate the halt-prob-aggregated PonderNet loss and backward() from that
imagined_actions[0] = actions
```

### 4. Train/evaluate Dream-&-Ponder on _same_ non-trivial environment for N steps (functionally ⟨=⟩ params as otherwise), saving results

___


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Callable
from functools import lru_cache


class PonderNet(nn.Module):
    """
    PonderNet: Adaptive computation time neural network.

    Dynamically decides how many steps to compute using a halting mechanism,
    balancing computation cost and accuracy with a geometric prior regularization.

    Args:
        step_module: Module taking (input, state) and returning (output, new_state).
        halting_module: Module taking state and returning halting probability logit.
        max_steps: Maximum number of pondering steps (default: 20).
        lambda_p: Parameter for geometric prior (default: 0.01).
        epsilon: Small value for numerical stability (default: 1e-6).
    """

    @lru_cache(maxsize=1)
    @staticmethod
    def _get_geometric_prior(max_steps: int, lambda_p: float = 0.01) -> torch.Tensor:
        """Generate normalized geometric distribution prior for regularization."""
        steps = torch.arange(1, max_steps + 1, dtype=torch.float32)
        probs = lambda_p * (1 - lambda_p) ** (steps - 1)
        return probs / probs.sum()  # Normalize to sum to 1

    def __init__(
        self,
        step_module: nn.Module,
        halting_module: nn.Module,
        max_steps: int = 20,
        lambda_p: float = 0.01,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.step_module = step_module
        self.halting_module = halting_module
        self.max_steps = max_steps
        self.lambda_p = lambda_p
        self.epsilon = epsilon

    def _compute_halting_probs(self, halting_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute halting probabilities from logits.

        p_halt(n) = p(n) * prod_{i=1}^{n-1} (1 - p(i)), where p(i) = sigmoid(halting_logit(i)).
        Remainder probability is assigned to the final step.

        Args:
            halting_logits: Logits of halting probabilities [batch, max_steps, 1].

        Returns:
            Halting probabilities [batch, max_steps].
        """
        p = torch.sigmoid(halting_logits.squeeze(-1))  # [batch, max_steps]
        not_halted = torch.cumprod(1 - p + self.epsilon, dim=1)  # Survival probability
        not_halted_prev = torch.cat(
            [torch.ones_like(p[:, :1]), not_halted[:, :-1]], dim=1
        )  # Shift for alignment
        halting_probs = p * not_halted_prev
        # Assign remainder to final step
        remainder = torch.clamp(1 - halting_probs.sum(dim=1, keepdim=True), min=0)
        halting_probs = halting_probs + torch.cat(
            [torch.zeros_like(halting_probs[:, :-1]), remainder], dim=1
        )
        return halting_probs.clamp(min=self.epsilon)

    def forward(
        self, x: torch.Tensor, initial_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through PonderNet.

        Args:
            x: Input tensor [batch, ...].
            initial_state: Initial hidden state [batch, hidden_size] (optional).

        Returns:
            final_output: Weighted average of step outputs [batch, output_dim].
            halting_probs: Halting probabilities [batch, max_steps].
            outputs: All step outputs [batch, max_steps, output_dim].
            states: All step states [batch, max_steps, hidden_size].
        """
        batch_size, device = x.shape[0], x.device
        state = (
            torch.zeros(batch_size, self.step_module.hidden_size, device=device)
            if initial_state is None
            else initial_state
        )

        outputs, states, halting_logits = [], [], []
        for _ in range(self.max_steps):
            output, state = self.step_module(x, state)
            outputs.append(output)
            states.append(state)
            halting_logits.append(self.halting_module(state))

        outputs = torch.stack(outputs, dim=1)  # [batch, max_steps, output_dim]
        states = torch.stack(states, dim=1)  # [batch, max_steps, hidden_size]
        halting_logits = torch.stack(halting_logits, dim=1)  # [batch, max_steps, 1]
        halting_probs = self._compute_halting_probs(halting_logits)  # [batch, max_steps]
        final_output = (outputs * halting_probs.unsqueeze(-1)).sum(dim=1)  # [batch, output_dim]

        return final_output, halting_probs, outputs, states


def compute_pondernet_loss(
    outputs: torch.Tensor,
    halting_probs: torch.Tensor,
    targets: torch.Tensor,
    max_steps: int,
    lambda_p: float = 0.01,
    epsilon: float = 1e-6,
    loss_fn: Callable = F.mse_loss,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute PonderNet loss with reconstruction and KL regularization terms.

    Args:
        outputs: Intermediate outputs [batch, max_steps, output_dim].
        halting_probs: Halting probabilities [batch, max_steps].
        targets: Target values [batch, output_dim].
        max_steps: Maximum number of steps.
        lambda_p: Geometric prior parameter (default: 0.01).
        epsilon: Small value for numerical stability (default: 1e-6).
        loss_fn: Reconstruction loss function (default: MSE).

    Returns:
        total_loss: Combined reconstruction and regularization loss.
        loss_dict: Dictionary with loss components and average steps.
    """
    # Reconstruction loss: p_halt-weighted average of step-wise losses
    step_losses = []
    for step in range(max_steps):
        step_loss = loss_fn(outputs[:, step], targets, reduction="none").mean(dim=-1)
        step_losses.append(step_loss)
    step_losses = torch.stack(step_losses, dim=1)  # [batch, max_steps]
    reconstruction_loss = (halting_probs * step_losses).sum(dim=1).mean()

    # KL divergence with geometric prior
    geometric_prior = PonderNet._get_geometric_prior(max_steps, lambda_p).to(outputs.device)
    kl_div_loss = (
        (halting_probs * torch.log(halting_probs / (geometric_prior + epsilon) + epsilon))
        .sum(dim=1)
        .mean()
    )

    total_loss = reconstruction_loss + kl_div_loss
    avg_steps = (
        (halting_probs * torch.arange(1, max_steps + 1, device=outputs.device)).sum(dim=1).mean()
    )

    return total_loss, {
        "total_loss": total_loss,
        "reconstruction_loss": reconstruction_loss,
        "kl_divergence": kl_div_loss,
        "avg_steps": avg_steps,
    }
```