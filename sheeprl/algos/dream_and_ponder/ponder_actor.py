# Configs:

# - halting module mlp layers/hidden dim
# - recursive_goal_module mlp layers/hidden dim
# - max_ponder_steps
# - cum_halt_prob_threshold
# - beta (for PonderNetLoss)
# - lambda_prior_geom (for PonderNetLoss)

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class PonderActorOutput:
    training_mode: bool
    # Training mode outputs
    halt_step_outputs: Optional[torch.Tensor] = None
    halt_probs: Optional[torch.Tensor] = None
    halt_distribution: Optional[torch.Tensor] = None
    # Inference mode outputs
    halted_at_output: Optional[torch.Tensor] = None
    halted_at_step: Optional[torch.Tensor] = None


class PonderActor(nn.Module):
    """
    Actor module based on PonderNet (Benino at al, 2021) that recurrently
    refines/ponders over the env state & a work-in-progress abstract goal representation
    until the goal can be decoded into logits from which the optimal action distribution
    is immediately calculated (e.g., with a softmax layer).
    """

    PRE_SIGMOID_CLAMP = (-7, 7)

    def __init__(
        self,
        latent_state_dim: int,
        goal_ponder_module: nn.Module,
        halt_module: nn.Module,
        action_decoder: nn.Module,
        max_ponder_steps: int = 4,
        cum_halt_prob_threshold: float = 0.95,
        deterministic_inference: bool = False,
    ):
        assert 0 < cum_halt_prob_threshold <= 1, "cum_halt_prob_threshold must be in (0, 1]"
        assert max_ponder_steps > 0, "max_ponder_steps must be positive"
        super().__init__()
        self.training = True  # Default to training mode
        self.latent_state_dim = latent_state_dim
        self.goal_ponder_module = goal_ponder_module
        self.halt_module = halt_module
        self.action_decoder = action_decoder
        self.max_ponder_steps = max_ponder_steps
        self.cum_halt_prob_threshold = cum_halt_prob_threshold
        self.deterministic_inference = deterministic_inference
        self.no_goal_yet_representation = nn.Parameter(torch.rand(latent_state_dim))

    # def _compute_halting_distribution(self, halt_probs: torch.Tensor) -> torch.Tensor:
    #     """
    #     Convert halting probabilities λ_n to distribution p_n.
    #     p_n = λ_n * ∏_{i=1}^{n-1} (1 - λ_i)
    #     """
    #     batch_size, max_steps = halt_probs.shape
    #     # Compute cumulative products of (1 - λ_i) for i=1 to n-1
    #     not_halt = torch.clamp(1 - halt_probs, min=1e-7)  # Min clamp prevents underflow
    #     cumprods = torch.cat(
    #         [
    #             torch.ones(batch_size, 1, device=halt_probs.device),
    #             torch.cumprod(not_halt[:, :-1], dim=1),
    #         ],
    #         dim=1,
    #     )
    #     # Compute p_n = λ_n * ∏_{i=1}^{n-1} (1 - λ_i)
    #     p_n = halt_probs * cumprods
    #     return p_n

    def _compute_halting_distribution(self, halt_probs: torch.Tensor) -> torch.Tensor:
        """
        Convert halting probabilities λ_n to distribution p_n.
        p_n = λ_n * ∏_{i=1}^{n-1} (1 - λ_i)
        """
        batch_size, max_steps = halt_probs.shape
        not_halt = torch.clamp(1 - halt_probs, min=1e-7)
        cumprods = torch.cat(
            [
                torch.ones(batch_size, 1, device=halt_probs.device),
                torch.cumprod(not_halt[:, :-1], dim=1),
            ],
            dim=1,
        )
        p_n = halt_probs * cumprods
        # Set final step to the leftover mass; avoids in-place on a tensor used to compute remainder
        last = 1.0 - p_n[:, :-1].sum(dim=1)
        last = torch.clamp(last, min=0.0)
        p_n = torch.cat([p_n[:, :-1], last.unsqueeze(1)], dim=1)
        return p_n

    def forward(self, env_state: torch.Tensor) -> PonderActorOutput:
        """Infers action logits from a latent environment state representation."""
        if self.training:
            return self._forward_train(env_state)
        else:
            return self._forward_inference(env_state)

    def _forward_train(self, env_state: torch.Tensor) -> PonderActorOutput:
        """
        Training-mode forward pass where entire batch is computed until max_ponder_steps.
        """
        ####################
        ## 1. Ponder Step ##
        ####################

        batch_size = env_state.size(0)
        device = env_state.device
        halt_step_goals = []
        halt_probs = []
        no_goal_yet = self.no_goal_yet_representation.expand(batch_size, -1)
        current_input = torch.cat([env_state, no_goal_yet], dim=-1)
        for step in range(self.max_ponder_steps):
            # Apply one more recurrent goal module forward pass
            goal = self.goal_ponder_module(current_input)

            # Infer probability that inferred/refined goal is ready to be decoded as a next-action
            halt_module_input = torch.cat([env_state, goal], dim=-1)
            halt_prob_logit = self.halt_module(halt_module_input)
            # Clamp to avoid vanishing gradients from extreme/small sigmoid input/outputs
            halt_prob_logit = torch.clamp(halt_prob_logit, *self.PRE_SIGMOID_CLAMP)
            halt_prob = torch.sigmoid(halt_prob_logit)

            halt_step_goals.append(goal)
            halt_probs.append(halt_prob)
            current_input = torch.cat([env_state, goal], dim=-1)  # Update for next step

        halt_step_goals = torch.stack(halt_step_goals, dim=1)  # [batch, max_steps, output_dim]
        halt_probs = torch.stack(halt_probs, dim=1).squeeze(-1)  # [batch, max_steps]
        halting_dist = self._compute_halting_distribution(halt_probs)

        ########################
        ## 2. Action Decoding ##
        ########################

        # Flatten the halt steps dimension into the batch dimension for action decoding
        all_goals = halt_step_goals.view(-1, self.latent_state_dim)
        all_outputs: torch.Tensor = self.action_decoder(all_goals)
        # Reshape back to [batch, max_steps, action_dim]
        halt_step_outputs = all_outputs.view(batch_size, self.max_ponder_steps, -1)

        return PonderActorOutput(
            training_mode=True,
            halt_step_outputs=halt_step_outputs,
            halt_probs=halt_probs,
            halt_distribution=halting_dist,
        )

    def _forward_inference(self, env_state: torch.Tensor) -> PonderActorOutput:
        """
        Inference-mode forward pass with early ponder-stopping as soon as the last
        instance in the batch halts.
        """
        #########################################
        ## 1. Ponder Step (w/ ponder-stopping) ##
        #########################################

        batch_size = env_state.size(0)
        device = env_state.device
        has_halted = torch.zeros(batch_size, dtype=torch.bool, device=device)
        halted_at_goal = torch.zeros(batch_size, self.latent_state_dim, device=device)
        halted_at_steps = torch.zeros(batch_size, dtype=torch.long, device=device)
        cumulative_halt_prob = torch.zeros(batch_size, device=device)

        no_goal_yet = self.no_goal_yet_representation.expand(batch_size, -1)
        current_input = torch.cat([env_state, no_goal_yet], dim=-1)
        for step in range(self.max_ponder_steps):
            is_active = ~has_halted
            if has_halted.all():  # Break early if all instances have halted
                break

            # Infer/refine goal & halt_prob only for still-pondering batch instances
            goal_ = self.goal_ponder_module(current_input[is_active])
            halt_module_input = torch.cat([env_state[is_active], goal_], dim=-1)
            halt_prob_logit = self.halt_module(halt_module_input)
            halt_prob_logit = torch.clamp(halt_prob_logit, *self.PRE_SIGMOID_CLAMP)
            halt_prob_ = torch.sigmoid(halt_prob_logit).squeeze(-1)
            # Create full-batch equivalents of goal and halt_prob
            goal = torch.zeros(batch_size, self.latent_state_dim, device=device)
            goal[is_active] = goal_
            halt_prob = torch.zeros(batch_size, device=device)
            halt_prob[is_active] = halt_prob_

            # Sample halting decisions only for still-pondering batch instances
            halt_decisions = torch.zeros(batch_size, dtype=torch.bool, device=device)
            halt_decisions[is_active] = (
                (halt_prob[is_active] > 0.5)
                if self.deterministic_inference
                else (torch.bernoulli(halt_prob[is_active]) == 1.0)
            )

            # Update outputs and halt status for those decided to be halted this step
            new_halts = halt_decisions & ~has_halted
            if new_halts.any():
                halted_at_goal[new_halts] = goal[new_halts]
                halted_at_steps[new_halts] = step + 1
            has_halted = has_halted | halt_decisions

            # Check prob accumulation and mark those beyond threshold as halted this step
            cumulative_halt_prob += halt_prob * (~has_halted).float()
            threshold_halts = (cumulative_halt_prob >= self.cum_halt_prob_threshold) & ~has_halted
            if threshold_halts.any():
                halted_at_goal[threshold_halts] = goal[threshold_halts]
                halted_at_steps[threshold_halts] = step + 1
                has_halted = has_halted | threshold_halts

            # Update current_input for still-active instances
            still_active = ~has_halted
            current_input = torch.zeros_like(current_input)
            current_input[still_active] = torch.cat(
                [env_state[still_active], goal[still_active]], dim=-1
            )

        else:  # We never hit the break, meaning not all instances halted before max steps
            # Manually halt outstanding instances
            never_halted = ~has_halted
            halted_at_goal[never_halted] = goal[never_halted]
            halted_at_steps[never_halted] = self.max_ponder_steps

        ########################
        ## 2. Action Decoding ##
        ########################

        halted_at_output = self.action_decoder(halted_at_goal)

        return PonderActorOutput(
            training_mode=False,
            halted_at_output=halted_at_output,
            halted_at_step=halted_at_steps,
        )


class PonderActorLoss(nn.Module):
    """
    PonderNet loss combining expected task loss with KL regularization.

    Loss = E_p[L_task(y_n)] + β * KL(p || p_G)
    where p is the halting distribution and p_G is a geometric prior.
    """

    geometric_prior: torch.Tensor

    def __init__(self, max_ponder_steps: int, beta: float = 0.01, lambda_prior_geom: float = 0.1):
        """
        Args:
            max_ponder_steps: Maximum number of pondering steps
            beta: Weight for KL divergence regularization term
            lambda_p: Parameter for geometric prior distribution
        """
        super().__init__()
        assert 0.01 <= lambda_prior_geom < 1, "lambda_prior_geom must be in [0.01, 1)"
        self.beta = beta
        self.lambda_prior_geom = lambda_prior_geom
        # Precompute and register geometric prior as buffer
        geometric_prior = self._compute_geometric_prior(max_ponder_steps)
        self.register_buffer("geometric_prior", geometric_prior.unsqueeze(0))

    # def _compute_geometric_prior(self, max_ponder_steps: int) -> torch.Tensor:
    #     """
    #     Compute truncated geometric prior distribution.
    #     p_G(n) = λ_p * (1 - λ_p)^(n-1) normalized over finite steps.
    #     """
    #     n = torch.arange(max_ponder_steps)
    #     geometric_prior = self.lambda_prior_geom * (1 - self.lambda_prior_geom) ** n
    #     # Normalize to sum to 1 over truncated support
    #     geometric_prior = geometric_prior / geometric_prior.sum()
    #     return geometric_prior

    def _compute_geometric_prior(self, max_ponder_steps: int) -> torch.Tensor:
        """
        Compute truncated geometric prior with tail mass at last step.
        p_G(n) = λ_p (1 - λ_p)^(n-1) for n < N; p_G(N) = (1 - λ_p)^(N-1).
        """
        N = max_ponder_steps
        if N == 1:
            return torch.tensor([1.0])
        base = 1.0 - float(self.lambda_prior_geom)
        head = float(self.lambda_prior_geom) * torch.pow(
            torch.full((N - 1,), base, dtype=torch.float32),
            torch.arange(N - 1, dtype=torch.float32),
        )
        tail_last = head.new_tensor([base ** (N - 1)])
        geometric_prior = torch.cat([head, tail_last], dim=0)
        return geometric_prior

    def forward(
        self,
        halt_step_task_losses: torch.Tensor,  # [max_ponder_steps] or [batch, max_ponder_steps]
        halt_distribution: torch.Tensor,  # [max_ponder_steps] or [batch, max_ponder_steps]
    ) -> torch.Tensor:
        """
        Compute PonderNet loss: expected task loss plus β-weighted KL divergence.

        Args:
            halt_step_task_losses: Task losses per step [batch, max_steps]
            halting_distribution: Halting probabilities per step [batch, max_steps]

        Returns:
            Combined loss (expected task loss + β * KL(p || p_G))
        """
        # Expected task loss under halting distribution
        if halt_distribution.dim() == 1:
            expected_loss = torch.dot(halt_step_task_losses, halt_distribution)
            # KL divergence: KL(p || q) = sum(p * log(p/q))
            eps = 1e-8
            prior = self.geometric_prior.squeeze(0)
            kl = torch.log((halt_distribution + eps) / (prior + eps))
            kl_div = (halt_distribution * kl).sum()
        elif halt_distribution.dim() == 2:
            # Shape: [batch, steps]
            expected_loss = (halt_step_task_losses * halt_distribution).sum(dim=1).mean()
            eps = 1e-8
            kl = torch.log((halt_distribution + eps) / (self.geometric_prior + eps))
            kl_div = (halt_distribution * kl).sum(dim=1).mean()
        else:
            raise ValueError("halt_distribution must be 1D or 2D tensor")
        # Combined loss
        loss = expected_loss + self.beta * kl_div
        return loss
