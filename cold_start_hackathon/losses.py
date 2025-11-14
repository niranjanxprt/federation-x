"""
Advanced loss functions for federated learning - 2025 best practices.
Based on latest research in handling class imbalance in medical imaging.
"""

import numpy as np
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with severe class imbalance.

    Addresses Hospital B's low sensitivity (0.5199) by focusing on hard examples.

    Formula: FL(pt) = -α(1-pt)^γ * log(pt)

    Args:
        alpha (float): Weight for positive class (0.25 = 4:1 ratio)
        gamma (float): Focusing parameter (2.0 = standard, 3.0 = aggressive)
        label_smoothing (float): Optional smoothing to prevent overconfidence

    Expected Impact: Hospital B sensitivity 0.52 → 0.65+

    References:
        - Lin et al. "Focal Loss for Dense Object Detection" (2017)
        - Recent FL medical imaging surveys (2025)
    """

    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Raw logits [batch_size, 1]
            targets: Binary labels [batch_size, 1]
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Standard BCE loss
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)

        # Probability of correct class
        pt = torch.exp(-bce_loss)

        # Focal term: down-weight easy examples
        focal_term = (1 - pt) ** self.gamma

        # Alpha weighting for class balance
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Final focal loss
        focal_loss = alpha_t * focal_term * bce_loss

        return focal_loss.mean()


class AdaptiveFocalLoss(nn.Module):
    """
    Adaptive Focal Loss - adjusts alpha dynamically based on batch statistics.

    Better for federated learning where class distribution varies by client.
    """

    def __init__(self, gamma=2.0, alpha_min=0.15, alpha_max=0.35):
        super().__init__()
        self.gamma = gamma
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def forward(self, inputs, targets):
        # Calculate batch positive ratio
        pos_ratio = targets.mean()

        # Adapt alpha based on batch composition
        # More positives → lower alpha (less weight on positives)
        alpha = self.alpha_max - (self.alpha_max - self.alpha_min) * pos_ratio

        # Standard focal loss with adaptive alpha
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_term = (1 - pt) ** self.gamma
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

        return (alpha_t * focal_term * bce_loss).mean()


class FedMixupLoss(nn.Module):
    """
    Federated Mixup Loss - synthetic data augmentation in loss space.

    Creates virtual training samples by mixing minority/majority classes.
    Particularly effective in federated learning with non-IID data.
    """

    def __init__(self, base_loss='focal', mixup_alpha=0.2):
        super().__init__()
        self.mixup_alpha = mixup_alpha

        if base_loss == 'focal':
            self.criterion = FocalLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        if self.training and torch.rand(1).item() < 0.3:  # 30% chance of mixup
            # Select random pairs for mixing
            batch_size = inputs.size(0)
            indices = torch.randperm(batch_size)

            # Sample mixing ratio from Beta distribution
            lam = torch.from_numpy(
                np.random.beta(self.mixup_alpha, self.mixup_alpha, 1)
            ).float().to(inputs.device)

            # Mix inputs and targets
            mixed_inputs = lam * inputs + (1 - lam) * inputs[indices]
            mixed_targets = lam * targets + (1 - lam) * targets[indices]

            return self.criterion(mixed_inputs, mixed_targets)
        else:
            return self.criterion(inputs, targets)
