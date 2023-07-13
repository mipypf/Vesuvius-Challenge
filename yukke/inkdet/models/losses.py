import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricBCEWithLogitsLoss(nn.Module):
    # Symmetric Cross Entropy for Robust Learning with Noisy Labels <https://arxiv.org/abs/1908.06112>
    def __init__(self, alpha: float = 1.0, beta: float = 0.1, eps: float = 1e-4):
        super().__init__()
        assert alpha > 0
        assert beta > 0
        assert eps > 0

        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            y_pred,
            y_true,
        )

        r_bce = F.binary_cross_entropy_with_logits(
            torch.clamp(y_true, min=self.eps, max=1.0),
            y_pred,
        )

        return self.alpha * bce + self.beta * r_bce


class FBetaLoss(nn.Module):
    def __init__(self, beta: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.beta = beta
        self.smooth = smooth

        assert beta > 0
        assert smooth > 0

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = torch.sigmoid(y_pred)
        if y_true.sum() == 0 and y_pred.sum() == 0:
            return 0.0

        y_true_count = y_true.sum()
        ctp = (y_pred * y_true).sum()
        cfp = (y_pred * (1 - y_true)).sum()
        beta_squared = self.beta * self.beta

        c_precision = ctp / (ctp + cfp + self.smooth)
        c_recall = ctp / (y_true_count + self.smooth)
        fbeta = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + self.smooth)

        return 1 - fbeta
