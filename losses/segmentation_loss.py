import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.float().contiguous().view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        loss = 1.0 - dice
        return loss.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        targets = targets.float()
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class BoundaryBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, boundary_logits, boundary_targets):
        boundary_targets = boundary_targets.float()
        return self.bce(boundary_logits, boundary_targets)


class UncertaintyBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, uncertainty_map, uncertainty_targets):
        uncertainty_targets = uncertainty_targets.float()
        return self.bce(uncertainty_map, uncertainty_targets)


class MultiBranchRefineLoss(nn.Module):
    def __init__(
        self,
        lambda_coarse=0.5,
        lambda_final=1.0,
        lambda_boundary=0.3,
        lambda_uncertainty=0.2,
        bce_weight=1.0,
        dice_weight=1.0,
        smooth=1.0,
    ):
        super().__init__()
        self.lambda_coarse = lambda_coarse
        self.lambda_final = lambda_final
        self.lambda_boundary = lambda_boundary
        self.lambda_uncertainty = lambda_uncertainty

        self.seg_loss = BCEDiceLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            smooth=smooth,
        )
        self.boundary_loss = BoundaryBCELoss()
        self.uncertainty_loss = UncertaintyBCELoss()

    def forward(
        self,
        coarse_logits,
        final_logits,
        boundary_logits,
        uncertainty_map,
        seg_targets,
        boundary_targets,
        uncertainty_targets,
    ):
        coarse_loss = self.seg_loss(coarse_logits, seg_targets)
        final_loss = self.seg_loss(final_logits, seg_targets)
        boundary_loss = self.boundary_loss(boundary_logits, boundary_targets)
        uncertainty_loss = self.uncertainty_loss(uncertainty_map, uncertainty_targets)

        total_loss = (
            self.lambda_coarse * coarse_loss
            + self.lambda_final * final_loss
            + self.lambda_boundary * boundary_loss
            + self.lambda_uncertainty * uncertainty_loss
        )

        loss_dict = {
            "coarse_loss": coarse_loss.detach().item(),
            "final_loss": final_loss.detach().item(),
            "boundary_loss": boundary_loss.detach().item(),
            "uncertainty_loss": uncertainty_loss.detach().item(),
            "total_loss": total_loss.detach().item(),
        }
        return total_loss, loss_dict
