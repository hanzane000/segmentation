import torch


def _prepare_predictions_and_targets(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = (targets.float() > 0.5).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    return preds, targets


def dice_score(logits, targets, threshold=0.5, eps=1e-7):
    preds, targets = _prepare_predictions_and_targets(logits, targets, threshold)
    intersection = (preds * targets).sum(dim=1)
    denominator = preds.sum(dim=1) + targets.sum(dim=1)
    score = (2.0 * intersection + eps) / (denominator + eps)
    return score.mean().item()


def iou_score(logits, targets, threshold=0.5, eps=1e-7):
    preds, targets = _prepare_predictions_and_targets(logits, targets, threshold)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    score = (intersection + eps) / (union + eps)
    return score.mean().item()


def precision_score(logits, targets, threshold=0.5, eps=1e-7):
    preds, targets = _prepare_predictions_and_targets(logits, targets, threshold)
    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1.0 - targets)).sum(dim=1)
    score = (tp + eps) / (tp + fp + eps)
    return score.mean().item()


def recall_score(logits, targets, threshold=0.5, eps=1e-7):
    preds, targets = _prepare_predictions_and_targets(logits, targets, threshold)
    tp = (preds * targets).sum(dim=1)
    fn = ((1.0 - preds) * targets).sum(dim=1)
    score = (tp + eps) / (tp + fn + eps)
    return score.mean().item()


def pixel_accuracy(logits, targets, threshold=0.5, eps=1e-7):
    preds, targets = _prepare_predictions_and_targets(logits, targets, threshold)
    correct = (preds == targets).float().sum(dim=1)
    total = targets.size(1)
    score = (correct + eps) / (total + eps)
    return score.mean().item()
