import torch


def accuracy(output_logits, target, multilabel=False, multilabel_threshold=0.5):
    with torch.no_grad():
        if not multilabel:
            preds = torch.argmax(output_logits, dim=1)
            return preds.eq(target).float().mean() * 100.
        # Handling the multilabel case
        preds = output_logits.sigmoid() > multilabel_threshold
        score = torch.tensor(0., requires_grad=False).cuda(target.device)
        for task in range(output_logits.size(1)):
            score += preds[:, task].eq(target[:, task]).float().mean() * 100.
        return score / output_logits.size(1)
