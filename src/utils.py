import torch

def get_masked_correct_and_total(logits, targets, ignore_index=-1):
    """
    Returns the number of correct predictions and the total number of non-padding tokens.
    
    Args:
        logits (torch.Tensor): Model predictions of shape (batch_size, seq_len, num_classes)
                              or (batch_size * seq_len, num_classes).
        targets (torch.Tensor): Ground truth labels of shape (batch_size, seq_len)
                               or (batch_size * seq_len).
        ignore_index (int): Index to ignore (e.g. -1 for padding).
        
    Returns:
        correct (int): Number of correctly predicted non-padded tokens.
        total (int): Total number of non-padded tokens.
    """
    preds = torch.argmax(logits, dim=-1)
    mask = (targets != ignore_index)
    correct = ((preds == targets) & mask).sum().item()
    total = mask.sum().item()
    return correct, total

def masked_accuracy(logits, targets, ignore_index=-1):
    """
    Computes token-level classification accuracy ignoring padding tokens.
    
    Args:
        logits (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth labels.
        ignore_index (int): Index to ignore.
        
    Returns:
        float: Accuracy score between 0.0 and 1.0.
    """
    correct, total = get_masked_correct_and_total(logits, targets, ignore_index)
    if total == 0:
        return 0.0
    return correct / total
