import torch

class LossComputer:
    def __init__():
        pass

    def loss():
        pass

    def compute_robust_loss():
        pass

    def compute_group_avg():
        pass

    def __call__():
        pass


def init_criterion():
    num_subclasses = None
    subclass_counts = None
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    size_adjustments = None
    if robust: size_adjustments = [N] * num_subclasses
    criterion = LossComputer(criterion,
                             robust,
                             num_subclasses,
                             subclass_counts,
                             robust_lr,
                             stable_dro,
                             size_adjustments,
                             auroc_version,
                             class_map)
    pass
