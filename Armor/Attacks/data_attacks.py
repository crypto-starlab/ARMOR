import torch


def is_label_flipping(name):
    return str(name).lower().replace('-', '_') in {'label_flipping', 'label_flip', 'labelflipping'}


def flip_labels(labels, num_classes, source=None, target=None):
    if num_classes < 2:
        raise ValueError('label flipping requires at least two classes')
    if (source is None) != (target is None):
        raise ValueError('Specify both label_flip_source and label_flip_target')
    if labels.dtype != torch.long or ((labels < 0) | (labels >= num_classes)).any():
        raise ValueError('Expected integer class labels in [0, num_classes)')
    if source is None:
        return num_classes - 1 - labels
    if not (0 <= source < num_classes and 0 <= target < num_classes) or source == target:
        raise ValueError('Source and target must be distinct valid class indices')
    return torch.where(labels == source, target, labels)
