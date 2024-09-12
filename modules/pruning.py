import torch.nn.utils.prune as prune
import torch.nn as nn

def prune_model(model, prune_percentage):
    # Prune all convolutional and linear layers in the model
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=prune_percentage / 100.0)
            print(f"Pruned {prune_percentage}% of weights in layer {name}")
    return model
