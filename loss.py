import torch.nn as nn


class CustomCELoss(nn.Module):
    def __init__(self, weights:dict, labels:dict):
        super(CustomCELoss, self).__init__()

        self.criterion = {"Train": nn.CrossEntropyLoss(weight=weights["Train"]),
                          "Val": nn.CrossEntropyLoss(weight=weights["Val"])}
        self.labels = {"Train": labels["Train"], "Val": labels["Val"]}

    def forward(self, outputs, phase):
        return self.criterion[phase](outputs, self.labels[phase])
