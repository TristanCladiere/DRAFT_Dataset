import torch
import torch.nn as nn


class FCLayer(nn.Module):
    def __init__(self, version:str):
        super(FCLayer, self).__init__()
        version_to_input_size = {"0.6B": 1024, "4B": 2560, "8B": 4096}
        self.head = nn.Linear(version_to_input_size[version] * 3, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.head(x)

    def compute_scores_and_preds(self, x):
        with torch.no_grad():
            logits = self.head(x)

        # score is the softmax value of the neuron assigned to the "relevant" class (ranking), while pred is simply the
        # argmax between "relevant" and "irrelevant" (classification)
        return self.softmax(logits)[:, 1].cpu().tolist(), torch.argmax(logits, dim=1).cpu().tolist()