import torch


class ShallowNN(torch.nn.Module):
    """
    Shallow Nueral Network model with three layers.

    Parameters:
    ------------
    feats: int; number of features
    """

    def __init__(self, feats):
        super(ShallowNN, self).__init__()
        self.layer_1 = torch.nn.Linear(feats, 64)
        self.relu_1 = torch.nn.ReLU()
        self.layer_2 = torch.nn.Linear(64, 32)
        self.relu_2 = torch.nn.ReLU()
        self.layer_3 = torch.nn.Linear(32, 1)
        self.track_layers = {
            "layer_1": self.layer_1,
            "layer_2": self.layer_2,
            "layer_3": self.layer_3,
        }

    def forward(self, inputs):
        """
        Forward pass.

        Parameters:
        ------------
        inputs: torch.tensor object; input data

        Returns:
        ------------
        x: torch.tensor object; output data
        """
        x = self.relu_1(self.layer_1(inputs))
        x = self.relu_2(self.layer_2(x))
        x = self.layer_3(x)
        return x


### When the feats = 197,