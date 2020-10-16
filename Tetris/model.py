import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self):
        """DQN Network

        Args:
            input_dim (int): `state` dimension.
                `state` is 2-D tensor of shape (n, input_dim)
            output_dim (int): Number of actions.
                Q_value is 2-D tensor of shape (n, output_dim)
            hidden_dim (int): Hidden dimension in fc layer
        """
    
        super(QNetwork, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.PReLU()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.PReLU()
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.PReLU()
        )

        self.final = torch.nn.Linear(128, 1)
    
    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        """Returns a Q_value

        Args:
            x (torch.Tensor): `State` 2-D tensor of shape (n, input_dim)

        Returns:
            torch.Tensor: Q_value, 2-D tensor of shape (n, output_dim)            
        """
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final(x)

        return x
     
