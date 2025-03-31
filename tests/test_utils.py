from torch import nn

from levee_hunter.utils import count_parameters


def test_count_parameters():
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(2, 3)  # 2 inputs, 3 outputs
            self.fc2 = nn.Linear(3, 1)  # 3 inputs, 1 output

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    model = SimpleNN()
    assert count_parameters(model) == 13
