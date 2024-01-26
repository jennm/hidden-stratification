from torch import nn

class LogisticRegression(nn.Module):
    def __init__(self, dim, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        return nn.functional.softmax(self.linear(x), dim=1)