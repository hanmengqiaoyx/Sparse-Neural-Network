from torch import nn
from layer import Fully_Connect0, Fully_Connect1, Fully_Connect2


class Fully_Connect_Net(nn.Module):
    def __init__(self, fc_in=784, fc_dims1=1024, fc_dims2=512, num_classes=10):
        super(Fully_Connect_Net, self).__init__()
        self.fc_in = fc_in
        self.fc_dims1 = fc_dims1
        self.fc_dims2 = fc_dims2
        self.num_classes = num_classes
        self.layer0 = Fully_Connect0(self.fc_in, self.fc_dims1)
        self.layer1 = Fully_Connect1(self.fc_dims1, self.fc_dims2)
        self.layer2 = Fully_Connect2(self.fc_dims2, self.num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.layers = []
        for m in self.modules():
            if isinstance(m, Fully_Connect1) or isinstance(m, Fully_Connect2):
                self.layers.append(m)

    def forward(self, x, epoch):
        o = x.view(x.size(0), -1)
        x = self.layer0(o)
        x = self.relu(x)
        x = self.layer1(x, epoch)
        x = self.relu(x)
        out = self.layer2(x, epoch)
        return out