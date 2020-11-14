from torch import nn
from layer import Fully_Connect0, Fully_Connect1, Fully_Connect2


class VggNet16(nn.Module):
    def __init__(self, fc_in=512, fc_dims1=4096, fc_dims2=4096, num_classes=10):
        super(VggNet16, self).__init__()
        self.features = nn.Sequential(
            # 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14
            # 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7
            # 5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 3
            # 8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 9
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),  # 2
            # 11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)   # 1
        )
        self.fc_in = fc_in
        self.fc_dims1 = fc_dims1
        self.fc_dims2 = fc_dims2
        self.num_classes = num_classes
        self.layer0 = Fully_Connect0(self.fc_in, self.fc_dims1)
        self.relu0 = nn.ReLU(inplace=True)
        self.layer1 = Fully_Connect1(self.fc_dims1, self.fc_dims2)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer2 = Fully_Connect2(self.fc_dims2, self.num_classes)
        self.layers = []
        for m in self.modules():
            if isinstance(m, Fully_Connect1) or isinstance(m, Fully_Connect2):
                self.layers.append(m)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, epoch):
        o = self.features(x)
        o = o.view(o.size(0), -1)
        # 17
        a = self.layer0(o, epoch)
        a_out = self.relu0(a)
        # 18
        b = self.layer1(a_out, epoch)
        b_out = self.relu1(b)
        # 19
        c = self.layer2(b_out, epoch)
        return c