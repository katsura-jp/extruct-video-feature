import torch
import torch.nn as nn

class C3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, num_classes, pretrained=False):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.__init_weight()


    def forward(self, x):
        # x : [1, 3, 16, 112, 112]
        x = self.relu(self.conv1(x)) # [1, 64, 16, 112, 112]
        x = self.pool1(x) # [1, 64, 16, 56, 56] | [1, 64, 16, 56, 56]

        x = self.relu(self.conv2(x)) # [1, 128, 16, 56, 56] | [1, 128, 16, 56, 56]
        x = self.pool2(x) # [1, 128, 8, 28, 28]) | [1, 128, 8, 28, 28]

        x = self.relu(self.conv3a(x)) # [1, 256, 8, 28, 28] | [1, 256, 8, 28, 28]
        x = self.relu(self.conv3b(x)) # [1, 256, 8, 28, 28] | [1, 256, 8, 28, 28]
        x = self.pool3(x) # [1, 256, 4, 14, 14] | [1, 256, 4, 14, 14]

        x = self.relu(self.conv4a(x)) # [1, 512, 4, 14, 14] | [1, 512, 4, 14, 14]
        x = self.relu(self.conv4b(x)) # [1, 512, 4, 14, 14] | [1, 512, 4, 14, 14]
        x = self.pool4(x) # [1, 512, 2, 7, 7] | [1, 512, 2, 7, 7]

        x = self.relu(self.conv5a(x)) # [1, 512, 2, 7, 7] | [1, 512, 2, 7, 7]
        x = self.relu(self.conv5b(x)) # [1, 512, 2, 7, 7] | [1, 512, 2, 7, 7]
        x = self.pool5(x) # [1, 512, 1, 4, 4] | [1, 512, 1, 4, 4]

        x = x.view(-1, 8192) # [1, 8192] | [1, 8192]
        x = self.relu(self.fc6(x)) # | [1, 4096]
        x = self.dropout(x)
        x = self.relu(self.fc7(x)) # [1, 4096]
        x = self.dropout(x)

        logits = self.fc8(x)

        return logits

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
         model.conv5a, model.conv5b, model.fc6, model.fc7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 112, 112)
    net = C3D(num_classes=101, pretrained=False)

    outputs = net.forward(inputs)
    print(outputs.size())