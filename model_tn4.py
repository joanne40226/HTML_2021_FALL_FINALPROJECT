import torch
import torch.nn as nn
import torch.nn.functional as F

print("PointNet - dataset : 3D points_t x N x 4 - feature_withoutLabel")

class PointNetDenseCls(nn.Module):
    def __init__(self, k=40, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform = feature_transform

        # Updated input channels from 3 to 4
        self.conv1 = nn.Conv1d(4, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)

        # Feature transform network
        self.fstn = FeatureTransformNet()

    def forward(self, x):
        batchsize, t, n, c = x.size()
        x = x.reshape(batchsize * t, c, n)

        # Forward pass through PointNet
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.max(x, 2, keepdim=False)[0]  # Global max pooling
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        pred = self.fc3(x)

        # Reshape for time steps
        pred = pred.view(batchsize, t, -1)

        if self.feature_transform:
            trans = self.fstn(x)
            return pred, trans, trans
        else:
            return pred, None, None


class FeatureTransformNet(nn.Module):
    def __init__(self):
        super(FeatureTransformNet, self).__init__()
        self.conv1 = nn.Conv1d(1024, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 16)  # 4x4 matrix for 4D input

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.max(2, keepdim=False)[0]
        x = F.relu(self.fc1(x))
        trans = self.fc2(x)
        trans = trans.view(-1, 4, 4)
        return trans


def feature_transform_regularizer(trans):
    I = torch.eye(4).cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, p='fro'))
    return loss

