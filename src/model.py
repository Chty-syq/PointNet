import torch
import torch.nn.functional as F


class TNet(torch.nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k

        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.pool = torch.nn.MaxPool1d(1024)
        self.flat = torch.nn.Flatten(1)

        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, k * k)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)

    def forward(self, input):  # input: [bsize, n, 3]
        bsize = input.shape[0]
        res = F.relu(self.bn1(self.conv1(input)))
        res = F.relu(self.bn2(self.conv2(res)))
        res = F.relu(self.bn3(self.conv3(res)))
        res = self.flat(self.pool(res))
        res = F.relu(self.bn4(self.fc1(res)))
        res = F.relu(self.bn5(self.fc2(res)))
        res = self.fc3(res)

        bias = torch.eye(self.k, requires_grad=True).repeat(bsize, 1, 1)
        if res.is_cuda:
            bias = bias.cuda()

        res = res.view(-1, self.k, self.k) + bias
        return res


class Transform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = TNet(3)
        self.feature_transform = TNet(64)

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.pool = torch.nn.MaxPool1d(1024)
        self.flat = torch.nn.Flatten(1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)

    def forward(self, input):  # input: [bsize, n, 3]
        matrix1 = self.input_transform(input)

        res = torch.bmm(torch.transpose(input, 1, 2), matrix1).transpose(1, 2)
        res = F.relu(self.bn1(self.conv1(res)))

        matrix2 = self.feature_transform(res)

        res = torch.bmm(torch.transpose(res, 1, 2), matrix2).transpose(1, 2)

        local_feature = res

        res = F.relu(self.bn2(self.conv2(res)))
        res = self.bn3(self.conv3(res))

        global_feature = self.flat(self.pool(res))
        return global_feature, local_feature, matrix1, matrix2


class Classifier(torch.nn.Module):
    def __init__(self, k=10):
        super().__init__()
        self.transform = Transform()
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, k)

        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)

        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input):
        res, _, matrix1, matrix2 = self.transform(input)
        res = F.relu(self.bn1(self.fc1(res)))
        res = F.relu(self.bn2(self.fc2(res)))
        res = self.fc3(res)
        res = self.logsoftmax(res)
        return res, matrix1, matrix2


class Segmentor(torch.nn.Module):
    def __init__(self, m=4):
        super().__init__()
        self.transform = Transform()
        self.fc1 = torch.nn.Conv1d(1088, 512, 1)
        self.fc2 = torch.nn.Conv1d(512, 256, 1)
        self.fc3 = torch.nn.Conv1d(256, 128, 1)
        self.fc4 = torch.nn.Conv1d(128, m, 1)

        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.bn3 = torch.nn.BatchNorm1d(128)

        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input):
        global_feature, local_feature, matrix1, matrix2 = self.transform(input)

        res = torch.cat((global_feature, local_feature), dim=1)
        res = F.relu(self.bn1(self.fc1(res)))
        res = F.relu(self.bn2(self.fc2(res)))
        res = F.relu(self.bn3(self.fc3(res)))
        res = self.fc4(res)
        res = self.logsoftmax(res)

        return res, matrix1, matrix2




