import torch
import torch.nn as nn


class VanillaNet(nn.Module):
    # 简单的前馈神经网络，包含两个全连接层（也称为线性层）。
    # 它接受大小为1280的输入，输出大小为out_dim的张量。两个隐藏层之间使用ReLU作为激活函数。
    def __init__(self, hidden_dim, out_dim, device, dtype):
        super(VanillaNet, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(1280, hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LayerNormNet(nn.Module):
    # 与VanillaNet类似，但在每个隐藏层之后都加入了层归一化。层归一化会独立地对特征维度上的激活值进行归一化。
    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(LayerNormNet, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(1280, hidden_dim, dtype=dtype, device=device)
        #层归一化是在每个样本（单个数据点）的特征维度上进行归一化
        self.ln1 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,
                             dtype=dtype, device=device)
        self.ln2 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.ln2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x
 


class BatchNormNet(nn.Module):
    # 与VanillaNet类似，但在每个隐藏层之后都加入了批归一化。批归一化会对批次维度上的激活值进行归一化。
    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(BatchNormNet, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(1280, hidden_dim, dtype=dtype, device=device)
        self.bn1 = nn.BatchNorm1d(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,
                             dtype=dtype, device=device)
        self.bn2 = nn.BatchNorm1d(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.dropout(self.bn1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.bn2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x
    
 
class InstanceNorm(nn.Module):
    # 与VanillaNet类似，但在每个隐藏层之后都加入了实例归一化。
    # 实例归一化会对特征维度上的激活值独立地进行归一化，每个样本在批次中都有不同的归一化参数。
    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(InstanceNorm, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(1280, hidden_dim, dtype=dtype, device=device)
        # 实例归一化是对每个样本单独进行归一化，而不依赖于其他样本
        self.in1 = nn.InstanceNorm1d(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,
                             dtype=dtype, device=device)
        self.in2 = nn.InstanceNorm1d(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.dropout(self.in1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.in2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x
 

