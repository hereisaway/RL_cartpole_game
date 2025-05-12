from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1=nn.Linear(4,128)
        self.fc2=nn.Linear(128,2)
        self.dropout=nn.Dropout(0.6)
    def forward(self,x):
        x=self.fc1(x)
        x=self.dropout(x)
        x=F.relu(x)
        x=self.fc2(x)
        x=F.softmax(x,dim=1)
        return x
