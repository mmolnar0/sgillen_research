
# coding: utf-8


import torch
import torch.nn as nn
#from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.functional import softmax


class Router(nn.Module):
    def __init__(self, input_size, hidden_size, router_size):
        super().__init__()
        
        # routing layer gates
        self.G1 = Parameter(torch.Tensor(input_size, router_size))
        self.G2 = Parameter(torch.Tensor(router_size, 2))
        
        self.W1 = Parameter(torch.Tensor(input_size, hidden_size))
        self.W2 = Parameter(torch.Tensor(hidden_size, 1))
        
        self.K = Parameter(torch.Tensor(input_size,1))
    
    def forward(self, x):
        g1 = torch.sigmoid(x.matmul(self.G1))
        g2 = torch.sigmoid(g1.matmul(self.G2))
        d = softmax(g2, dim=0)
        return g1,g2,d,x
        

if __name__ == "__main__":
    net = Router(2,4,4)
    (g1, g2, d, x) = net(torch.randn(2))
    print("g1: ", g1)
    print("g2: ", g2)
    print("d: ", d)
    print("x:", x)
    print()
    print("G1", net.G1)
    print("G2", net.G2)


