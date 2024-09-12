import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,dims=[10,50,50,2]) -> None:
        super().__init__()
        layers=[]
        norms=[]
        for dim_in,dim_out in zip(dims[:-1],dims[1:]):
            layers.append(nn.Linear(in_features=dim_in,out_features=dim_out))
            norms.append(nn.LayerNorm(dim_out))
        self.layers=nn.ModuleList(layers)
        self.norms=nn.ModuleList(norms)
        self.activate=nn.LeakyReLU()
    
    def forward(self,x):
        for l,n in zip(self.layers[:-1],self.norms[:-1]):
            x=l(x)
            x=self.activate(x)
            x=n(x)
        x=self.layers[-1](x)
        return x
