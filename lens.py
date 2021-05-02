import torch

class Lens:
    def __init__(self, getter, setter):
        self.getter = getter
        self.setter = setter
    def get(self, v):
        return self.getter(v)
    def set(self, v,s):
        return self.setter(v,s)

def compose(lens0,lens1):
    return Lens(lambda v: lens1.get(lens0.get(v)),
                lambda v,s: lens1.set(lens0.get(v),s))

def setv(tensor,idx,v):
    t = tensor.clone().detach()
    t[idx]=v;
    return t
    

def index2lens(idx):
    return Lens(lambda v: v[idx],
                lambda v, s: setv(v, idx, s))

a = torch.zeros([3,4,5])

lens0 = index2lens((slice(None),0,slice(None)))
lens1 = index2lens((slice(None),1,slice(None)))

print(a.shape)
print(lens0.get(a).shape)
print(lens0.set(a,torch.tensor(1)))
print(lens1.set(a,torch.tensor(1)))
print(a.shape)



