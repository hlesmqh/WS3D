import torch
import numpy as np
def distance_2(a,b=torch.zeros(1,2).cuda()): return torch.sqrt(torch.sum((a[None,:] - b[:,None]) ** 2,dim=2))

def distance_2_numpy(a,b): return np.sqrt(np.sum((a[None,:] - b[:,None]) ** 2,axis=2))

def cos_distance(a,b=torch.zeros(1,2).cuda()):
    return torch.sum((a[None,:] * b[:,None]),dim=2)/torch.sum(b[:,:]**2,dim=1).view(-1,1)

def cos_matrix_distance(a,b=torch.zeros(1,2).cuda(),c=torch.zeros(1,2).cuda()):
    #matrix = (a[None, :] - b[:, None])
    matrix = torch.sum((a[None, :] - b[:, None])* c.view(-1,1,2), dim=2)/torch.sum(c[:,:]**2,dim=1).view(-1,1)
    return matrix

def xz_distance(a,b=torch.zeros(1,2).cuda()):
    return (a[None,:] - b[:,None])
def distance_2_numpy(a,b): return np.sqrt(np.sum((a[None,:] - b[:,None]) ** 2,axis=2))