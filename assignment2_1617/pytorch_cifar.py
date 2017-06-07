import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

import timeit

class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start = 0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

NUM_TRAIN = 49000
NUM_VAL = 1000

cifar10_train = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                           transform=T.ToTensor())
loader_train = DataLoader(cifar10_train, batch_size=64, sampler=ChunkSampler(NUM_TRAIN, 0))

cifar10_val = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                           transform=T.ToTensor())
loader_val = DataLoader(cifar10_val, batch_size=64, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))

cifar10_test = dset.CIFAR10('./cs231n/datasets', train=False, download=True,
                          transform=T.ToTensor())
loader_test = DataLoader(cifar10_test, batch_size=64)

import torchvision.models as models
gpu_dtype = torch.cuda.FloatTensor

print("-----Start importing ResNet34 model-----")

RN_model = models.resnet34(pretrained=False)
model_gpu = RN_model.type(gpu_dtype)

print("-----Done importing-----")
loss_fn = nn.CrossEntropyLoss().type(gpu_dtype)
optimizer = optim.SGD(model_gpu.parameters(), lr=0.001, momentum=0.9, dampening=0, weight_decay=0.0001, nesterov=True)

x = torch.randn(64, 3, 32, 32).type(gpu_dtype)
x_var = Variable(x.type(gpu_dtype))
ans = model_gpu(x_var)
print(np.array_equal(np.array(ans.size()), np.array([64, 10])))

# def train(model, loss_fn, optimizer, num_epochs = 1):
#     for epoch in range(num_epochs):
#         print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
#         model.train()
#         for t, (x, y) in enumerate(loader_train):
#             x_var = Variable(x.type(gpu_dtype))
#             y_var = Variable(y.type(gpu_dtype).long())

#             scores = model(x_var)
            
#             loss = loss_fn(scores, y_var)
#             if (t + 1) % print_every == 0:
#                 print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

# def check_accuracy(model, loader):
#     if loader.dataset.train:
#         print('Checking accuracy on validation set')
#     else:
#         print('Checking accuracy on test set')   
#     num_correct = 0
#     num_samples = 0
#     model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
#     for x, y in loader:
#         x_var = Variable(x.type(gpu_dtype), volatile=True)

#         scores = model(x_var)
#         _, preds = scores.data.cpu().max(1)
#         num_correct += (preds == y).sum()
#         num_samples += preds.size(0)
#     acc = float(num_correct) / num_samples
#     print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

# # torch.cuda.random.manual_seed(12345)
# # fixed_model_gpu.apply(reset)
# # train(fixed_model_gpu, loss_fn, optimizer, num_epochs=1)
# # check_accuracy(fixed_model_gpu, loader_val)

