

import torch
import torchvision 
import numpy as np
import torch.utils.data as data 
import torchvision.transforms as transforms

import matplotlib.pyplot as plt 

BATCH_SIZE = 32 
transform = transforms.Compose([transforms.ToTensor()])

#Train data and train loader
train_data = torchvision.datasets.MNIST(root='./MNIST/', train = True, download = True, transform= transform)
train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle = True )

#Test data and test loader
test_data = torchvision.datasets.MNIST(root='./MNIST/', train = False, download= True, transform= transform)
test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle = False )

  
examples = enumerate(train_loader)
batch_idx, (images, labels) = next(examples) #images is [32, 1, 28, 28]

tmp = images[0] # [1, 28, 28]
tmp2 = tmp[0][0] 


import matplotlib.pyplot as plt

fig = plt.figure()
for i in range(BATCH_SIZE):
  plt.subplot(4,8,i+1)
  plt.tight_layout()
  plt.imshow(images[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(labels[i]))
  plt.xticks([])
  plt.yticks([])

plt.show()

print(1)
 