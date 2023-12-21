import torch
import torch.nn as nn 
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.pool = nn.MaxPool2d(2,2) #kernel + stride
        self.conv2 = nn.Conv2D(10,20,kernel_size=5)
        self.drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20 * 4 * 4, 50)
        self.fc2 = nn.Linear(50,10)

    def forward(self,x):
        #Convolution -> Activation -> Dropout -> Pooling

        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.drop(x)
        x = self.pool(x)

        x = torch.flatten(x,1)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        return x

        # If you are using cross-entropy loss, simply
        # return the input x.
        # When using negative log-likelihood, make sure
        # to convert x to softmax outputs of probabilities
        # by doing extra step of return x = F.log_softmax(x) or log(F.softmax(x)) 

    #   super().__init__()
    #     self.conv1 = nn.Conv2d(3, 6, 5)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     self.fc1 = nn.Linear(16 * 5 * 5, 120)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, 10)

    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = torch.flatten(x, 1) # flatten all dimensions except batch
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x