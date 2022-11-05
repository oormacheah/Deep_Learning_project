# -------------------------
# Copied from load_data.py
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

train_dir = './train_images'
test_dir = './test_images'

transform = transforms.Compose(
    [transforms.Grayscale(), 
     transforms.ToTensor(), 
     transforms.Normalize(mean=(0,),std=(1,))])

train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)

valid_size = 0.2
batch_size = 32

num_train = len(train_data)
indices_train = list(range(num_train))
np.random.shuffle(indices_train)
split_tv = int(np.floor(valid_size * num_train))
train_new_idx, valid_idx = indices_train[split_tv:],indices_train[:split_tv]

train_sampler = SubsetRandomSampler(train_new_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=1)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)

classes = ('noface','face')
# -------------------

# Our implementation

from net import Net

def main():

    n_epochs = 1

    # Hyperparameters
    lr = 0.01
    momentum = 0.5

    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum) # May select a different optimizer, like Adam

    for epoch in range(1, n_epochs+1):
        for data, target in train_loader:
            '''
            According to documentation
            data = inputs
            target = labels
            '''
            # Zeroing gradient for each batch
            optimizer.zero_grad()

            # Make predictions
            outputs = model(data)
            


            exit()

if __name__ == '__main__':
    main()