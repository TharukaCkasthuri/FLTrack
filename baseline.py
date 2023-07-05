import os
import torch
import argparse

import pandas as pd
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import Client, CustomDataSet
from utils import load_file, get_device
from models import ShallowNN

from torch.utils.tensorboard import SummaryWriter

device = get_device()

parser = argparse.ArgumentParser(description="Baseline training parameters")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--learning_rate", type=float, default=0.00005)
args = parser.parse_args()

# Args
data_path = "../kv_data/kv/"
checkpt_path = "checkpt/"

features = 197

# Hyper Parameters
loss_fn = torch.nn.MSELoss()
batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.learning_rate

writer = SummaryWriter(comment="_baseline_training_batch_size_"+str(batch_size))

#init data
files = os.listdir(data_path)
files_path = [os.path.join(data_path,file) for file in files]
clients = [Client(i,load_file(files_path[i])) for i in range(len(files_path))]

#train data
x_train = pd.concat([client.get_x_train() for client in clients], axis=0,  ignore_index=True)
y_train = np.concatenate(tuple([client.get_y_train() for client in clients]))
train_dataset = CustomDataSet(x_train,y_train)
trainloader = DataLoader(train_dataset,batch_size,shuffle=True)


#model setup
model = ShallowNN(features)
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

for epoch in tqdm(range(epochs)):
    print("\n")
    batch_loss = []

    for batch_idx, (x, y) in enumerate(trainloader):
        outputs = model(x)
        loss = loss_fn(outputs, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 50 == 0:
                print('Epoch: {} \t[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(x), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
                
        batch_loss.append(loss.item())

    loss_avg = sum(batch_loss)/len(batch_loss)
    print('\nTrain loss:', loss_avg)
    writer.add_scalar("Baseline Training Loss", loss_avg, epoch)

writer.flush()
writer.close()

model.eval()
torch.save(model.state_dict(), checkpt_path+"_"+str(batch_size)+"_baseline.pth")