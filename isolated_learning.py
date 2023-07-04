import os
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader

from utils import Client, CustomDataSet
from utils import load_file, get_device

import torch.optim as optim

from models import ShallowNN

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# Args
data_path = "../kv_data/kv/"
n_clients = 24
checkpt_path = "checkpt/isolated/"

files = os.listdir(data_path)
files_path = [os.path.join(data_path, file) for file in files]

device = get_device()

uids = [u for u in range(n_clients)]

# Hype Parameters
loss_fn = torch.nn.MSELoss()  # nn.MSELoss()
batch_size = 128
features = 197

clients = [Client(i+1, load_file(files_path[i]))
           for i in range(len(files_path))]

epochs = 1000
local_round_count = 10
learning_rate = 0.00005


for client in clients:
    client_id = client.get_client_id()
    client.set_model(ShallowNN(features))
    
    optimizer = torch.optim.SGD(
            client.get_model().parameters(), lr=learning_rate)
    
    for epoch in tqdm(range(epochs)):

        client_model, client_loss = client.train(
            client.get_model(), loss_fn, optimizer, batch_size, epoch)
        writer.add_scalar("Client_"+str(client_id) +
                          " Training Loss", client_loss, epoch)
    model_path =  checkpt_path + "client_" + str(client_id) +".pth"
    client.save_model(model_path)
