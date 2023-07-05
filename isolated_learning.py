import os
import torch
import argparse
from tqdm import tqdm

from utils import Client
from utils import load_file, get_device

from models import ShallowNN

from torch.utils.tensorboard import SummaryWriter

device = get_device()

parser = argparse.ArgumentParser(description="Isolated client training parameters")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int,  default=1000)
parser.add_argument("--learning_rate", type=float, default=0.00005)
args = parser.parse_args()

# Args
data_path = "../kv_data/kv/"
checkpt_path = "checkpt/isolated/"

features = 197

# Hyper Parameters
loss_fn = torch.nn.MSELoss() 
batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.learning_rate

writer = SummaryWriter(comment="_isolated_training_batch_size_"+str(batch_size))

files = os.listdir(data_path)
files_path = [os.path.join(data_path, file) for file in files]
clients = [Client(i+1, load_file(files_path[i])) for i in range(len(files_path))]

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
