import os
import torch
import argparse

from tqdm import tqdm
from utils import Client
from utils import  get_device, evaluate

from models import ShallowNN

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

device = get_device()

parser = argparse.ArgumentParser(description="Federated training parameters")
parser.add_argument("--batch_size",type=int, default=128)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--learning_rate",type=float, default=0.00005)
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

writer = SummaryWriter(comment="_federated_training_batch_size_"+str(batch_size))

client_ids = ["0_0","0_1","0_2","0_3","0_4","0_5","1_0","1_1","1_2","1_3","1_4","1_5","2_0","2_1","2_2","2_3","2_4","2_5","3_0","3_1","3_2","3_3","3_4","3_5"]
clients = [Client(id, torch.load("trainpt/"+id+".pt"), torch.load("testpt/"+id+".pt"), batch_size) for id in client_ids]

test_dataset = torch.utils.data.ConcatDataset([torch.load("testpt/"+id+".pt") for id in client_ids])
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

# initiate global model
global_model = ShallowNN(features)
#global_model.to(device)
global_model.train()
global_weights = global_model.state_dict()
model_layers = global_model.track_layers.keys()


for epoch in tqdm(range(epochs)):

    local_models, local_loss = [], []

    print(f'\n | Global Training Round : {epoch+1} |\n')

    for client in clients:

        client_id = client.client_id
        client_model = ShallowNN(features)

        # loading the global model weights to client model
        client_model.load_state_dict(global_weights)

        # setting up the optimizer
        optimizer = torch.optim.SGD(
            client_model.parameters(), lr=learning_rate)
        
        # training
        this_client_state_dict, client_loss = client.train(
            client_model, loss_fn, optimizer, epoch)
        local_models.append(this_client_state_dict)
        local_loss.append(client_loss)
        
        validation_loss = client.eval(client_model, loss_fn)

        writer.add_scalars("Client_"+str(client_id) +
                          " Loss", {"Training Loss":client_loss, "Validation Loss": validation_loss}, epoch)
        
        validation_loss = client.eval(client_model, loss_fn)
        
    # update global model parameters here
    state_dicts = [model.state_dict() for model in local_models]
    for key in model_layers:
        global_model.track_layers[key].weight.data = torch.stack([item[str(key)+ ".weight"] for item in state_dicts]).mean(dim=0)
        global_model.track_layers[key].bias.data = torch.stack([item[str(key)+ ".bias"] for item in state_dicts]).mean(dim=0)
        # info here - https://discuss.pytorch.org/t/how-to-change-weights-and-bias-nn-module-layers/93065/2


    global_weights = global_model.state_dict()

    global_training_loss = sum(local_loss)/len(local_loss)
    validation_loss, mse, _ = evaluate(global_model,test_dataloader,loss_fn)

    writer.add_scalars('Global Model - Federated Learning', {'Training Loss': global_training_loss,
                                    'Validation Loss': validation_loss,
                                    'Validation MSE': mse}, epoch)

writer.flush()
writer.close()

global_model.eval()
torch.save(global_model.state_dict(), checkpt_path+"_"+str(batch_size)+"_fedl_global.pth")
