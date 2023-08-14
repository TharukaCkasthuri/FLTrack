import time
import torch
import argparse

import pandas as pd

from tqdm import tqdm
from utils import Client
from utils import  get_device, evaluate

from models import ShallowNN

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

class Federation():

    def __init__(self, checkpt_path, features, loss_fn, batch_size, epochs, learning_rate):
        self.checkpt_path = checkpt_path
        self.features = features
        self.loss_fn = loss_fn
        self.batch_size =  batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.writer = SummaryWriter(comment="_fed_train_batch"+str(batch_size))

    def set_clients(self,client_ids):
        self.client_ids = client_ids
        self.clients = [Client(id, torch.load("trainpt/"+id+".pt"), torch.load("testpt/"+id+".pt"), batch_size) for id in self.client_ids]

        self.client_model_dict = {}
        for i in self.client_ids:
            self.client_model_dict[i] = ShallowNN(features) 

        test_dataset = torch.utils.data.ConcatDataset([torch.load("testpt/"+id+".pt") for id in self.client_ids])
        self.test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

    def train(self, model):

        # initiate global model
        global_model = model(self.features)
        #global_model.to(device)
        global_model.train()
        
        global_weights = global_model.state_dict()
        model_layers = global_model.track_layers.keys()

        training_stats = []   
        for epoch in tqdm(range(epochs)):
            local_models, local_loss = [], []

            print(f'\n | Global Training Round : {epoch+1} |\n')

            for client in self.clients:

                client_id = client.client_id
                training_stat_dict = {"client_id": client_id, "training_round": epoch}

                # loading the global model weights to client model
                self.client_model_dict[client_id].load_state_dict(global_weights)

                # setting up the optimizer
                optimizer = torch.optim.SGD(
                    self.client_model_dict[client_id].parameters(), lr=learning_rate)
                
                # training
                this_client_state_dict, training_loss = client.train(
                    self.client_model_dict[client_id], loss_fn, optimizer, epoch)
                local_models.append(this_client_state_dict)
                local_loss.append(training_loss)
                
                training_stat_dict["fed_train"] = training_loss
                validation_loss = client.eval(self.client_model_dict[client_id], loss_fn)
                training_stat_dict["fed_val"] = validation_loss

                training_stats.append(training_stat_dict)  

            # update global model parameters here
            state_dicts = [model.state_dict() for model in local_models]
            for key in model_layers:
                global_model.track_layers[key].weight.data = torch.stack([item[str(key)+ ".weight"] for item in state_dicts]).mean(dim=0)
                global_model.track_layers[key].bias.data = torch.stack([item[str(key)+ ".bias"] for item in state_dicts]).mean(dim=0)
                # info here - https://discuss.pytorch.org/t/how-to-change-weights-and-bias-nn-module-layers/93065/2

            global_weights = global_model.state_dict()

        return global_model, training_stats
            

    def save_stats(self,model,training_stat,path_det:str=None):
        stats_dataframe =  pd.DataFrame.from_dict(training_stat).to_csv("losses/" +path_det+ "_fed_stats_epoch_"+str(epochs)+".csv", index=False)
        torch.save(model.state_dict(), checkpt_path + path_det+"_fedl_global_"+str(epochs)+".pth")


   
if __name__ == "__main__":

    device = get_device()
    parser = argparse.ArgumentParser(description="Federated training parameters")
    parser.add_argument("--batch_size",type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning_rate",type=float, default=0.005)
    args = parser.parse_args()

    checkpt_path = "checkpt/"
    features = 197

    # Hyper Parameters
    loss_fn = torch.nn.MSELoss() 
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate

    fed = Federation(checkpt_path, features, loss_fn, batch_size, epochs, learning_rate)
    client_ids = ["0_0","0_1","0_2","0_3","0_4","0_5","1_0","1_1","1_2","1_3","1_4","1_5","2_0","2_1","2_2","2_3","2_4","2_5","3_0","3_1","3_2","3_3","3_4","3_5"]

    skips_list = ["0_1","0_2","0_4","0_5","1_3"]

    for item in skips_list:
        try:
            client_ids.remove(item)
        except:
            pass

    print("Federation with clients " + ', '.join(client_ids))

    for item in skips_list:
        start = time.time()
        fed.set_clients(client_ids=client_ids)
        trained_model, training_stats = fed.train(ShallowNN)
        fed.save_stats(trained_model, training_stats, path_det="influence/" + str(item))
        print(str((time.time()-start)/60) + " minutes")
        