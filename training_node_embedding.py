import argparse
import pickle
import time
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam,SGD


from models.definitions.GAT import GAT
from utils.data_loading import load_graph_data
from utils.constants import *
import utils.utils as utils


class GAP_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, adj_matrix, probe_matrix, degree_vector):
        '''
        adj_matrix with shape (num_node, num_node)
        probe_matrix with shape (num_node, nparts)
        degree_vector with shape (num_node, 1)
        '''
        #print("adj_matrix:",adj_matrix.shape)
        Tao = torch.matmul(probe_matrix.transpose(1,0),degree_vector)[:,0]

        item_1 = torch.div(probe_matrix,Tao)
        #print("item_1:",item_1.shape)
        
        item_2 = torch.transpose((1-probe_matrix),1,0)
        #print("item_2:",item_2.shape)

        loss_ = torch.mul(torch.matmul(item_1,item_2),adj_matrix)
        #return torch.mean(loss_)
        return torch.sum(loss_)


# Simple decorator function so that I don't have to pass arguments that don't change from epoch to epoch
def get_main_loop(config, gat, loss_fn, optimizer, node_features, adj_matrix, degree_vector, total_demand, edge_index, train_indices, val_indices, test_indices, patience_period, time_start):

    # node_features shape = (N, FIN), edge_index shape = (2, E)
    graph_data = (node_features, edge_index)  # I pack data into tuples because GAT uses nn.Sequential which requires it

    def get_node_indices(phase):
        if phase == LoopPhase.TRAIN:
            return train_indices
        elif phase == LoopPhase.VAL:
            return val_indices
        else:
            return test_indices

    def main_loop(phase, epoch=0):
        global BEST_VAL_PERF, BEST_VAL_LOSS, PATIENCE_CNT, writer

        # Certain modules behave differently depending on whether we're training the model or not.
        # e.g. nn.Dropout - we only want to drop model weights during the training.
        if phase == LoopPhase.TRAIN:
            gat.train()
        else:
            gat.eval()

        select_indices = get_node_indices(phase)

        probe_matrix = gat(graph_data)
        #print("probe_matrix:",probe_matrix.shape)

        #loss = cross_entropy_loss(nodes_unnormalized_scores, gt_node_labels)
        local_adj_matrix = adj_matrix[select_indices]
        local_adj_matrix = local_adj_matrix[:,select_indices]
        local_probe_matrix = probe_matrix[select_indices]
        local_degree_vector = degree_vector[select_indices]

        loss = loss_fn(local_adj_matrix, local_probe_matrix, local_degree_vector)

        #print("loss:",loss)

        if phase == LoopPhase.TRAIN:
            optimizer.zero_grad()  # clean the trainable weights gradients in the computational graph (.grad fields)
            loss.backward()  # compute the gradients for every trainable weight in the computational graph
            optimizer.step()  # apply the gradients to weights
        #
        # Logging
        #

        if phase == LoopPhase.TRAIN:
            # Log metrics
            if config['enable_tensorboard']:
                writer.add_scalar('training_loss', loss.item(), epoch)
                #writer.add_scalar('training_acc', accuracy, epoch)

            # Save model checkpoint
            if config['checkpoint_freq'] is not None and (epoch + 1) % config['checkpoint_freq'] == 0:
                ckpt_model_name = f'gat_{config["dataset_name"]}_ckpt_epoch_{epoch + 1}.pth'
                config['test_perf'] = -1
                torch.save(utils.get_training_state(config, gat), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))

        elif phase == LoopPhase.VAL:
            # Log metrics
            if config['enable_tensorboard']:
                writer.add_scalar('val_loss', loss.item(), epoch)
                #writer.add_scalar('val_acc', accuracy, epoch)

            # Log to console
            if config['console_log_freq'] is not None and epoch % config['console_log_freq'] == 0:
                print(f'GAT training: time elapsed= {(time.time() - time_start):.2f} [s] | epoch={epoch + 1} | val loss={loss.item()}')

            # The "patience" logic - should we break out from the training loop? If either validation acc keeps going up
            # or the val loss keeps going down we won't stop
            if  loss.item() < BEST_VAL_LOSS:
                #BEST_VAL_PERF = max(accuracy, BEST_VAL_PERF)  # keep track of the best validation accuracy so far
                BEST_VAL_LOSS = min(loss.item(), BEST_VAL_LOSS)  # and the minimal loss
                PATIENCE_CNT = 0  # reset the counter every time we encounter new best accuracy
            else:
                PATIENCE_CNT += 1  # otherwise keep counting

            if PATIENCE_CNT >= patience_period:
                raise Exception('Stopping the training, the universe has no more patience for this training.')

        else:
            return loss.item()  # in the case of test phase we just report back the test accuracy

    return main_loop  # return the decorated function


def load_graph_by_path_cly(config, device):
    
    # loading adj_matrix
    with open(config["adj_path"],"rb") as fp:
        graph_info = pickle.load(fp)
    
    total_demand = torch.tensor(graph_info["total_deman"],dtype=torch.float32,device=device)
    adj_matrix = torch.tensor(graph_info["adj_matrix"],dtype=torch.float32,device=device)

    degree_vector = torch.sum(adj_matrix,dim=1,keepdim=True)
    num_node = len(adj_matrix)

    # specify the random seed
    torch.manual_seed(int(config["random_seed"]))
    np.random.seed(int(config["random_seed"]))

    # init the node embedding
    node_features = torch.randn((num_node,config["embedding_dims"]),device=device)

    # Indices that help us extract nodes that belong to the train/val and test splits
    train_len, val_len, test_len = [int(item*num_node) for item in config["dateset_split_ratio"]]
    node_indices = np.random.permutation(np.arange(num_node))
    
    train_indices = torch.tensor(node_indices[:train_len],dtype=torch.long, device=device)
    val_indices = torch.tensor(node_indices[train_len:train_len+val_len],dtype=torch.long, device=device)
    test_indices = torch.tensor(node_indices[-test_len:],dtype=torch.long, device=device)

    # adjacent relationship
    topology = torch.tensor(np.where(adj_matrix==1),device=device)

    #embedding_dims
    return node_features, adj_matrix, degree_vector, total_demand, topology, train_indices, val_indices, test_indices


def train_gat_gap(config):
    global BEST_VAL_PERF, BEST_VAL_LOSS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

    # Step 1: load the graph data
    #node_features, node_labels, edge_index, train_indices, val_indices, test_indices = load_graph_data(config, device)

    node_features, adj_matrix, degree_vector, total_demand, edge_index, train_indices, val_indices, test_indices = load_graph_by_path_cly(config, device)


    print("node_features",node_features.shape)
    print("adj_matrix",adj_matrix.shape)
    print("degree_vector",degree_vector.shape)
    print("total_demand",total_demand.shape)

    #print("unique(node_labels)",np.unique(node_labels))

    print("edge_index",edge_index.shape)
    print("edge_index max:",max(edge_index[0]))
    print("edge_index max:",max(edge_index[1]))
    print("edge_index min:",min(edge_index[0]))
    print("edge_index min:",min(edge_index[1]))

    #print("edge_index sample:",edge_index[:,:10])
    print("train_indices",train_indices.shape)
    print("val_indices",val_indices.shape)
    print("test_indices",test_indices.shape)

    # Step 2: prepare the model
    gat = GAT(
        num_of_layers=config['num_of_layers'],
        num_heads_per_layer=config['num_heads_per_layer'],
        num_features_per_layer=config['num_features_per_layer'],
        add_skip_connection=config['add_skip_connection'],
        bias=config['bias'],
        dropout=config['dropout'],
        layer_type=config['layer_type'],
        log_attention_weights=False,  # no need to store attentions, used only in playground.py for visualizations
        nparts=config["nparts"]
    ).to(device)

    # Step 3: Prepare other training related utilities (loss & optimizer and decorator function)
    #loss_fn = nn.CrossEntropyLoss(reduction='mean')
    loss_fn = GAP_loss()

    optimizer = Adam(gat.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    #optimizer = SGD(gat.parameters(), lr=config['lr'], momentum=0.9)

    # The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops
    main_loop = get_main_loop(
        config,
        gat,
        loss_fn,
        optimizer,
        node_features,
        adj_matrix, 
        degree_vector,
        total_demand,
        edge_index,
        train_indices,
        val_indices,
        test_indices,
        config['patience_period'],
        time.time())

    BEST_VAL_PERF, BEST_VAL_LOSS, PATIENCE_CNT = [0, 0, 0]  # reset vars used for early stopping

    # Step 4: Start the training procedure
    for epoch in range(config['num_of_epochs']):
        # Training loop
        main_loop(phase=LoopPhase.TRAIN, epoch=epoch)

        # Validation loop
        with torch.no_grad():
            try:
                main_loop(phase=LoopPhase.VAL, epoch=epoch)
            except Exception as e:  # "patience has run out" exception :O
                print(str(e))
                break  # break out from the training loop

    # Step 5: Potentially test your model
    # Don't overfit to the test dataset - only when you've fine-tuned your model on the validation dataset should you
    # report your final loss and accuracy on the test dataset. Friends don't let friends overfit to the test data. <3
    if config['should_test']:
        test_loss = main_loop(phase=LoopPhase.TEST)
        config['test_perf'] = test_loss
        print(f'Test loss = {test_loss}')
    else:
        config['test_perf'] = -1

    # Save the latest GAT in the binaries directory
    torch.save(
        utils.get_training_state(config, gat),
        os.path.join(BINARIES_PATH, utils.get_available_binary_name(config['dataset_name']))
    )


def get_training_args():
    parser = argparse.ArgumentParser()

    # Training related
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=10000)
    parser.add_argument("--patience_period", type=int, help="number of epochs with no improvement on val before terminating", default=1000)
    parser.add_argument("--lr", type=float, help="model learning rate", default=5e-4)
    parser.add_argument("--weight_decay", type=float, help="L2 regularization on model weights", default=5e-4)
    parser.add_argument("--should_test", action='store_true', help='should test the model on the test dataset? (no by default)')

    # Dataset related
    parser.add_argument("--dataset_name", choices=[el.name for el in DatasetType], help='dataset to use for training', default=DatasetType.CORA.name)
    parser.add_argument("--should_visualize", action='store_true', help='should visualize the dataset? (no by default)')

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable_tensorboard", action='store_true', help="enable tensorboard logging (no by default)")
    parser.add_argument("--console_log_freq", type=int, help="log to output console (epoch) freq (None for no logging)", default=100)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (epoch) freq (None for no logging)", default=1000)

    # GAP parameter
    parser.add_argument("--embedding_dims", type=int, help="the dimensions of node embedding.", default=64)
    parser.add_argument("--adj_path", type=str, help="adj_path.", default="/Users/chenliyue/Documents/GitHub/pytorch-GAT/adj_data/adj_demand_1848.pkl")
    parser.add_argument("--random_seed", type=int, help="random_seed.", default=42)
    parser.add_argument("--nparts", type=int, help="number of parts.", default=600)
    
    args = parser.parse_args()

    # Model architecture related
    gat_config = {
        "num_of_layers": 2,  # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "num_heads_per_layer": [4, 2],
        "num_features_per_layer": [args.embedding_dims, 16, 32],
        "add_skip_connection": False,  # hurts perf on Cora
        "bias": True,  # result is not so sensitive to bias
        "dropout": 0.6,  # result is sensitive to dropout
        "layer_type": LayerType.IMP3,  # fastest implementation enabled by default
        "dateset_split_ratio": [0.4,0.4,0.2] # train/test/val
    }

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    # Add additional config information
    training_config.update(gat_config)

    return training_config


if __name__ == '__main__':

    # Train the graph attention network (GAT)
    train_gat_gap(get_training_args())
