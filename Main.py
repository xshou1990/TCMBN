import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,f1_score,hamming_loss

import transformer.Constants as Constants
# import Utils

from preprocess.Dataset import get_dataloader
from transformer.Models import Transformer
from tqdm import tqdm
from copy import deepcopy




def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    print('[Info] Loading train data...')
    train_data, num_types = load_data(opt.data + 'train.pkl', 'train')
    print('[Info] Loading dev data...')
    dev_data, _ = load_data(opt.data + 'dev.pkl', 'dev')
    print('[Info] Loading test data...')
    test_data, _ = load_data(opt.data + 'test.pkl', 'test')

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    devloader = get_dataloader(dev_data, opt.batch_size, shuffle=False)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
    
    
    return trainloader, devloader, testloader, num_types#, inter_time_log_mean, inter_time_log_std

def train_epoch(model, training_data, optimizer, opt):
    """ Epoch operation in training phase. """

    model.train()


    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
        

        """ forward """
        optimizer.zero_grad()
        

        enc_out, non_pad_mask = model(event_type, event_time)
        
        a,b,c = enc_out[:,:-1,:].shape[0],enc_out[:,:-1,:].shape[1],enc_out[:,:-1,:].shape[2]
        

        """ backward """

        # calculate P*(y_i+1) by mbn: 
        log_loss_type = model.MBN.loss(enc_out[:,:-1,:].reshape(a*b,c), event_type[:,1:,:].reshape(a*b,model.num_types) )
        
        pred_type = (model.MBN.predict(enc_out[:,:-1,:].reshape(a*b,c) )[(non_pad_mask[:,1:,:].repeat(1,1, model.num_types)==1).reshape(a*b,model.num_types)]).reshape(-1,model.num_types)
        
        #bk loss
        #defi : 0:5, synthetic 0:2
        constraint_loss1 = torch.sum(torch.square( torch.sum(pred_type[:,0:2],dim=1)-1))
        constraint_loss2 = torch.sum(torch.square( torch.sum(pred_type[:,2:],dim=1)-1))
        
        combined_constraint_loss = constraint_loss1 + constraint_loss2
        
        # sum log loss
        loss = torch.sum(log_loss_type.reshape(a,b)  * non_pad_mask[:,1:,0]) + 0.001*combined_constraint_loss #
              
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """

    return loss


def eval_epoch(model, validation_data, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()
    
    total_ll = 0
    total_event = 0
    total_time_nll = 0
    pred_label = []
    true_label = []

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """

            event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

            enc_out, non_pad_mask = model(event_type, event_time)

            a,b,c = enc_out[:,:-1,:].shape[0],enc_out[:,:-1,:].shape[1],enc_out[:,:-1,:].shape[2]
            

            # calculate P*(y_i+1) by mbn: 
            log_loss_type = model.MBN.loss(enc_out[:,:-1,:].reshape(a*b,c), event_type[:,1:,:].reshape(a*b,model.num_types))
            
            pred_type = (model.MBN.predict(enc_out[:,:-1,:].reshape(a*b,c) )[(non_pad_mask[:,1:,:].repeat(1,1, model.num_types)==1).reshape(a*b,model.num_types)]).reshape(-1,model.num_types)
            
            
            pred_label += list(pred_type.cpu().numpy())

            true_type = (event_type[:,1:,:][(non_pad_mask[:,1:,:].repeat(1,1, model.num_types)==1)]).reshape(-1,model.num_types)
            
                      
            true_label += list(true_type.cpu().numpy())
           
            
            #  log loss
            loss = torch.sum(log_loss_type.reshape(a,b) * non_pad_mask[:,1:,0])

            """ note keeping """
            total_ll += loss.item()

             
    roc_auc = roc_auc_score(y_true=true_label, y_score=pred_label, average=None)
    
    print(" weighted roc_auc{}".format(roc_auc_score(y_true=true_label, y_score=pred_label, average='weighted')))
    
    
    roc_auc_mean = np.mean(roc_auc) 
    
    
    return total_ll, roc_auc_mean


def train(model, training_data, validation_data, test_data,  optimizer, scheduler, opt):
    """ Start training. """
    best_auc_roc = 0
    impatience = 0 
    best_model = deepcopy(model.state_dict())

    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event = train_epoch(model, training_data, optimizer, opt)
        print('  - (Train)    negative loglikelihood: {ll: 8.4f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=train_event, elapse=(time.time() - start) / 60))

        start = time.time()
        valid_event, valid_roc_auc = eval_epoch(model, validation_data, opt)
        print('  - (dev)    nll: {ll: 8.4f}, '
              ' roc auc : {type:8.4f},'
              'elapse: {elapse:3.3f} min'
              
              .format(ll=valid_event, type=valid_roc_auc, elapse=(time.time() - start) / 60))
        
        start = time.time()
        test_event, test_roc_auc = eval_epoch(model, test_data, opt)
        print('  - (test)    nll: {ll: 8.4f}, '
              ' roc auc :{type:8.4f},'
              'elapse: {elapse:3.3f} min'
              
              .format(ll=test_event, type=test_roc_auc, elapse=(time.time() - start) / 60))


        if (valid_roc_auc - best_auc_roc ) < 1e-5:
            impatient += 1
            if best_auc_roc < valid_roc_auc:
                best_auc_roc = valid_roc_auc
                best_model = deepcopy(model.state_dict())
  
        else:
            best_auc_roc = valid_roc_auc
            best_model = deepcopy(model.state_dict())
            impatient = 0
        
            
        if impatient >= 20:
            print(f'Breaking due to early stopping at epoch {epoch}')
            break

        scheduler.step()

    return best_model


def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=16)

    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_inner', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-ber_comps', type=int, default=32)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)

    parser.add_argument('-log', type=str, default='log.txt')

    opt = parser.parse_args()


    # default device is CUDA
    opt.device = torch.device('cpu')



    print('[Info] parameters: {}'.format(opt))
    
    np.random.seed(0)
    torch.manual_seed(0)

    """ prepare dataloader """
    trainloader, devloader, testloader, num_types = prepare_dataloader(opt)

    """ prepare model """
    model = Transformer(
        num_types=num_types,
        d_model=opt.d_model,
        d_inner=opt.d_inner,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        b_comps=opt.ber_comps,
        dropout=opt.dropout,
    )
    model.to(opt.device)

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.9)


    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    best_model = train(model, trainloader, devloader, testloader, optimizer, scheduler, opt)
    
    model.load_state_dict(best_model)
    model.eval()
    # save the model
    torch.save(model.state_dict(), "saved_models/defi_nobk")



import time
start = time.time()
np.random.seed(0)
torch.manual_seed(0)

if __name__ == '__main__':
    main()
end= time.time()
print("total training time is {}".format(end-start))

