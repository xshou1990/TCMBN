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
from preprocess.Dataset import get_dataloader
from transformer.Models import Transformer
from tqdm import tqdm



def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)


    print('[Info] Loading dev data...')
    dev_data, num_types = load_data(opt.data + 'dev.pkl', 'dev')
    print('[Info] Loading test data...')
    test_data, _ = load_data(opt.data + 'test.pkl', 'test')

    devloader = get_dataloader(dev_data, opt.batch_size, shuffle=False)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
    
    
    return  devloader, testloader, num_types



def eval_epoch(model, validation_data, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()
    
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
         

             
    roc_auc = roc_auc_score(y_true=true_label, y_score=pred_label, average=None)
    
    hl = []
    tau_list = np.arange(100)/100
    for tau in list(tau_list):
        hl.append(f1_score(y_true=true_label, y_pred=np.array(pred_label)>tau, average='weighted'))
       # hl.append(hamming_loss(y_true=true_label, y_pred=np.array(pred_label)>tau))
    min_value = max(hl) 
    min_index = hl.index(min_value) 
    opt_tau = tau_list[min_index]

    print("dev: current roc_auc{}".format(roc_auc))
    
    print("dev: mean roc_auc{}".format(np.mean(roc_auc)))

    print("dev: weighted roc_auc{}".format(roc_auc_score(y_true=true_label, y_score=pred_label, average='weighted')))

    print('dev: Hamming loss: {}'.format(hamming_loss(y_true=true_label, y_pred = np.array(pred_label)>opt_tau))) 

    print('dev: F1 micro Measure: {}'.format(f1_score(y_true=true_label, y_pred=np.array(pred_label)>opt_tau, average='micro'))) 
    print('dev: F1 macro Measure: {}'.format(f1_score(y_true=true_label, y_pred=np.array(pred_label)>opt_tau, average='macro'))) 
    print('dev: F1 weighted Measure: {}'.format(f1_score(y_true=true_label, y_pred=np.array(pred_label)>opt_tau, average='weighted'))) 
    print('dev: F1 none Measure: {}'.format(f1_score(y_true=true_label, y_pred=np.array(pred_label)>opt_tau, average=None))) 

    
    return opt_tau

def test_epoch(model, validation_data, opt, opt_tau):
    """ Epoch operation in evaluation phase. """

    model.eval()
    
    pred_label = []
    true_label = []
#     total_time = 0
#     total_time_nll = 0

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
             
            
            
    roc_auc = roc_auc_score(y_true=true_label, y_score=pred_label, average=None)
    

    print("test: current roc_auc{}".format(roc_auc))

    print("test: mean roc_auc{}".format(np.mean(roc_auc)))
    
    print("test: weighted roc_auc{}".format(roc_auc_score(y_true=true_label, y_score=pred_label, average='weighted')))

    print('test: Hamming loss: {0}'.format(hamming_loss(y_true=true_label, y_pred=np.array(pred_label)>opt_tau))) 

    print('test: F1 micro Measure: {0}'.format(f1_score(y_true=true_label, y_pred=np.array(pred_label)>opt_tau, average='micro'))) 
    print('test: F1 macro Measure: {0}'.format(f1_score(y_true=true_label, y_pred=np.array(pred_label)>opt_tau, average='macro'))) 
    print('test: F1 weighted Measure: {}'.format(f1_score(y_true=true_label, y_pred=np.array(pred_label)>opt_tau, average='weighted'))) 

    


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

    
    np.random.seed(0)
    torch.manual_seed(0)

    """ prepare dataloader """
    devloader, testloader, num_types = prepare_dataloader(opt)

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

    """ evaluate on test set"""
    model.load_state_dict( torch.load("saved_models/defi_nobk") ) 
    model.eval()
    opt_tau = eval_epoch(model, devloader, opt)
    test_epoch(model, testloader, opt, opt_tau)
              


import time
start = time.time()
np.random.seed(0)
torch.manual_seed(0)

if __name__ == '__main__':
    main()
end= time.time()
print("total training time is {}".format(end-start))

