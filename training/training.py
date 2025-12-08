import torch
from .models import *
import datetime
import random
import pickle
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from .contrastive_pretraining import gauss_normalize, prepare_ae_data
from tqdm import tqdm
import os
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.utils import *
from lion_pytorch import Lion
import numpy as np
from transformers import Adafactor

# Import dataset processing classes and functions
from dataset_creation.dataset_processing import (
    GenerateDataDict,
    QTrainingData,
    convert_reward_to_one_hot,
    convert_reward_to_one_hot_batch,
    _convert_reward_to_one_hot_batch,
    smooth_one_hot_reward,
    prepare_direct_pred_data
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def generate_EL2N_pruned_dataset(dataset_pth, proportion, model_pth, device='cuda',
                                  dtype=torch.float32, num_bins=320):
    model = torch.load(model_pth).eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    dataset = pic_load(dataset_pth)
    losses = [None for i in range(len(dataset))]

    # Calculate the losses for the dataset
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataset)):
            target = None
            data = data[0]
            data, summary, prices = data[0], data[1], data[2]
            data = set_nan_inf(data)
            
            # For light data augmentation
            #data = gaussian_noise(data, 1e-6).unsqueeze(0) # Unsqueeze to add batch dim
            #summary = gaussian_noise(summary, 1e-7).unsqueeze(0)

            pred = model(data.to(device, dtype=dtype), summary.to(device, dtype=dtype))
            buy_price = prices[0]
            for sell_price in prices[1:]:
                profit = [sell_price/buy_price]
                if target == None:
                    target = [convert_reward_to_one_hot_batch(item, num_bins) for item in profit]
                else:
                    _target = [convert_reward_to_one_hot_batch(item, num_bins) for item in profit]
                    _target = torch.stack(_target)
                    target = torch.cat((target, _target), dim=0)

            target = torch.permute(target, (1, 2, 0)) # b x 100 x 4
            ti = target.size(2)
            for i in range(ti):
                if i == 0:
                    loss = loss_fn(pred[:,:,i], target[:,:,i].to(device))
                else:
                    loss += loss_fn(pred[:,:,i], target[:,:,i].to(device))
            
            losses[i] = loss.item()
    
    # Initialize new list with sorted losses
    sorted_losses = sorted(losses)

    # Get number of elements to keep
    num_items_to_keep = int(len(dataset)*proportion)
    
    # Get the highest loss values corresponding to the pruned dataset
    losses_to_keep = sorted_losses[-num_items_to_keep:]

    # Get the indices of the pruned dataset
    indices_to_keep = [losses.index(item) for item in losses_to_keep]

    # Generate the list
    pruned_data = [dataset[item] for item in indices_to_keep]
    
    # Save the data
    save_pickle(pruned_data, f'{proportion}P_EL2N_{dataset_pth}')

def generate_outlier_dataset(dataset_pth, proportion):
    '''
    This function generates a dataset which contains the data points corresponding to the largest absolute
    changes in price.  This is doesn't take the model's prediction into account, only the data itself.
    '''
    print(f'Generating {proportion} outlier on {dataset_pth}')
    dataset = pic_load(dataset_pth)
    price_change = [None for i in range(len(dataset))]
    print('Num data points: ', len(price_change))
    for i, data in tqdm(enumerate(dataset)):
        profit = []
        data = data[0]
        _data = data[0]
        prices = data[1]
        buy_price = prices[0]
        for sell_price in prices[1:]:
            profit.append(sell_price.item()/buy_price.item())
        change = sum(item for item in profit)
        price_change[i] = change
    
    num_items_to_keep = int(len(dataset)*proportion)
    sorted_price_change = sorted(price_change)
    keep_price_changes = sorted_price_change[-num_items_to_keep:]
    ind_keep_changes = [price_change.index(item) for item in tqdm(keep_price_changes)]
    pruned_data = [dataset[item] for item in tqdm(ind_keep_changes)]
    print('Saving ', len(pruned_data), 'data points')
    print('Sample point: ', pruned_data[0])
    save_pickle(pruned_data, f'{proportion}P_Outlier_{dataset_pth}')

def generate_model_outlier_dataset(dataset_pth, proportion, model_pth, device='cuda', 
                                   dtype=torch.float32, num_bins=320, target_n=2):
    '''
    This function generates a dataset of points which the model predicts will have the largest absolute change. These are the
    points which it will eventually choose to buy, so it they are the one's most crucial for the model to predict accurately.
    '''

    model = torch.load(model_pth).eval()
    dataset = pic_load(dataset_pth)
    mean_predictions = [None for i in range(len(dataset))]

    # Calculate the outliers for the dataset
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataset)):
            _data = data[0]
            prices = data[1]
            pred, data = _data[0].unsqueeze(0), _data[1].unsqueeze(0)
            #pred, data = tup[0], tup[1]
            #print(data.shape, pred.shape)
            data = set_nan_inf(data)

            pred = model(data.to(device, dtype=dtype), pred.to(device, dtype=dtype)).squeeze(0)
            mean_pred = get_expected_price(pred[:,0], num_bins=num_bins, full_dist=False)
            mean_predictions[i] = abs(mean_pred)
    
    # Initialize new list with sorted predictions
    sorted_predictions = sorted(mean_predictions)

    # Get number of elements to keep
    num_items_to_keep = int(len(dataset)*proportion)
    
    # Get the highest absolute prediction values corresponding to the pruned dataset
    items_to_keep = sorted_predictions[-num_items_to_keep:]

    # Get the indices of the pruned dataset
    indices_to_keep = [mean_predictions.index(item) for item in tqdm(items_to_keep)]

    # Generate the list
    pruned_data = [dataset[item] for item in tqdm(indices_to_keep)]
    
    # Save the data
    save_pickle(pruned_data, f'{proportion}P_Out_{dataset_pth}')

def generate_naive_layer_dataset(dataset_pth, model_pth, device='cuda', 
                                 dtype=torch.float32, data_stack=True, num_chunks=1, half=0):
    '''
    Generates the prediction and transformer output of the model over a dataset, and stores it in a 
    new dataset for downstream greedy training.

    Args:
        dataset_pth (str): The path to the dataset, must be stored as a pickle file
        model_pth (str): A path to the PyTorch Model 
        data_stack (bool): If doing stack, the original data is stacked onthe transformer activation
                    in the embedding dimension, to provide as much information as possible to the next module.
        num_chunks (int): The number of seporate datasets to divide the new dataset into.
        half (0 or 1): Which half of the dataset to encode (This is necessary so the CPU doesn't run out 
                       of memory)
    '''

    model = torch.load(model_pth).eval()
    dataset = pic_load(dataset_pth)
    if half == 0:
        dataset = dataset[:int(len(dataset)/2)]
    else:
        dataset = dataset[int(len(dataset)/2):]
    new_dataset = []

    # Run the dataset through the model
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataset)):
            #target = None
            data = data[0]
            data, summary, prices = data[0].unsqueeze(0).to(device, dtype=dtype), data[1].to(device, dtype=dtype), data[2]
            data = set_nan_inf(data)

            pred = model(data, summary).cpu()
            t_activation = model.transformer(data, summary)
            
            if data_stack:
            # This is roughly equivelent to the first part of the model's forward method
                data = torch.flip(data,[1])
                data = data + model.pos_encode(data)
            
                # Reshape this to (batch, _, 52) so it can be appended to the end of the sequence
                summary = torch.reshape(summary, (1, 19, 52))
                
                # Add these data points to existing seqence
                data = torch.cat((data, summary), dim=1)

            t_activation = torch.cat((t_activation, data), dim=2).cpu()
            pred, t_activation = pred.squeeze(0), t_activation.squeeze(0)
            new_dataset.append(((pred, t_activation), prices))
    i = len(new_dataset)
    sub_dataset_size = int(i/num_chunks)
    for i in range(num_chunks):
        ind = i*sub_dataset_size
        sub_dataset = new_dataset[ind:ind+sub_dataset_size]
        save_pickle(sub_dataset, f'L1_H{half}_C{i}_{dataset_pth}')

def generate_naive_layer_2_dataset(dataset_pth, model_pth, device='cuda', 
                                 dtype=torch.float32, num_chunks=1, half=0):
    '''
    Generates the prediction and transformer output of the model over a dataset, and stores it in a 
    new dataset for downstream greedy training.

    Args:
        dataset_pth (str): The path to the dataset, must be stored as a pickle file
        model_pth (str): A path to the PyTorch Model 
        data_stack (bool): If doing stack, the original data is stacked onthe transformer activation
                    in the embedding dimension, to provide as much information as possible to the next module.
        num_chunks (int): The number of seporate datasets to divide the new dataset into.
        half (0 or 1): Which half of the dataset to encode (This is necessary so the CPU doesn't run out 
                       of memory)
    '''

    model = torch.load(model_pth).eval()
    dataset = pic_load(dataset_pth)
    if half == 0:
        dataset = dataset[:int(len(dataset)/2)]
    else:
        dataset = dataset[int(len(dataset)/2):]
    new_dataset = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataset)):
            _data = data[0]
            prices = data[1]
            pred, data = _data[0].unsqueeze(0), _data[1].unsqueeze(0)
            #print(data[:,0,0:51], data[:,0,52:])
            # t_act is 0:52, base data is 52:104
            base_data = data[:,:,:52]
            #print(base_data.shape, data.shape)
            data = set_nan_inf(data)
            pred = model(data.to(device, dtype=dtype), pred.to(device, dtype=dtype)).squeeze(0).cpu()
            t_activation = model.transformer(data.to(device, dtype=dtype)).cpu()
            t_activation = torch.cat((base_data, t_activation), dim=2).cpu()
            #print(pred.shape, t_activation.shape)
            new_dataset.append(((pred, t_activation), prices))
    i = len(new_dataset)
    sub_dataset_size = int(i/num_chunks)
    for i in range(num_chunks):
        ind = i*sub_dataset_size
        sub_dataset = new_dataset[ind:ind+sub_dataset_size]
        save_pickle(sub_dataset, f'L2_H{half}_C{i}_{dataset_pth}')

def convert_dataset_np(dataset_pth):
    data = pic_load(dataset_pth)
    data = np.array(data, dtype=object)
    save_pickle(data, f'np_{dataset_pth}')

def warmup_lambda(current_step):
    warmup_steps = 1000
    if current_step < warmup_steps:
        return current_step / warmup_steps
    return 1.0

def Train_Dist_Direct_Predictor(model, epochs, save_name, lr, num_bins, t_max, 
                        weight_decay, grd_nrm, misc, device='cuda', dtype=torch.float32, 
                        bounds=0.2,thr=1000,load_optim=True,grad_acc_steps=1, use_warmup=1,
                        train_prop=0.2):
    '''
    This is the primary training function, operating on the base layer. It takes in a model, list of dataset paths,
    various training parameters, and runs training on a direct distributional stock price prediction. 
    Args:
        model (nn.Module): the Pytorch Model to be trained
        epochs (int): Maximum epochs before terminating training
        save_name (str): Name for saving the model weights
        lr (float): learning rate
        num_bins (int): The number of bins the model uses to model its distribution
        t_max (int): The number of steps for one iteration of the Cosine Annealing learning rate scheduler
        weight_decay (float): Optimizer weight decay strength (L2 regularization)
        grd_nrm (float): Max gradient norm for clipping
        misc (iter): an iterable containing
                misc[0] -> batch size (int)
                misc[1] -> ndays (int): 
                                The number of market days in the future for the model to predict
                misc[2] -> data_pths (iter (containing strings)): 
                                An iterable containing the paths of the datasets to be trained on (stored as pickle files)

                misc[3] -> save_rate (int):
                                How often the model weights are saved (every _ epochs)
        device: The device to train on (i.e 'cuda:0')
        dtype: The dtype to cast the model weights and data
    '''
    batch_size, n_days, data_pths, save_rate = misc[0], misc[1], misc[2], misc[3]
    #optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    #optim = Adafactor(model.parameters(), lr, weight_decay=weight_decay, scale_parameter=False, relative_step=False)
    extra_params = list(model.pos_emb.parameters()) + list(model.linear_in.parameters())
    lr_configs = linear_layer_lr_scaling(model.layers, lr, lr*1.5, extra_params=extra_params)
    #optim = Lion(lr_configs,betas=(0.95, 0.98),weight_decay=weight_decay*10,use_triton=True)
    model.train()
    
    loss_fn = torch.nn.CrossEntropyLoss()

    if load_optim:
        optim.load_state_dict(torch.load('optim.pth'))

    scaler = torch.cuda.amp.GradScaler()
    #scaler.load_state_dict(torch.load('scaler.pth'))
    bin_edges = torch.load(f'Bin_Edges_{num_bins}')
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}:")
        it = 0
        for pth in data_pths:
            if len(data_pths) == 1 and epoch != 0:
                pbar = tqdm(dataloader)
                running_loss = 0
                steps = 0
                
            else:
                d = pic_load(pth)
                dataloader = DataLoader(d, batch_size, shuffle=True, pin_memory=1)
                pbar = tqdm(dataloader)
                running_loss = 0
                steps = 0
                bp_steps = 0
                ema_loss = 20
                it += 1
                print(f'\nUsing dataset: {pth} | ({it}/{len(data_pths)})\n')
            if epoch == 0 and use_warmup:
                scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lambda)
            #else:
            #    del scheduler
            for i, data in enumerate(dataloader):
                    target = None
                    data = data[0]
                    data, summary, prices = data[0], data[1], data[2]
                    #print(data.shape)
                    data = set_nan_inf(data)
                    summary = set_nan_inf(summary)
                    # For light data augmentation
                    data[:,:,15:] = gaussian_noise(data[:,:,15:], 7e-2)
                    #data[:,:,-180:] = gaussian_noise(data[:,:,-180:], 5e-2)
                    #summary = gaussian_noise(summary, 5e-5)
                    
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        #try:
                        pred = model(data.to(device, dtype=dtype), summary.to(device, dtype=dtype))
                        buy_price = prices[0]
                        for sell_price in prices[1:]:
                            profit = [sell_price/buy_price]
                            if target == None:
                                target = [_convert_reward_to_one_hot_batch(item, bin_edges) for item in profit]
                                #target = [smooth_one_hot_reward(item, num_bins, smoothing=0.2) for item in profit]
                                target = torch.stack(target)
                            else:
                                _target = [_convert_reward_to_one_hot_batch(item, bin_edges) for item in profit]
                                _target = torch.stack(_target)
                                target = torch.cat((target, _target), dim=0)
                        target = torch.permute(target, (1, 2, 0)) # b x 100 x 4
                        ti = target.size(2)
                        for i in range(ti):
                            if i == 0:
                                #if steps == 0:
                                    #print(torch.sum(pred[:,:,i]))
                                loss = loss_fn(pred[:,:,i], target[:,:,i].to(device))
                                #loss = compute_loss_with_gaussian_regularization(pred[:,:,i], target[:,:,i].to(device),loss_fn)
                            else:
                                loss += loss_fn(pred[:,:,i], target[:,:,i].to(device))
                                #loss += compute_loss_with_gaussian_regularization(pred[:,:,i], target[:,:,i].to(device),loss_fn)
                        t_loss = loss.item()
                        running_loss += t_loss
                        steps += 1
                        avg_loss = running_loss/steps

                        if (not torch.isnan(loss).any() and loss.item() > 0):
                            
                            #if t_loss >= running_loss/(steps+1):
                            #print(1)
                            scaler.scale(loss).backward()
                            bp_steps += 1
                            
                            if loss.item() < thr and bp_steps % grad_acc_steps == 0:
                                scaler.unscale_(optim)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), grd_nrm)
                                
                                # Step the optimizer only if gradients were computed
                                if any(param.grad is not None for param in model.parameters()):
                                    scaler.step(optim)
                                    scaler.update()
                                    optim.zero_grad()
                                    if epoch == 0 and use_warmup:
                                        scheduler.step()
                            #else:
                            #    optim.zero_grad()
                        
                        #except Exception as e:
                        #    print(e)
                    if steps % 1 == 0 and steps > 0:
                        a = f'Running Epoch Loss: {(running_loss/steps):.6f} Epoch Steps: {steps}/{len(dataloader)}  Step Loss: {t_loss:.6f}'
                        pbar.set_postfix_str(a)
                        pbar.update(1)
            torch.save(model, save_name+f"E{epoch+1}")
            torch.save(optim.state_dict(), 'optim.pth')
            torch.save(scaler, "scaler.pth")
            print(f'saved model ({steps})')

def Train_Dist_Direct_Predictor_SAM(model, epochs, save_name, lr, num_bins, t_max, 
                        weight_decay, grd_nrm, misc, device='cuda', dtype=torch.float32, 
                        bounds=0.2,thr=100000000,load_optim=True,grad_acc_steps=1, use_warmup=1,
                        train_prop=0.2):
    '''
    This is the primary training function, operating on the base layer. It takes in a model, list of dataset paths,
    various training parameters, and runs training on a direct distributional stock price prediction. 
    Args:
        model (nn.Module): the Pytorch Model to be trained
        epochs (int): Maximum epochs before terminating training
        save_name (str): Name for saving the model weights
        lr (float): learning rate
        num_bins (int): The number of bins the model uses to model its distribution
        t_max (int): The number of steps for one iteration of the Cosine Annealing learning rate scheduler
        weight_decay (float): Optimizer weight decay strength (L2 regularization)
        grd_nrm (float): Max gradient norm for clipping
        misc (iter): an iterable containing
                misc[0] -> batch size (int)
                misc[1] -> ndays (int): 
                                The number of market days in the future for the model to predict
                misc[2] -> data_pths (iter (containing strings)): 
                                An iterable containing the paths of the datasets to be trained on (stored as pickle files)

                misc[3] -> save_rate (int):
                                How often the model weights are saved (every _ epochs)
        device: The device to train on (i.e 'cuda:0')
        dtype: The dtype to cast the model weights and data
    '''
    batch_size, n_days, data_pths, save_rate = misc[0], misc[1], misc[2], misc[3]
    extra_params = list(model.pos_emb.parameters()) + list(model.linear_in.parameters())
    #lr_configs = linear_layer_lr_scaling(model.layers, lr, lr*1.5, extra_params=extra_params)
    #optim = Lion(lr_configs,betas=(0.95, 0.98),weight_decay=weight_decay*10,use_triton=True)
    optim = Lion
    optim = SAM(model.parameters(), optim, rho=0.05, betas=(0.95,0.98), weight_decay=weight_decay*10,use_triton=True)
    model.train()
    
    loss_fn = torch.nn.CrossEntropyLoss()

    if load_optim:
        optim.load_state_dict(torch.load('optim.pth'))

    scaler = torch.cuda.amp.GradScaler()
    #scaler.load_state_dict(torch.load('scaler.pth'))
    bin_edges = torch.load(f'Bin_Edges_{num_bins}')
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}:")
        it = 0
        for pth in data_pths:
            if len(data_pths) == 1 and epoch != 0:
                pbar = tqdm(dataloader)
                running_loss = 0
                steps = 0
                
            else:
                d = pic_load(pth)
                dataloader = DataLoader(d, batch_size, shuffle=True, pin_memory=1)
                pbar = tqdm(dataloader)
                running_loss = 0
                steps = 0
                bp_steps = 0
                ema_loss = 20
                it += 1
                print(f'\nUsing dataset: {pth} | ({it}/{len(data_pths)})\n')
            if epoch == 0 and use_warmup:
                scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lambda)
            #else:
            #    del scheduler
            batch_data = []
            for i, data in enumerate(dataloader):
                    target = None
                    data = data[0]
                    data, summary, prices = data[0], data[1], data[2]
                    #print(data.shape)
                    data = set_nan_inf(data)
                    summary = set_nan_inf(summary)
                    # For light data augmentation
                    data[:,:,15:] = gaussian_noise(data[:,:,15:], 7e-2)
                    #data[:,:,-180:] = gaussian_noise(data[:,:,-180:], 5e-2)
                    #summary = gaussian_noise(summary, 5e-5)
                    
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        #try:
                        pred = model(data.to(device, dtype=dtype), summary.to(device, dtype=dtype))
                        buy_price = prices[0]
                        for sell_price in prices[1:]:
                            profit = [sell_price/buy_price]
                            if target == None:
                                target = [_convert_reward_to_one_hot_batch(item, bin_edges) for item in profit]
                                #target = [smooth_one_hot_reward(item, num_bins, smoothing=0.2) for item in profit]
                                target = torch.stack(target)
                            else:
                                _target = [_convert_reward_to_one_hot_batch(item, bin_edges) for item in profit]
                                _target = torch.stack(_target)
                                target = torch.cat((target, _target), dim=0)
                        target = torch.permute(target, (1, 2, 0)) # b x 100 x 4
                        ti = target.size(2)
                        for i in range(ti):
                            if i == 0:
                                #if steps == 0:
                                    #print(torch.sum(pred[:,:,i]))
                                loss = loss_fn(pred[:,:,i], target[:,:,i].to(device))
                                
                                #loss = compute_loss_with_gaussian_regularization(pred[:,:,i], target[:,:,i].to(device),loss_fn)
                            else:
                                loss += loss_fn(pred[:,:,i], target[:,:,i].to(device))

                                #loss += compute_loss_with_gaussian_regularization(pred[:,:,i], target[:,:,i].to(device),loss_fn)
                    batch_data.append((data.detach(),summary.detach(),target.detach()))
                    t_loss = loss.item()
                    running_loss += t_loss
                    steps += 1
                    #print(torch.isnan(pred).any(), torch.isinf(pred).any(),pred.max(),pred.min())

                    if (not torch.isnan(loss).any() and loss.item() > 0):
                        
                        scaler.scale(loss).backward()
                        bp_steps += 1
                        
                        if loss.item() < thr and bp_steps % grad_acc_steps == 0:
                            scaler.unscale_(optim.base_optimizer)
                            #torch.nn.utils.clip_grad_norm_(model.parameters(), grd_nrm)
                            optim.first_step(zero_grad=True)
                            for acc_step in range(grad_acc_steps):  # Iterate through accumulated batches
                                data,summary,target = batch_data[acc_step]  # Re-fetch mini-batches
                                #print(1)
                                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                                    pred = model(data.to(device, dtype=dtype), summary.to(device, dtype=dtype))
                                    ti = target.size(2)
                                    for i in range(ti):
                                        if i == 0:
                                            loss = loss_fn(pred[:,:,i], target[:,:,i].to(device))
                                        else:
                                            loss += loss_fn(pred[:,:,i], target[:,:,i].to(device))
                                scaler.scale(loss).backward()
                            #scaler.unscale_(optim.base_optimizer)
                            #torch.nn.utils.clip_grad_norm_(model.parameters(), grd_nrm)
                            batch_data = []
                            # Step the optimizer only if gradients were computed
                            if any(param.grad is not None for param in model.parameters()):
                                #scaler.step(optim)
                                optim.second_step(zero_grad=True)
                                scaler.update()
                                optim.zero_grad()
                                if epoch == 0 and use_warmup:
                                    scheduler.step()
                        #else:
                        #    optim.zero_grad()
                    
                    #except Exception as e:
                    #    print(e)
                    if steps % 1 == 0 and steps > 0:
                        a = f'Running Epoch Loss: {(running_loss/steps):.6f} Epoch Steps: {steps}/{len(dataloader)}  Step Loss: {t_loss:.6f}'
                        pbar.set_postfix_str(a)
                        pbar.update(1)
            torch.save(model, save_name+f"E{epoch+1}")
            torch.save(optim.state_dict(), 'optim.pth')
            torch.save(scaler, "scaler.pth")
            print(f'saved model ({steps})')

def Train_Layer_Dist_Direct_Predictor(model, epochs, save_name, lr, num_bins, t_max, 
                        weight_decay, grd_nrm, misc, device='cuda', dtype=torch.float32):
    '''
    This is the primary training function, operating on the base layer. It takes in a model, list of dataset paths,
    various training parameters, and runs training on a direct distributional stock price prediction. 
    Args:
        model (nn.Module): the Pytorch Model to be trained
        epochs (int): Maximum epochs before terminating training
        save_name (str): Name for saving the model weights
        lr (float): learning rate
        num_bins (int): The number of bins the model uses to model its distribution
        t_max (int): The number of steps for one iteration of the Cosine Annealing learning rate scheduler
        weight_decay (float): Optimizer weight decay strength (L2 regularization)
        grd_nrm (float): Max gradient norm for clipping
        misc (iter): an iterable containing
                misc[0] -> batch size (int)
                misc[1] -> ndays (int): 
                                The number of market days in the future for the model to predict
                misc[2] -> data_pths (iter (containing strings)): 
                                An iterable containing the paths of the datasets to be trained on (stored as pickle files)
                misc[3] -> save_rate (int):
                                How often the model weights are saved (every _ epochs)
                misc[4] -> noise_level (float): 
                                The variance of the noise to add to the input data
                misc[5] -> full_stack (bool):
                                Whether or not to train the full network, or one layer (False is layerwise). In
                                practice, this corresponds to using the AdamW or Lion Optimizer.
        device: The device to train on (i.e 'cuda:0')
        dtype: The dtype to cast the model weights and data
    '''
    for param in model.parameters():
        param = param.to(dtype)
    batch_size, n_days, data_pths, save_rate, noise_level, full_stack = misc[0], misc[1], misc[2], misc[3], misc[4], misc[5]
    
    #optim = Adafactor(model.parameters(), lr, weight_decay=weight_decay, scale_parameter=False, relative_step=False)
    if full_stack:
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optim = Lion(model.parameters(), lr=lr/3, weight_decay=weight_decay*3)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=t_max, eta_min=lr/10)
    loss_fn = torch.nn.CrossEntropyLoss()

    #optim.load_state_dict(torch.load('optim.pth'))

    scaler = torch.cuda.amp.GradScaler()
    #scaler.load_state_dict(torch.load('scaler.pth'))
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}:")
        j = 0
        for pth in data_pths:
            j += 1
            if len(data_pths) == 1 and epoch != 0:
                pbar = tqdm(dataloader)
                running_loss = 0
                steps = 0
            else:
                try:
                    d = pic_load(pth)
                    dataloader = DataLoader(d, batch_size, shuffle=True, pin_memory=0)
                    pbar = tqdm(dataloader)
                    running_loss = 0
                    steps = 0
                except Exception as e:
                    print(e)
                print(f'\nUsing dataset: {pth} ({j}/{len(data_pths)})\n')
            for i, data in enumerate(dataloader):
                    target = None
                    _data = data[0]
                    prices = data[1]
                    pred, data = _data[0], _data[1]
                    #pred, data = tup[0], tup[1]
                    data = set_nan_inf(data.squeeze(1))
                    
                    # For data augmentation
                    data = gaussian_noise(data, noise_level)

                    with torch.autocast(device_type=device, dtype=torch.float32):
                        pred = model(data.to(device, dtype=dtype), pred.to(device, dtype=dtype))
                        buy_price = prices[0]
                        for sell_price in prices[1:]:
                            profit = [sell_price/buy_price]
                            if target == None:
                                target = [convert_reward_to_one_hot_batch(item, num_bins) for item in profit]
                                target = torch.stack(target)
                            else:
                                _target = [convert_reward_to_one_hot_batch(item, num_bins) for item in profit]
                                _target = torch.stack(_target)
                                target = torch.cat((target, _target), dim=0)
                        target = torch.permute(target, (1, 2, 0)) # b x 100 x 4
                        ti = target.size(2)
                        for i in range(ti):
                            if i == 0:
                                loss = loss_fn(pred[:,:,i], target[:,:,i].to(device))
                            else:
                                loss += loss_fn(pred[:,:,i], target[:,:,i].to(device))
                    
                    if not torch.isnan(loss).any() and loss.item() > 0:
                        t_loss = loss.item()
                        #loss.backward()

                        # For evening out the predictions
                        loss = (loss/avg_loss)**2
                        scaler.scale(loss).backward()
                        scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grd_nrm)
                        scaler.step(optim)
                        scaler.update()
                        #optim.step()
                        #scheduler.step()
                        optim.zero_grad()
                        running_loss += t_loss
                    steps += 1
                    avg_loss = running_loss/steps
                    if steps % 1 == 0 and steps > 0:
                        a = f'Running Epoch Loss: {(running_loss/steps):.6f} Epoch Steps: {steps}/{len(dataloader)}  Step Loss: {t_loss:.6f}'
                        pbar.set_postfix_str(a)
                        pbar.update(1)
            torch.save(model, save_name+f"E{epoch+1}")
            torch.save(optim.state_dict(), 'optim.pth')
            torch.save(scaler, "scaler.pth")
            print(f'saved model ({steps})')

def dataset_k_means(dataset, n_clusters=10, batch=0):
    print(type(dataset))
    #data = data[0]
    #data, summary, prices = data[0], data[1], data[2]
    #data = set_nan_inf(data)
    batch_len = int(len(dataset)/3)
    dataset = [torch.cat((set_nan_inf(item[0][0].flatten()), set_nan_inf(item[0][1].flatten()))) for i, item in tqdm(enumerate(dataset[batch_len*batch:(batch_len*batch+batch_len)]))]
    data = torch.stack(dataset, dim=0)
    print('Dataset array shape: ', data.shape)
    data = data.numpy()
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    #kmeans = KMeans(n_clusters=n_clusters,)
    #kmeans.fit(data)
    K = range(8, 20)
    fits = []
    score = []


    for k in tqdm(K):
        # train the model for current value of k on training data
        model = KMeans(n_clusters = k, random_state = 0, n_init='auto').fit(data)
        
        # append the model to fits
        fits.append(model)
        
        # Append the silhouette score to scores
        score.append(silhouette_score(data, model.labels_, metric='euclidean'))

    print(fits, score)
    ind = score.index(score.max())
    best_model = fits[ind]
    #best_model.labels_
    for i in range(len(dataset)):
        label = best_model.labels_[i]
        dataset[i] = label

def rescale_init(model, coeff):
    for name, param in model.named_parameters():
        param.data *= coeff

def save_torch(item, pth):
    torch.save(item, pth)

from torch.nn.utils import spectral_norm, remove_spectral_norm, weight_norm, remove_weight_norm

def apply_spectral_norm(module):
    """
    Recursively apply spectral normalization to all Linear and Conv layers in a module.
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.Linear, nn.Conv2d)):
            setattr(module, name, spectral_norm(child))  # Replace layer with spectral normalized version
        else:
            apply_spectral_norm(child)

def remove_spectral_norm(module):
    """
    Recursively remove spectral normalization to all Linear and Conv layers in a module.
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.Linear, nn.Conv2d)):
            remove_spectral_norm(child) # Replace layer with spectral normalized version
        else:
            remove_spectral_norm(child)

def apply_weight_norm_after_spectral_norm(module):
    """
    Remove spectral norm from all layers and apply weight norm instead.
    """
    for name, child in module.named_children():
        if isinstance(child, (torch.nn.Linear, torch.nn.Conv2d)):
            try:
                # Remove spectral norm if it was applied
                remove_spectral_norm(child)
            except AttributeError:
                pass  # Spectral norm not applied to this layer, skip

            # Apply weight norm
            setattr(module, name, weight_norm(child))
        else:
            apply_weight_norm_after_spectral_norm(child)

def apply_weight_norm(module):
    """
    Recursively apply weight normalization to all Linear and Conv layers in a module.
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.Linear, nn.Conv2d)):
            setattr(module, name, weight_norm(child))  # Replace layer with spectral normalized version
        else:
            apply_weight_norm(child)

def remove_weight_norm(module):
    """
    Recursively remove spectral normalization to all Linear and Conv layers in a module.
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.Linear, nn.Conv2d)):
            setattr(module, name, remove_weight_norm(child))  # Replace layer with spectral normalized version
        else:
            remove_spectral_norm(child)

def initialize_weights(m):
    init = nn.init.xavier_uniform_
    if isinstance(m, nn.Linear):
        init(m.weight) # Xavier initialization for linear layers
        if m.bias is not None:
            nn.init.zeros_(m.bias)         # Zero bias initialization
    elif isinstance(m, nn.MultiheadAttention):
        init(m.in_proj_weight)  # Initialize attention weights
        #m.in_proj_weight.data /= sqrt(243)
        nn.init.zeros_(m.in_proj_bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)           # Initialize LayerNorm weights to 1
        nn.init.zeros_(m.bias)  

def set_lr(optim, lr):
    for param_group in optim.param_groups:
        param_group['lr'] = lr

def linear_layer_lr_scaling(module_list, base_lr, max_lr, extra_params):
    """
    Create layer-wise learning rates that scale linearly across the layers in the ModuleList.
    
    Args:
        module_list (torch.nn.ModuleList): The layers of the network.
        base_lr (float): The learning rate for the first layer.
        max_lr (float): The learning rate for the last layer.
        
    Returns:
        list: A list of dictionaries with parameters and their corresponding learning rates.
    """
    num_layers = len(module_list)
    lr_configs = []

    for idx, layer in enumerate(module_list):
        # Linearly scale the learning rate for this layer
        layer_lr = base_lr + (max_lr - base_lr) * (idx / (num_layers - 1))
        
        # Add the layer parameters with the calculated learning rate to the list
        lr_configs.append({'params': layer.parameters(), 'lr': layer_lr})
    
    if extra_params is not None:
        lr_configs.append({
            'params': extra_params,
            'lr': base_lr,
        })

    return lr_configs
    
    return lr_configs

#@torch.compile
def generate_meta_model_dataset(dataset_pth, proportion, model_pth, device='cuda',
                                  dtype=torch.float32, num_bins=200):
    print('Generating Meta Model Dataset...')
    model = torch.load(model_pth).eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    dataset = pic_load(dataset_pth)
    losses = [None for i in range(len(dataset))]
    t_acts = [None for i in range(len(dataset))]
    preds = [None for i in range(len(dataset))]
    exp_price_diffs = [None for i in range(len(dataset))]
    new_dataset = []
    dataset = DataLoader(dataset)
    bin_edges = torch.load(f'Bin_Edges_{num_bins}')
    # Calculate the losses for the dataset
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataset)):
            target = None
            data = data[0]
            data, summary, prices = data[0], data[1], data[2]
            data = set_nan_inf(data)
            
            # For light data augmentation
            #data = gaussian_noise(data, 1e-3) # Unsqueeze to add batch dim
            #summary = gaussian_noise(summary, 1e-3)
            #print(data.shape,summary.shape)
            pred, t_act = model.forward_with_t_act(data.to(device, dtype=dtype), summary.to(device, dtype=dtype))
            preds[i] = pred
            t_acts[i] = t_act
            buy_price = prices[0]
            #print(prices)
            for sell_price in prices[1:]:
                profit = [sell_price/buy_price]
                if target == None:
                    target = [_convert_reward_to_one_hot_batch(item, bin_edges) for item in profit]
                    target = torch.stack(target)
                else:
                    _target = [_convert_reward_to_one_hot_batch(item, bin_edges) for item in profit]
                    _target = torch.stack(_target)
                    target = torch.cat((target, _target), dim=0)

            target = torch.permute(target, (1, 2, 0)) # b x 100 x 4
            ti = target.size(2)
            for j in range(ti):
                if j == 0:
                    loss = loss_fn(pred[:,:,j], target[:,:,j].to(device))
                else:
                    loss += loss_fn(pred[:,:,j], target[:,:,j].to(device))
            losses[i] = loss
            exp_price = get_expected_price(pred[:,:,0],bin_edges)
            target_price = prices[1]/buy_price
            exp_price_diffs[i] = exp_price.cpu()-target_price
            #if i > 100:
            #    break
    #print(losses[0])
    for i, data in tqdm(enumerate(dataset)):
        new_dataset.append((t_acts[i], preds[i], losses[i], exp_price_diffs[i]))

    save_pickle(new_dataset, f'MM_{dataset_pth}')

def train_meta_model(model, dataset_pths, batch_size, epochs, lr, wd):
    from utils import _get_expected_price
    optim = Lion(model.parameters(),lr=lr,weight_decay=wd)
    loss_fn = torch.nn.MSELoss()
    datasets = []
    bin_edges = torch.load('Bin_Edges_300')
    print('Loading Datasets...')
    for pth in tqdm(dataset_pths):
        datasets.append(pic_load(pth))
    for i in range(epochs):
        print(f'Epoch {i+1}')
        #datasets = []
        for dataset in datasets:
            #dataset = pic_load(pth)
            #print(dataset[100])
            #visualize_binned_distribution(dataset[0][1][0,:,2].cpu())
            dataloader = DataLoader(dataset,batch_size,shuffle=True)
            
            running_loss = 0
            for i, data in enumerate(dataloader):
                #data, label = data[0], data[1].to('cuda')
                t_act, pred, label, diff = data[0].to('cuda'),data[1].to('cuda'), data[2].to('cuda'), data[3].to('cuda')
                diff = torch.abs(diff).float()
                #price_pred = _get_expected_price(pred,bin_edges)
                mm_pred = model(t_act,pred)
                #label = torch.nn.functional.one_hot(label,2)
                loss = loss_fn(mm_pred, diff)
                running_loss += loss.item()
                loss.backward()
                optim.step()
                optim.zero_grad()
                if i%10 == 0:
                    print(f'Running Loss: {running_loss/(i+1):.4f}\r', end=' ')

    torch.save(model,'MetaModel')

def generate_meta_model_dataset_with_labels(dataset_pth, proportion, model_pth, device='cuda',
                                            dtype=torch.float32, num_bins=320):
    model = torch.load(model_pth).eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    dataset = pic_load(dataset_pth)
    losses = [None for _ in range(len(dataset))]
    t_acts = [None for _ in range(len(dataset))]
    preds = [None for _ in range(len(dataset))]
    #jac_norms = [None for _ in range(len(dataset))]

    # Calculate the losses for the dataset
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataset)):
            target = None
            data = data[0]
            data, summary, prices = data[0], data[1], data[2]
            data = set_nan_inf(data)

            pred, t_act = model.forward_with_t_act(data.to(device, dtype=dtype), summary.to(device, dtype=dtype))
            preds[i] = pred
            t_acts[i] = t_act
            #jac_norms[i] = get_jacobian_norm(model,data,summary)
            buy_price = prices[0]
            for sell_price in prices[1:]:
                profit = [sell_price / buy_price]
                if target is None:
                    target = [convert_reward_to_one_hot_batch(item, num_bins) for item in profit]
                else:
                    _target = [convert_reward_to_one_hot_batch(item, num_bins) for item in profit]
                    _target = torch.stack(_target)
                    target = torch.cat((target, _target), dim=0)

            target = torch.permute(target, (1, 2, 0))  # b x 100 x 4
            ti = target.size(2)
            for j in range(ti):
                if j == 0:
                    loss = loss_fn(pred[:, :, j], target[:, :, j].to(device))
                else:
                    loss += loss_fn(pred[:, :, j], target[:, :, j].to(device))

            losses[i] = loss.item()

    # Determine the threshold for the top proportion
    sorted_losses = sorted(losses)
    num_items_to_keep = int(len(dataset) * proportion)
    threshold = sorted_losses[num_items_to_keep - 1]  # Last loss in the top proportion

    # Generate new dataset with binary labels
    new_dataset = []
    for i, data in enumerate(dataset):
        is_top_proportion = losses[i] <= threshold  # Binary indicator
        # Add a new field/value to the dataset
        new_data = (t_act[i], pred[i], is_top_proportion)
        new_dataset.append(new_data)

    # Save the updated dataset
    save_pickle(new_dataset, f'{proportion}P_MM_{dataset_pth}')
    print(f"New dataset with binary labels saved to {proportion}P_Easy_WithLabels_{dataset_pth}")

def get_bin_width(dataset,num_bins):
    dataset = DataLoader(dataset)
    all_profits = []
    for i, item in tqdm(enumerate(dataset)):
        item = item[0]
        prices = item[2]
        buy_price = prices[0]
        for sell_price in prices[1:]:
            profit = sell_price/buy_price
            #if i == 0:
            #    print(profit)
            if profit < 0.1:
                profit = torch.tensor(0.5)
            if profit > 1.5:
                profit = torch.tensor(1.5)
            all_profits.append(profit.item())
        if i > 30000:
            break
    all_profits = np.array(all_profits)
    bin_edges = np.quantile(all_profits, q=np.linspace(0,1,num_bins+1))
    bin_edges = torch.tensor(bin_edges)
    print(bin_edges)
    save_torch(bin_edges,f'Bin_Edges_{num_bins}')

def main():
    seed = 155
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    seq_len = 600
    trade_period = 5
    k = 20
    c_prop = 0.25
    dataloader_pth_list = []
    #dataloader_pth = f'S_Price_Dataset_[a-n, seq-{seq_len}, n-{trade_period}, k%{k}, c_prop-{c_prop}].pickle'
    o_dataloader_pths = ['0.15P_Outlier_' + item for item in dataloader_pth_list]
    
    d_list = [item for item in os.listdir() if item.startswith('Dataset')]
    d_list = ['0.15P_Outlier_'+item for item in d_list]
    # If Training was interuppted, you can specify datasets that were already covered to ignore
    exclude_list = [1,2,3,5,7,8]
    d_list = [item for item in d_list if not int(item[-1]) in exclude_list]
    print(f'Effective Num Datasets: {len(d_list)}/14')

    dataloader_pth_list = d_list
    model_pth = 'DistPred_m_2_E6'
    #model_pth = 'full_model_so_E8_E3'
    data_dict_pth = 'DataDict_2024-06-26'
    summaries_pth = 'a-z_summary_embs.pickle'
    sectors_pth = 'a-z_sector_dict'
    prices_pth = 'unnorm_price_series_a-z_-10y-2024-06-26__.pickle'
    iter = 3
    generate_data = 0
    genreate_layer_data = 0
    np_dataset = 0
    train_outlier = 0
    use_warmup = 0
    tr_meta_model = 0

    if tr_meta_model:
        num_bins=300
        generate_mm_data = 0
        if generate_mm_data:
            for item in d_list:
                #if not item.endswith('+10'):
                #   print(item)
                generate_meta_model_dataset(item,0.1,model_pth,num_bins=num_bins)
        w = 1000
        d = 4
        epochs = 20
        bs = 64
        lr = 3e-6
        wd = 0.03
        model = meta_model(w,d).to('cuda')
        mm_data_pths = [item for item in os.listdir() if item.startswith('MM')]
        print(f'Num Meta Model Datasets: {len(mm_data_pths)}')
        train_meta_model(model,mm_data_pths,bs,epochs,lr,wd)
        return 0

    #gauss_normalize_dataset(d_list[0])
    #if gauss_normal:
    #    for item in d_list:
    #        _sgauss_normalize_dataset(item)

    if genreate_layer_data:
        for item in dataloader_pth_list:
            generate_naive_layer_2_dataset(item, model_pth, half=0)
            generate_naive_layer_2_dataset(item, model_pth, half=1)
            #generate_naive_layer_dataset(item, model_pth,half=1)
    if generate_data: 
        for i in range(15):
            dataloader_pth = f'Dataset_[a-z, seq-600, n-5, c_prop-{c_prop}]_{i+1+iter}_k{k}'
            dataloader = QTrainingData(summaries_pth, prices_pth, seq_len, trade_period,
                                    k=k, c_prop=c_prop, full=True, load=1, 
                                    sectors_pth=sectors_pth, data_slice_num=i+iter,
                                    data_dict_pth=data_dict_pth)
            save_pickle(dataloader.dataloader, dataloader_pth)
    if np_dataset:
        for item in dataloader_pth_list:
            #generate_model_outlier_dataset(item, proportion=0.05, model_pth=model_pth)
            generate_outlier_dataset(item, 0.15)
    if train_outlier:
        dataloader_pth_list = o_dataloader_pths
    gen_bin_indices = 0
    if gen_bin_indices:
        print('Loading Dataset for bin indices generation...')
        dataset = pic_load(d_list[0])
        num_bins=300
        get_bin_width(dataset,num_bins)
    
    batch_size = 1
    lr = 1e-6
    num_bins = 300
    #dim = 52
    dim = 218
    ff_dim = 300000
    n_head = 1
    layers = 16
    epochs = 409
    scale = 0.07
    num_lin_layers = 3
    dtype = torch.float32
    t_max = 1892
    w_decay = 0.3
    grd_nrm = 1.0
    noise_level = 5e-7
    full_stack = 0
    grad_acc_steps = int(64/batch_size)
    init_scale = 0.7
    dropout = 0.2
    train_prop = 0.2


    debug_cuda = False
    train = 1
    gen_model = 0
    train_lora = 0
    k_means = 0

    if k_means:
        dataset_k_means(pic_load(d_list[0]), batch=0)

    if debug_cuda:
        os.environ["NCCL_P2P_LEVEL"] =  "NVL"
        os.environ["NCCL_SHM_DISABLE"] = '1'
        os.environ['NCCL_DEBUG'] = 'INFO'
        os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
        #torch.cuda.empty_cache()
        print(f"CUDA_VISIBLE_DEVICES in Python: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(torch.cuda.device_count())
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['TORCH_USE_CUDA_DSA'] = '1'
        print("CUDA available:", torch.cuda.is_available())
        print("Current device:", torch.cuda.current_device())
        print("All available devices:", [torch.cuda.device(i) for i in range(torch.cuda.device_count())])
    
    if train:
        use_wandb = 0
        use_dist = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        if use_dist:    
            local_rank = int(os.environ["LOCAL_RANK"])
            dist.init_process_group(backend='nccl')
            torch.multiprocessing.set_sharing_strategy('file_system')
            dataset = pic_load(dataloader_pth)
            sampler = DistributedSampler(dataset)
            dataloader = DataLoader(dataset, batch_size, shuffle=False, sampler=sampler)
            model = torch.load(model_pth).to(f'cuda:{local_rank}')
            #model = Models.GL_DirectPredSumTran(seq_len=seq_len, data_dim=dim, nhead=n_head, ff=ff_dim, layers=layers, scale=1).to(f'cuda:{local_rank}',dtype=dtype)
            model = DDP(model)
            model = torch.compile(model) 
        generate_stack=0
        if generate_stack:
            transfer_model = torch.load(model_pth)
            get_model_parameter_count(transfer_model)
            model = t_Dist_Pred(seq_len,dim,num_bins,nhead=n_head,ff=ff_dim,layers=layers,sum_emb=832,scale=scale,dropout=dropout)
            get_model_parameter_count(model)
            for i in range(28):
                model.layers[i] = transfer_model.layers[i]
            model = replace_square_linear_layers(model)
            get_model_parameter_count(model)
            for i in range(14):
                for param in model.layers[i].parameters():
                    param.requires_grad = False
            

        else:
            # = [dataloader_pth, dataloader_pth2]
            #dataloader = pic_load(dataloader_pth)
            #data_len = len(dataloader)
            #dataloader = DataLoader(dataloader, batch_size, pin_memory=True, shuffle=True)
            if not gen_model:
                model = torch.load(model_pth)
                load_optim = 1
            if full_stack and gen_model:
                l_model_pth = 'L1_Dist600_E40'
                l2_model_pth = 'mL2_Dist600_E16.5_E59'
                b_model_pth = 'Dist600__E12'
                model_pth = 'full_model_so'
                #model = Full_L1_Dist_Pred(b_model_pth, l_model_pth, train=True)
                model = Composed_Dist_Pred(b_model_pth, l_model_pth, l2_model_pth, train=True)
                #model = Full_Dist_Pred('full_model_E8')
                load_optim = 0
            elif full_stack:
                model = torch.load(model_pth)
                a = 0
                if a:
                    # Only train the base layer
                    #for pram in model.layer.parameters():
                    #    pram.requires_grad = True
                    freeze = True
                    #for param in model.layer2.linear_out.parameters():
                    #    param.requires_grad = freeze
                    for param in model.layer2.linear_in.parameters():
                        param.requires_grad = freeze
                    for param in model.layer2.cls_head_in.parameters():
                        param.requires_grad = freeze
                    for param in model.layer.linear_out.parameters():
                        param.requires_grad = freeze
                    for param in model.layer.linear_in.parameters():
                        param.requires_grad = freeze
                    for param in model.layer.cls_head_in.parameters():
                        param.requires_grad = freeze
                    for param in model.base.linear_in.parameters():
                        param.requires_grad = freeze
            elif gen_model:
                #model = t_Dist_Pred(seq_len=seq_len, data_dim=dim, num_bins=num_bins, nhead=n_head, ff=ff_dim, layers=layers, sum_emb=832, scale=scale, dropout=dropout).to('cuda', dtype=dtype)
                model = t_universal_transformer_Dist_Pred(seq_len=seq_len, data_dim=dim, num_bins=num_bins, nhead=n_head, ff=ff_dim, layers=layers, sum_emb=832, scale=scale, dropout=dropout).to('cuda', dtype=dtype)
                #model.apply(initialize_weights)
                #rescale_init(model, init_scale)
                #model.linear_in = replace_square_linear_layers(model.linear_in,perms=3)
                #apply_weight_norm(model)
                model_pth = 'DistPred_UT_1'
                load_optim = 0
            train_upper = 0
            if train_upper:
                for param in model.layer.parameters():
                    param.requires_grad = False
                for param in model.base.parameters():
                    param.requires_grad = False
            #torch.cuda.empty_cache()
            #model = L2_Dist_Pred(seq_len=seq_len, data_dim=dim*3, num_bins=num_bins, nhead=n_head, ff=ff_dim, layers=layers, sum_emb=832, scale=scale).to('cuda', dtype=dtype)
            #model = DataParallel(model).to('cuda')
            model = model.to('cuda')
            model.train()
        if use_wandb:
            wandb.init(
            project = 'Dist Prediction',
            config={
                'model_name' : model_pth,
                'lr' : lr,
                'ff_dim' : ff_dim,
                'Batch size' : batch_size,
                'layers' : layers,
                'dtype' : dtype,
                't_max' : t_max,
                'w_decay' : w_decay,
                'grd_nrm' : grd_nrm,
                'trade period' : trade_period,
                'nhead' : n_head,
                'scale' : scale,
                'dataset' : dataloader_pth_list
            }
            )
        #for name, param in model.named_parameters():
        #    if name.startswith('layers.0'):
        #        param.requires_grad=False
        #    elif name.startswith('layers'):
        #        param.requires_grad=True
        world_size = 2
        torch.set_float32_matmul_precision('high')
        save_rate = 1
        misc = [batch_size, trade_period, dataloader_pth_list, save_rate, noise_level, full_stack]
        params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad==True)
        print('Network size: ', params, '\nTrainable: ', trainable_params)
        #save_name = f'Dist_pred_{params}'
        save_name = f'{model_pth}_'
        load_optim = 0
        #model = torch.compile(model)
        model = model.to('cuda')
        #model.train()
        torch.compile(Train_Dist_Direct_Predictor)
        
        
            #print(name)
        torch.cuda.empty_cache()
        #remove_spectral_norm(model)
        #apply_weight_norm_after_spectral_norm(model)
        Train_Dist_Direct_Predictor_SAM(model, epochs, save_name, lr, num_bins, t_max, w_decay, grd_nrm, misc=misc, 
                                    device=f'cuda', dtype=dtype, load_optim=load_optim, grad_acc_steps=grad_acc_steps,
                                    use_warmup=use_warmup, train_prop=train_prop)
        #Train_Layer_Dist_Direct_Predictor(model, epochs, save_name, lr, num_bins, t_max, w_decay, grd_nrm, misc=misc, device=f'cuda', dtype=dtype)
    else:
        pass
    
    
if __name__ == '__main__':
    main()