import torch 
from Models import PositionalEncoding, Dist_Pred
#import Stock
#from Stock import stock_info, all_stocks, test_stock_tickers
import datetime
from datetime import date
import random 
#import statistics
import numpy as np
import torch.nn as nn
import pickle 
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from contrastive_pretraining import gauss_normalize, prepare_ae_data
from tqdm import tqdm
import os
import keyboard
import time
import wandb
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel
from utils import *
from transformers import Adafactor
from lion_pytorch import Lion

class GenerateDataDict:
    def __init__(self, bulk_prices_pth, summaries_pth, seq_len, save=True, j=True):
        self.bulk_prices = pic_load(bulk_prices_pth)
        self.summaries = pic_load(summaries_pth)
        self.seq_len = seq_len
        self.companies = list(self.bulk_prices.keys())
        print(len(self.companies))
        dates = []
        max_ind = 0
        max_len = 0
        for i in range(20):
            if i == 0:
                max_len = len(list(self.bulk_prices[self.companies[i]].keys()))
            else:
                if len(list(self.bulk_prices[self.companies[i]].keys())) >= max_len:
                    max_ind = i
                    max_len = len(list(self.bulk_prices[self.companies[i]].keys()))
            #print(max_len)
        self.dates = list(self.bulk_prices[self.companies[max_ind]].keys())
        self.iterable_dates = self.dates[seq_len:]
        if j:
            self.r_data = self.prepare_price_only()
        else:
            self.r_data = self.encode_data()
        if save:
            with open(f'DataDict_{datetime.date.today()}', 'wb') as f:
                pickle.dump(self.r_data, f)

    
    def get_ae_prices_chunk(self, date, company):
        ind = self.dates.index(date)
        dates = self.dates[ind-self.seq_len:ind]
        data = []
        for date in dates:
            try:
                data.append(self.bulk_prices[company][date])
            except Exception as e:
                return None
        a = torch.stack(data, dim=0).unsqueeze(0)
        return a

    def get_nor_seq_price_data(self, date, company):
        ind = self.dates.index(date)
        dates = self.dates[ind-self.seq_len:ind]    
        data = []
        for date in dates:
            data.append(self.bulk_prices[company][date])
        data = torch.stack(data, dim=1)
        return data 
    
    def encode_data(self):
        self.data = {}
        counter = 0
        top = len(self.iterable_dates)*len(self.companies)
        counter2 = 0
        with torch.no_grad():
            for date in self.iterable_dates:
                self.data[date] = {}
                for company in self.companies:
                    #try:
                    p = self.get_ae_prices_chunk(date, company)
                    try:
                        s = self.summaries[company]
                        a = self.bulk_prices[company][date].unsqueeze(0)
                    except Exception as e:
                        p = None
                    if p is None:
                        counter2+=1
                        pass
                    else:
                        s, p = prepare_ae_data([s, p])
                        if torch.isnan(p).any() or torch.isnan(s).any():
                            print(1)
                        if torch.isinf(p).any() or torch.isinf(s).any():
                            print(1)
                        d1 = self.encoder.encode(p, s) # Tensor of size 913
                        if counter %5000 == 0:
                            print(d1[0][0:10])
                        #d2 = self.get_nor_seq_price_data(date, company)
                        d1 = torch.cat((d1.cpu(), a), dim=1)
                        self.data[date][company] = d1
                        counter += 1
                        print(f'\r{counter}, {counter2} / {top}', end='')
        return self.data

    def prepare_price_only(self):
        self.data = {}
        counter = 0
        top = len(self.iterable_dates)*len(self.companies)
        counter2 = 0
        for date in self.iterable_dates:
            self.data[date] = {}
            for company in self.companies:
                #try:
                #p = self.get_ae_prices_chunk(date, company)
                try:
                    a = self.bulk_prices[company][date].unsqueeze(0)
                except Exception as e:
                    a = None
                if a is None:
                    counter2+=1
                    pass
                else:
                    '''
                    # S is dim 71, repeat it n times to match seq_len
                    n = int(self.seq_len/71)
                    s = s.repeat(n, 1).unsqueeze(1)
                    #print(s.shape)
                    d1 = torch.cat((s, p), dim=1)
                    #print(d1.shape)
                    if counter %5000 == 0:
                        print(d1[0][0:10])
                    #d2 = self.get_nor_seq_price_data(date, company)
                    '''
                    #s, p = prepare_ae_data([s, p])
                    self.data[date][company] = a
                    counter += 1
                    print(f'\r{counter}, {counter2} / {top}', end='')
        return self.data

class QTrainingData(Dataset): 
    def __init__(self, summaries_pth, bulk_prices_pth, seq_len, n=5, k=1, c_prop=0.5, full=True, layer_args=None, load=True, inference=False, sectors_pth=str, data_slice_num=int):
        self.data_slice_num = data_slice_num
        if layer_args is not None:
            mdl_pth, layer_seq_len = layer_args[0], layer_args[1]
            self.seq_len = layer_seq_len
            self.layer = True
        else:
            self.seq_len = seq_len
        self.n = n # Trade period len
        self.k = k # How often to take a point from the seqence
        self.summaries = pic_load(summaries_pth)
        self.sectors_info_dict = pic_load(sectors_pth)
        if load:
            self.data = pic_load('DataDict_inf_10y')
        else:
            self.data = GenerateDataDict(bulk_prices_pth, summaries_pth, seq_len, save=True)
            self.data = self.data.r_data
            save_pickle(self.data, 'DataDict_inf_10y')

        self.dates = list(self.data.keys())
        self.iterable_dates = self.dates[self.seq_len:]
        print('iterable dates: ', len(self.iterable_dates))
        self.companies = list(self.data[self.dates[0]].keys())
        self.date_chunks = {}
        for i in range(len(self.dates)-self.seq_len):
            self.date_chunks[self.dates[i+self.seq_len]] = self.dates[i:i+self.seq_len]
        self.dataloader = []
        self.relative_price_dict = {}
        self.generate_full_market_stats()
        self.get_sector_industry_classes()
        if full and (layer_args == None):
            self.prepare_dataset(c_prop)
        if full and (layer_args is not None):
            self.model = torch.load(mdl_pth).eval().to('cuda')
            self.prepare_dataset_layer(c_prop)
        if inference is not False:
            self.inference_data = {}
            start, end = inference[0], inference[1]
            self.inference_dataset(start, end)
            #save_pickle(self.inference_data, 'InferenceDataset')

    def generate_full_market_stats(self, dim=5):
        self.pca_stats = {}
        print('Generating Market Stats')
        for date in tqdm(self.dates):
            pca = self.get_day_market_stats(date, dim)

            self.pca_stats[date] = pca
        
    def get_price_volume_bounds(self, date):
        price_volumes = []
        for company in self.companies:
            try:
                day_data = self.data[date][company]
                price_volumes.append(day_data[0][-2]*day_data[0][-1])
            except Exception as e:
                pass
        #print(price_volumes[0].shape)
        t_price_volumes = torch.stack(price_volumes, 0)
        min = torch.min(t_price_volumes)
        max = torch.max(t_price_volumes)
        mean = torch.mean(t_price_volumes)
        std = torch.std(t_price_volumes)
        #print(mean, std)
        return min, max, mean, std
    
    def get_company_chunk_data(self, company, date):
        dates = self.date_chunks[date] 
        try:
            comp_data = self.data[dates[0]][company].unsqueeze(0)
            pca_stats = self.pca_stats[dates[0]].unsqueeze(0)
            rel_data = self.relative_price_dict[dates[0]][company]
            #print(comp_data)
            for _date in dates[1:]:
                rel_data = torch.cat((rel_data, self.relative_price_dict[_date][company]), dim=0)
                comp_data = torch.cat((comp_data, self.data[_date][company].unsqueeze(0)), dim=0)
                pca_stats = torch.cat((pca_stats, self.pca_stats[_date].unsqueeze(0)), dim=0)
            comp_data = comp_data.unsqueeze(0)
            c = self.summaries[company]
            price = self.data[dates[-1]][company][0][-2] # For computing target values
            #pca_stats = torch.stack(self.pca_stats[dates], dim=0)
            #pca_stats.unsqueeze(0)
            #print(comp_data.shape, c.shape, pca_stats.shape)
        except Exception as e:
            #print(e)
            return None
        return (comp_data, c, pca_stats, price, rel_data)

    def get_n_company_chunks(self, company, date):
        data_chunks = []
        ind = self.dates.index(date)
        for i in range(self.n):
            a = self.get_company_chunk_data(company, date)
            #print('n_chk', type(a))
            if a is not None:
                #print('yes', a)
                data_chunks.append(a)
            else:
                return None
            try:
                date = self.dates[ind+i]
            except Exception as e:
                break
        return data_chunks
    
    def get_n_company_prices(self, company, date):
        try:
            data = self.get_company_chunk_data(company, date)
            prices = []
            ind = self.dates.index(date)
            for i in range(self.n):
                date = self.dates[ind+i]
                prices.append(self.data[date][company][0][-2])
        except Exception as e:
            return None
        if data is None:
            return None
        else:
            return (data, prices)        

    def get_abs_price_dim(self, price_seq, s):
        # This represents the price volume score relative to the other companies on the market day
        # Shape (batch, seq, dim)
        # 2.465073550667833 16111693133.838446 99981918.3880809 557781883.0235642
        _min, _max, mean, std = s[0], s[1], s[2], s[3]
        close = price_seq[:, :, -2]
        volume = price_seq[:, :, -1]
        price_volume = close*volume
        gauss_price_volume = (price_volume-mean)/std
        cube_price_volume = 2*(price_volume-_min)/(_max-_min)-1
        added_data = torch.cat((gauss_price_volume.unsqueeze(2), cube_price_volume.unsqueeze(2)), dim=2).to(torch.float32)
        return added_data
    
    def prepare_seq_data(self, data, s):
        data = data.squeeze(2)
        _data = set_nan_inf(data)
        _data = gauss_normalize(_data, dim=1)
        __data = cube_normalize(_data, dim=1)
        a_data = self.get_abs_price_dim(data, s)
        _data = torch.cat((_data, __data, a_data), dim=2)
        return _data

    def get_day_market_stats(self, date, pca_dim=5):
        stats = []
        ind = self.dates.index(date)
        if ind == 0:
            return None
        prev_date = self.dates[ind-1]
        self.relative_price_dict[date] = {}
        #sectors_change_dict = {}
        for company in self.companies:
            try:
                day_data = self.data[date][company]
                prev_day_data = self.data[prev_date][company]
                change = (day_data-prev_day_data)/prev_day_data
            except Exception:
                change = torch.zeros(5).unsqueeze(0)
            stats.append(change)
            self.relative_price_dict[date][company] = change
        stats = torch.stack(stats, 0)
        stats = stats.squeeze(1)
        stats = set_nan_inf(stats)
        u, s, v = torch.pca_lowrank(stats, q=pca_dim) # 3000 x dim, dim, dimx5
        mean = torch.mean(stats,  0) # 1 x 5
        std = torch.std(stats, 0) # 2 x 5
        r_data = torch.cat((v.flatten(), mean.flatten(), std.flatten()), dim=0) # 35 dimensional
        return r_data

    def get_sector_industry_classes(self):
        self.sectors = []
        self.industries = []
        for company, data in self.sectors_info_dict.items():
            sector, industry = data[0], data[1]
            if not (sector in self.sectors):
                self.sectors.append(sector)
            if not (industry in self.industries):
                self.industries.append(industry)
        self.num_sectors = len(self.sectors)
        self.num_industries = len(self.industries)
        print('Num sectors: ', self.num_sectors, 'Num Industries ', self.num_industries)
        self.sectors_sinusoidal_encoding = PositionalEncoding(d_model=64+52, max_len=self.num_sectors).encoding
        self.industry_sinusoidal_encoding = PositionalEncoding(d_model=52+52, max_len=self.num_industries).encoding
    
    def add_sector_industry_embedding(self, summary, company):
        sector, industry = self.sectors_info_dict[company][0], self.sectors_info_dict[company][1]
        sector_ind, industry_ind = self.sectors.index(sector), self.industries.index(industry)
        sector_emb, industry_emb = self.sectors_sinusoidal_encoding[sector_ind, :], self.industry_sinusoidal_encoding[industry_ind, :]
        final_emb = torch.cat((summary, sector_emb, industry_emb), dim=0).detach().cpu()
        return final_emb

    def prepare_dataset(self, c_prop):
        counter = 0
        c_prop = int(c_prop*len(self.companies))
        c_slice_ind = self.data_slice_num
        start = 0 + c_slice_ind*c_prop
        print('Generating Dataset')
        for date in tqdm(self.iterable_dates):
            counter += 1
            s = self.get_price_volume_bounds(date)
            stats = self.pca_stats[date]
            companies = self.companies[start:start+c_prop]
            #companies = random.sample(self.companies, c_prop)
            #print('len companies', len(companies))
            for company in companies:
                if (counter-2) % self.k == 0: 
                    print(f'\r{counter}/{int(len(self.iterable_dates)/self.k)}', end='')
                    all_data = self.get_n_company_prices(company, date)
                    r_data = []
                    if all_data is None:
                        #print(counter)
                        pass
                    else:
                        data, prices = all_data[0], all_data[1]
                        prices = [torch.tensor(price).cpu() for price in prices]
                        _data = data[0]
                        summary = data[1].squeeze(0).cpu()
                        summary = self.add_sector_industry_embedding(summary, company)
                        stats = data[2]
                        price = data[3]
                        rel_data = data[4]
                        _data = self.prepare_seq_data(_data, s).squeeze(0) # Shape 1 x 300 x 12
                        _data = torch.cat((_data.cpu(), stats.cpu(), rel_data.cpu()), dim=1).to(device='cpu', dtype=torch.float32)
                        r_data.append([_data, summary, prices])
                        if len(data) == self.n:  
                            self.dataloader.append(r_data)

    def prepare_dataset_layer(self, c_prop):
        with torch.no_grad():
            counter = 0
            c_prop = int(c_prop*len(self.companies))
            for date in tqdm(self.iterable_dates):
                counter += 1
                s = self.get_price_volume_bounds(date)
                stats = self.pca_stats[date]
                companies = random.sample(self.companies, c_prop)
                for company in companies:
                    if (counter-2) % self.k == 0: 
                        print(f'\r{counter}/{int(len(self.iterable_dates)/self.k)}', end='')
                        data = self.get_n_company_chunks(company, date)
                        r_data = []
                        if data is None:
                            #print(counter)
                            pass
                        else:
                            for list_data in data:
                                #print('Tup shape:', _data[0].shape, _data[1].shape, _data[2].shape)
                                _data = list_data[0]
                                summary = list_data[1].cpu()
                                stats = list_data[2]
                                price = list_data[3]
                                rel_data = list_data[4]
                                _data = self.prepare_seq_data(_data, s).squeeze(0) # Shape 1 x 300 x 12
                                #print('1x350x12, 350x35?',_data.shape, stats.shape)
                                #print(_data.shape, stats.shape, rel_data.shape)
                                _data = torch.cat((_data.cpu(), stats.cpu(), rel_data.cpu()), dim=1).to(device='cpu', dtype=torch.float32).detach()
                                #print(_data.shape)
                                #print('1x350x47?:', _data.shape)
                                #print(_data.shape, summary.shape, price.shape)
                                price = torch.tensor(price).cpu().detach()
                                encoder_data = _data[0:200, :].to('cuda').unsqueeze(0)
                                extended_data = _data[200:, :]
                                #print(encoder_data.shape, extended_data.shape)
                                tran_act, out_act = self.model.encode(encoder_data, summary.to('cuda'))
                                tran_act = tran_act.cpu().squeeze(0)
                                extended_data = torch.cat((extended_data, tran_act), dim=0).detach()

                                r_data.append([extended_data, out_act.cpu(), summary, price])
                                #print(_data.shape, summary.shape, price.shape)
                            if len(data) == self.n:  
                                self.dataloader.append(r_data)

    def inference_dataset(self, start_date, end_date):
        counter = 0
        #print(self.iterable_dates[0:3], self.iterable_dates[-3:], len(self.iterable_dates))
        s_ind = self.iterable_dates.index(start_date)
        e_ind = self.iterable_dates.index(end_date)
        dates = self.iterable_dates[s_ind:e_ind]
        for date in tqdm(dates):
            counter += 1
            s = self.get_price_volume_bounds(date)
            stats = self.pca_stats[date]
            self.inference_data[date] = {}
            for company in self.companies:
                #print(f'\r{counter}/{int(len(self.iterable_dates)/self.k)}', end='')
                all_data = self.get_n_company_prices(company, date)
                if all_data is None:
                    #print(counter)
                    pass
                else:
                    data, prices = all_data[0], all_data[1]
                    prices = [torch.tensor(price).cpu() for price in prices]
                    _data = data[0]
                    summary = data[1].squeeze(0).cpu()
                    stats = data[2]
                    price = data[3]
                    rel_data = data[4]
                    _data = self.prepare_seq_data(_data, s).squeeze(0) # Shape 1 x 300 x 12
                    _data = torch.cat((_data.cpu(), stats.cpu(), rel_data.cpu()), dim=1).to(device='cpu', dtype=torch.float32)
                    self.inference_data[date][company] = [_data, summary, price]

    def __len__(self):
        return len(self.dataloader)
    
    def __getitem__(self, ind):
        return self.dataloader[ind]

def convert_reward_to_one_hot(value, num_bins, bounds=0.2, correct=True):
    if correct:
        value = value-1
    scaled_value = (value + bounds) / (bounds*2) * (num_bins - 1)
    bin_index = int(scaled_value)   
    if bin_index > num_bins-1:
        bin_index = num_bins-1
    if bin_index < 0:
        bin_index = 0
    one_hot = torch.zeros(num_bins)
    one_hot[bin_index] = 1.0
    return one_hot

def convert_reward_to_one_hot_batch(values, num_bins, bounds=0.2, correct=True):
    # Adjust values if correct flag is set
    if correct:
        values = values - 1
    
    # Scale values to bin indices
    scaled_values = (values + bounds) / (bounds * 2) * (num_bins - 1)
    bin_indices = scaled_values.long()
    
    # Clip bin indices to be within [0, num_bins-1]
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)
    
    # Create one-hot encoding
    one_hot = torch.zeros(values.size(0), num_bins, device=values.device)
    one_hot.scatter_(1, bin_indices.unsqueeze(1), 1.0)
    
    return one_hot

def prepare_direct_pred_data(data, device='cuda', dtype=torch.float32):
    data = data.squeeze(2)
    _data = set_nan_inf(data).to(device)
    _data = gauss_normalize(_data, dim=1)
    __data = cube_normalize(_data, dim=1)
    a_data = get_abs_price_dim(data).to(device)
    _data = torch.cat((_data, __data, a_data), dim=2).to(device, dtype=dtype)
    return _data

def Train_Dist_Direct_Predictor(rank, model, epochs, save_name, lr, num_bins, t_max, weight_decay, grd_nrm, misc, epsilon=0.5, device='cuda', dtype=torch.float16):
    for param in model.parameters():
        param = param.to(dtype)
    #print(dataloader, len(dataloader))
    batch_size, n_days, data_pths, save_rate = misc[0], misc[1], misc[2], misc[3]
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    #optim = Adafactor(model.parameters(), lr, weight_decay=weight_decay, scale_parameter=False, relative_step=False)
    #optim = Lion(model.parameters(), lr=lr/3, weight_decay=weight_decay*3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=t_max, eta_min=lr/10)
    loss_fn = torch.nn.CrossEntropyLoss()
    load = 0
    if load:
        optim.load_state_dict(torch.load('optim.pth'))
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}:")

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
                print(f'\nUsing dataset: {pth}\n')
            for i, data in enumerate(dataloader):
                    target = None
                    data = data[0]
                    data, summary, prices = data[0], data[1], data[2]
                    data = set_nan_inf(data)
                    
                    # For light data augmentation
                    data = gaussian_noise(data, 1e-8)
                    #summary = gaussian_noise(summary, 1e-7)

                    pred = model(data.to(device, dtype=dtype), summary.to(device, dtype=dtype))
                    buy_price = prices[0]
                    for sell_price in prices[1:]:
                        #profit = [sell_price.item()/buy_price.item()]
                        profit = [sell_price/buy_price]
                        if target == None:
                            target = [convert_reward_to_one_hot_batch(item, num_bins) for item in profit]
                            target = torch.stack(target)
                            #target = torch.stack(target).unsqueeze(0)
                        else:
                            _target = [convert_reward_to_one_hot_batch(item, num_bins) for item in profit]
                            _target = torch.stack(_target)
                            #_target = torch.stack(_target).unsqueeze(0)
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
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grd_nrm)
                        optim.step()
                        scheduler.step()
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
            print(f'saved model ({steps})')
        #if rank == 0:
        #    wandb.log({'GL Loss':(running_loss/steps)})
        #    if (epochs+1)%save_rate == 0:
        #        torch.save(model, save_name+f"E{epoch+1}")
        #        torch.save(optim.state_dict(), 'optim.pth')
        #    print(f'saved model ({steps})')

def Train_Layer_Dist_Direct_Predictor(rank, model, epochs, save_name, lr, num_bins, t_max, weight_decay, grd_nrm, misc, epsilon=0.5, device='cuda', dtype=torch.float16):
    for param in model.parameters():
        param = param.to(dtype)
    #print(dataloader, len(dataloader))
    batch_size, n_days, data_pths, save_rate = misc[0], misc[1], misc[2], misc[3]
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=t_max, eta_min=lr/100)
    loss_fn = torch.nn.CrossEntropyLoss()
    load = 1
    if load:
        optim.load_state_dict(torch.load('optim.pth'))
    #nn.Softmax()
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}:")
        for pth in data_pths:
            running_loss = 0
            steps = 0
            d = pic_load(pth)
            dataloader = DataLoader(d, batch_size, shuffle=True, pin_memory=1)
            pbar = tqdm(dataloader)
            print(f'\nUsing dataset: {pth}\n')
            for i, n_step_data_list in enumerate(dataloader):
                if i > 0:
                    counter = 0
                    target = None
                    train = True 
                    for data in n_step_data_list:
                        # [extended_data, activation, summary, price]
                        data, out_act, summary, price = data[0].to(device, dtype=dtype), data[1].to(device, dtype=dtype), data[2].to(device, dtype=dtype), data[3]
                        data = set_nan_inf(data)
                        if counter == 0:
                            pred = model(data, summary, out_act)
                            buy_price = price.to(device)
                        else:
                            sell_price = price.to(device)
                            profit = [sell_price.item()/buy_price.item()]
                            if target == None:
                                target = [convert_reward_to_one_hot(item, num_bins) for item in profit]
                                target = torch.stack(target).unsqueeze(0)
                            else:
                                _target = [convert_reward_to_one_hot(item, num_bins) for item in profit]
                                _target = torch.stack(_target).unsqueeze(0)
                                target = torch.cat((target, _target), dim=0)
                        counter += 1
                        if torch.isinf(pred).any():
                            #print(counter)
                            train = False
                            break
                            inf_mask = torch.isinf(pred)
                            pred[inf_mask] = 0
                        if torch.isnan(pred).any():
                            #print(counter)
                            train = False
                            break
                            nan_mask = torch.isnan(pred)
                            pred[nan_mask] = 0
                    if train:
                        target = torch.permute(target, (1, 2, 0)) # b x 100 x 4
                        ti = target.size(2)
                        for i in range(ti):
                            if i == 0:
                                loss = loss_fn(pred[:,:,i], target[:,:,i].to(device))
                            else:
                                loss += loss_fn(pred[:,:,i], target[:,:,i].to(device))
                        if not loss.item() > 0:
                            print(pred)
                            print(target)
                            print(loss)
                            return
                        #if loss.item()
                        if not torch.isnan(loss).any() and loss.item() > 0:
                            t_loss = loss.item()
                            #scale = (loss.item()/11)**2 # Square scale the loss to optimize for outliers
                            #loss = loss * scale
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grd_nrm)
                            optim.step()
                            scheduler.step()
                            optim.zero_grad()
                            running_loss += t_loss
                            steps += 1
                    if steps % 1 == 0 and steps > 0:
                        a = f'Running Epoch Loss: {(running_loss/steps):.6f} Epoch Steps: {steps}/{len(dataloader)}  Step Loss: {t_loss:.6f}'
                        pbar.set_postfix_str(a)
                        pbar.update(1)
                        #print(a, end='')
                    '''
                    if steps % 1000 == 0:
                        v_model.eval()
                        gl_model.eval()
                        with torch.no_grad():
                            for n_step_data_list in valid_data:
                                for data in n_step_data_list:
                                    try:
                                        _data = set_nan_inf(data)
                                        _data = _data.squeeze(2).to(device)
                                        _data = gauss_normalize(_data, dim=1)
                                        __data = cube_normalize(_data)
                                        _data = torch.cat((_data, __data), dim=2).to(device, dtype=dtype)
                                        if torch.isinf(_data).any():
                                            inf_mask = torch.isinf(_data)
                                            _data[inf_mask] = 0
                                            #print(_data)
                                        if torch.isnan(_data).any():
                                            nan_mask = torch.isnan(_data)
                                            _data[nan_mask] = 0
                                            #print(_data)
                                        pred_v = v_model(_data)
                                        print(pred)
                                    except Exception as e:
                                        print(e, type(data), data)'''
        if rank == 0:
            wandb.log({'GL Loss':(running_loss/steps)})
            if (epochs+1)%save_rate == 0:
                torch.save(model, save_name+f"E{epoch+1}")
                torch.save(optim.state_dict(), 'optim.pth')
            print(f'saved model ({steps})')

    
def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] =  "expandable_segments:True"
    seq_len = 600
    trade_period = 5
    k = 1
    c_prop = 0.0666
    dataloader_pth_list = []
    dataloader_pth = f'S_Price_Dataset_[a-n, seq-{seq_len}, n-{trade_period}, k%{k}, c_prop-{c_prop}].pickle'
    dataloader_pth_list.append('Dataset_[a-z, seq-600, n-5, c_prop-0.0666]_1')
    dataloader_pth_list.append('Dataset_[a-z, seq-600, n-5, c_prop-0.0666]_2')
    dataloader_pth_list.append('Dataset_[a-z, seq-600, n-5, c_prop-0.0666]_3')
    dataloader_pth_list.append('Dataset_[a-z, seq-600, n-5, c_prop-0.0666]_4')
    dataloader_pth_list.append('Dataset_[a-z, seq-600, n-5, c_prop-0.0666]_5')
    #dataloader_pth_list.append(dataloader_pth)
    model_pth = 'Dist_600_warm'
    #model_pth = 'Dist600_warm141'
    #model_pth = 'Dist_pred_warm3.2m'
    summaries_pth = 'a-z_summary_embs.pickle'
    sectors_pth = 'a-z_sector_dict'
    prices_pth = 'unnorm_price_series_a-z_-10y-2024-06-26__.pickle'
    layer_args = (model_pth, 200)
    #dataloader = QTrainingData(summaries_pth, prices_pth, seq_len, trade_period, k=k, c_prop=c_prop, full=True, layer_args=layer_args)
    iter = 1
    generate_data = 0
    if generate_data: 
        for i in range(15):
            dataloader_pth = f'Dataset_[a-z, seq-600, n-5, c_prop-{c_prop}]_{i+1+iter}'
            dataloader = QTrainingData(summaries_pth, prices_pth, seq_len, trade_period, k=k, c_prop=c_prop, full=True, load=1, sectors_pth=sectors_pth,data_slice_num=i+iter)
            save_pickle(dataloader.dataloader, dataloader_pth)
    #dataloader = DataLoader(d, 32, shuffle=True)
    
    batch_size = 8
    lr = 1e-9
    num_bins = 320
    dim = 52
    ff_dim = 14500
    n_head = 1
    layers = 24
    epochs = 409
    scale = 0.07
    num_lin_layers = 3
    dtype = torch.float32
    t_max = 15000
    w_decay = 0.03
    grd_nrm = 1.0
    debug_cuda = False
    train = 1
    
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
    #V_model = Models.V_DirectPredSumTran(seq_len, dim, num_bins, n_head, ff_dim, layers).to('cuda')
    if train:
        use_wandb = 1
        use_dist = False
        #model = torch.load(model_pth)
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
        else:
            # = [dataloader_pth, dataloader_pth2]
            #dataloader = pic_load(dataloader_pth)
            #data_len = len(dataloader)
            #dataloader = DataLoader(dataloader, batch_size, pin_memory=True, shuffle=True)
            #model = torch.load(model_pth)
            #torch.cuda.empty_cache()
            model = Dist_Pred(seq_len=seq_len, data_dim=dim, num_bins=num_bins, nhead=n_head, ff=ff_dim, layers=layers, sum_emb=832, scale=scale, num_cls_layers=num_lin_layers).to('cuda', dtype=dtype)
            #model = DataParallel(model).to('cuda')
            model = model.to('cuda')
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
        print('Loaded Pic') 
        world_size = 2
        save_rate = 1
        misc = [batch_size, trade_period, dataloader_pth_list, save_rate]
        params = sum(p.numel() for p in model.parameters())
        print('Network size: ', params)
        #save_name = f'Dist_pred_{params}'
        save_name = f'{model_pth}_'
        Train_Dist_Direct_Predictor(0, model, epochs, save_name, lr, num_bins, t_max, w_decay, grd_nrm, misc=misc, epsilon=0.5, device=f'cuda', dtype=dtype)
    else:
        print(0)
    
    
if __name__ == '__main__':
    main()