"""Dataset processing utilities for stock prediction.

This module contains classes and functions for:
- Converting raw scraped data into efficient data structures (GenerateDataDict)
- Creating training/inference datasets (QTrainingData)
- Helper functions for data preprocessing and binning
"""

import torch
from torch.utils.data import Dataset
import datetime
import random
import pickle
from tqdm import tqdm
import numpy as np

from training.contrastive_pretraining import gauss_normalize, prepare_ae_data
from training.models import PositionalEncoding
from utils.utils import pic_load, set_nan_inf, cube_normalize, get_abs_price_dim


class GenerateDataDict:
    def __init__(self, bulk_prices_pth, summaries_pth, seq_len,
                 save=True, j=True, save_name=None):
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
            print('Saving Data Dict')
            if save_name:
                torch.save(self.r_data, save_name+'.pt')
                #with open(save_name, 'wb') as f:
                #    pickle.dump(self.r_data, f)
            else:
                torch.save(self.r_data, f'DataDict_{datetime.date.today()}.pt')
                #with open(f'DataDict_{datetime.date.today()}', 'wb') as f:
                #    pickle.dump(self.r_data, f)


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
                    day_of_the_week = date.isoweekday()
                    day_of_the_week = torch.nn.functional.one_hot(torch.tensor(day_of_the_week)-1, 5).unsqueeze(0)
                    #print(day_of_the_week.shape, a.shape)
                    b = torch.cat((a,day_of_the_week),dim=1)
                    self.data[date][company] = {'Price':a,"Price_With_Day":b}
                    counter += 1
                    print(f'\r{counter}, {counter2} / {top}', end='')
        return self.data

class QTrainingData(Dataset):
    def __init__(self, summaries_pth, bulk_prices_pth, seq_len, n=5, k=1,
        c_prop=0.5, full=True, layer_args=None, load=True, inference=None,
        sectors_pth=str, data_slice_num=int, data_dict_pth=str,inf_company_keep_rate=1):
        self.data_slice_num = data_slice_num
        if layer_args is not None:
            mdl_pth = layer_args[0]
            self.seq_len = seq_len
        self.seq_len = seq_len
        self.i_keep = inf_company_keep_rate
        self.n = n # Trade period len
        self.k = k # How often to take a point from the seqence
        print('Loading Summaries...')
        self.summaries = pic_load(summaries_pth)
        print('Loading Sectors...')
        self.sectors_info_dict = pic_load(sectors_pth)
        print('Loading Fundamentals...')
        # Try to load fundamentals - if not found, use None (for backward compatibility)
        try:
            fundamentals_pth = data_dict_pth.replace('DataDict', 'fundamentals').replace('.pt', '.pkl')
            self.fundamentals = pic_load(fundamentals_pth)
            print(f'✅ Loaded {len(self.fundamentals)} fundamental metric tensors')
        except Exception as e:
            print(f'⚠️  No fundamentals found ({e}), will run without fundamental features')
            self.fundamentals = None
        if load:
            #self.data = pic_load('DataDict_inf_10y')
            print('Loading Data Dict...')
            self.data = torch.load(data_dict_pth+'.pt')
            #self.data = pic_load(data_dict_pth)
            #torch.save(self.data, data_dict_pth+'.pt')
        else:
            print('Using ', bulk_prices_pth, 'as ', data_dict_pth)
            self.data = GenerateDataDict(bulk_prices_pth, summaries_pth, seq_len,
                                         save=True, save_name=f'raw_{data_dict_pth}')
            self.data = self.data.r_data
            torch.save(self.data, data_dict_pth+'.pt')
            #save_pickle(self.data, data_dict_pth)

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
            #self.prepare_dataset_layer(c_prop)
        if inference is not None:
            print("Generating inference dataset...")
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
                day_data = self.data[date][company]['Price']
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
            comp_data = self.data[dates[0]][company]['Price_With_Day'].unsqueeze(0)
            pca_stats = self.pca_stats[dates[0]].unsqueeze(0)
            rel_data = self.relative_price_dict[dates[0]][company]
            #print(comp_data)
            for _date in dates[1:]:
                rel_data = torch.cat((rel_data, self.relative_price_dict[_date][company]), dim=0)
                comp_data = torch.cat((comp_data, self.data[_date][company]['Price_With_Day'].unsqueeze(0)), dim=0)
                pca_stats = torch.cat((pca_stats, self.pca_stats[_date].unsqueeze(0)), dim=0)
            #print(rel_data.shape, comp_data.shape, pca_stats.shape)
            comp_data = comp_data.unsqueeze(0)
            c = self.summaries[company]
            price = self.data[dates[-1]][company]['Price'][0][-2] # For computing target value

            # Get fundamental metrics for the SPECIFIC date (time-varying!)
            if self.fundamentals is not None and company in self.fundamentals:
                from datetime import datetime
                target_date = dates[-1]  # The prediction date (last in sequence)

                # Convert to datetime.date if needed (FMP fundamentals use datetime.date keys)
                if isinstance(target_date, str):
                    target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
                elif isinstance(target_date, datetime):
                    target_date = target_date.date()

                # Look up fundamentals for this specific date
                if target_date in self.fundamentals[company]:
                    fundamentals = self.fundamentals[company][target_date]
                else:
                    # Date not in fundamentals (e.g., weekends, holidays) - use zeros
                    fundamentals = torch.zeros(27)
            else:
                # If fundamentals not available for this company, use zeros (27 dimensions)
                fundamentals = torch.zeros(27)
        except Exception as e:
            #print(e)
            return None
        return (comp_data, c, pca_stats, price, rel_data, fundamentals)

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
                prices.append(self.data[date][company]['Price'][0][-2])
            #print(data)
        except Exception as e:
            #print(e)
            return None
        if data is None:
            #print('Fuck')
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
                day_data = self.data[date][company]['Price']
                prev_day_data = self.data[prev_date][company]['Price']
                change = (day_data-prev_day_data)/prev_day_data
            except Exception:
                change = torch.zeros(5).unsqueeze(0)
                #change = torch.zeros(5)
            stats.append(change)
            self.relative_price_dict[date][company] = change
        stats = torch.stack(stats, 0)
        stats = stats.squeeze(1)
        stats = set_nan_inf(stats)
        #u, s, v = torch.pca_lowrank(stats, q=pca_dim) # 3000 x dim, dim, dimx5
        mean = torch.mean(stats,  0) # 1 x 5
        std = torch.std(stats, 0) # 2 x 5
        #r_data = torch.cat((v.flatten(), mean.flatten(), std.flatten()), dim=0) # 35 dimensional
        r_data = torch.cat((mean.flatten(), std.flatten()), dim=0)
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
        do_corrected = 1
        sector, industry = self.sectors_info_dict[company][0], self.sectors_info_dict[company][1]
        sector_ind, industry_ind = self.sectors.index(sector), self.industries.index(industry)
        if do_corrected:
            try:
                self.sector_one_hot_encoding = torch.nn.functional.one_hot(torch.tensor(sector_ind), 11)
                self.industry_one_hot_encoding = torch.nn.functional.one_hot(torch.tensor(industry_ind), 170)
            except Exception as e:
                self.sector_one_hot_encoding = torch.nn.functional.one_hot(torch.tensor(10), 11)
                self.industry_one_hot_encoding = torch.nn.functional.one_hot(torch.tensor(169), 170)
        else:
            self.sector_one_hot_encoding = torch.nn.functional.one_hot(torch.tensor(sector_ind), len(self.sectors))
            self.industry_one_hot_encoding = torch.nn.functional.one_hot(torch.tensor(industry_ind), len(self.industries))
        #sector_emb, industry_emb = self.sectors_sinusoidal_encoding[sector_ind, :], self.industry_sinusoidal_encoding[industry_ind, :]
        #final_emb = torch.cat((summary, sector_emb, industry_emb), dim=0).detach().cpu()
        app = torch.ones((988-768))
        summary = torch.cat((app, summary), dim=0)
        final_emb = summary.cpu()
        return final_emb

    def prepare_dataset(self, c_prop):
        counter = 0
        c_prop = int(c_prop*len(self.companies))
        print(len(self.companies), ' Companies Available')
        c_slice_ind = self.data_slice_num
        start = 0 + c_slice_ind*c_prop
        #start = 0
        companies = self.companies[start:start+c_prop]
        print(f'Generating Dataset ({len(companies)} companies)')
        for date in tqdm(self.iterable_dates):
            counter += 1
            s = self.get_price_volume_bounds(date)
            stats = self.pca_stats[date]
            companies = self.companies[start:start+c_prop]
            #companies = random.sample(self.companies, c_prop)
            #print('len companies', len(companies))
            #print(date)
            for i, company in enumerate(companies):
                if (counter-2) % self.k == 0:
                    #print(f'\r{counter}/{int(len(self.iterable_dates)/self.k)}', end='')
                    all_data = self.get_n_company_prices(company, date)
                    r_data = []
                    if all_data is None:

                        pass
                    else:
                        data, prices = all_data[0], all_data[1]
                        prices = [torch.tensor(price).cpu() for price in prices]
                        _data = data[0]
                        summary = data[1].squeeze(0).cpu()
                        summary = self.add_sector_industry_embedding(summary, company)
                        sector_one_hot = self.sector_one_hot_encoding.repeat(600,1)
                        industry_one_hot = self.industry_one_hot_encoding.repeat(600,1)
                        stats = data[2]
                        price = data[3]
                        rel_data = data[4]
                        fundamentals = data[5].cpu()  # Time-varying fundamentals (27 dims)

                        # Add fundamentals to summary embedding (insert after BERT, before padding)
                        # Current: [768 BERT] + [220 padding] = 988
                        # New: [768 BERT] + [27 fundamentals] + [193 padding] = 988
                        summary = torch.cat([
                            summary[:768],  # BERT embedding
                            fundamentals,   # Fundamental metrics (27 dims)
                            torch.ones(193) # Reduced padding to maintain 988 total
                        ], dim=0)
                        #print(_data.shape)
                        _data = self.prepare_seq_data(_data, s).squeeze(0) # Shape 1 x 300 x 12
                        #print(_data.shape)
                        _data = torch.cat((_data.cpu(), stats.cpu(), rel_data.cpu()), dim=1).to(device='cpu', dtype=torch.float32)
                        #print(_data.shape)
                        _data = torch.cat((_data.cpu(), sector_one_hot, industry_one_hot), dim=1).to(device='cpu', dtype=torch.float32)
                        #print(_data.shape)
                        #if i == 3:
                        #    print(_data.shape)
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
                                fundamentals = list_data[5].cpu()  # Time-varying fundamentals (27 dims)

                                # Add fundamentals to summary embedding
                                summary = torch.cat([
                                    summary[:768],  # BERT embedding
                                    fundamentals,   # Fundamental metrics (27 dims)
                                    torch.ones(193) # Padding to maintain 988 total
                                ], dim=0)
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

    def inference_dataset(self, start_date, end_date, layer=False):
        counter = 0
        company_keep_rate = self.i_keep

        self.companies = random.sample(self.companies,int(len(self.companies)*company_keep_rate))
        # For Debugging
        print('First dates: ', self.iterable_dates[0:3], 'Last dates: ',
              self.iterable_dates[-3:], 'Num dates: ', len(self.iterable_dates),
              'num companies: ', len(self.companies))

        s_ind = self.iterable_dates.index(start_date)
        e_ind = self.iterable_dates.index(end_date)
        dates = self.iterable_dates[s_ind:e_ind]
        #companies = self.companies[0:int(len(self.companies)/4)]
        for date in tqdm(dates):
            counter += 1
            s = self.get_price_volume_bounds(date)
            stats = self.pca_stats[date]
            self.inference_data[date] = {}
            for company in self.companies:
                all_data = self.get_n_company_prices(company, date)
                if all_data is None:
                    pass
                else:
                    data, prices = all_data[0], all_data[1]
                    prices = [torch.tensor(price).cpu() for price in prices]
                    _data = data[0]
                    summary = data[1].squeeze(0).cpu()
                    summary = self.add_sector_industry_embedding(summary, company)
                    sector_one_hot = self.sector_one_hot_encoding.repeat(600,1)
                    industry_one_hot = self.industry_one_hot_encoding.repeat(600,1)
                    stats = data[2]
                    price = data[3]
                    rel_data = data[4]
                    fundamentals = data[5].cpu()  # Time-varying fundamentals (27 dims)

                    # Add fundamentals to summary embedding
                    summary = torch.cat([
                        summary[:768],  # BERT embedding
                        fundamentals,   # Fundamental metrics (27 dims)
                        torch.ones(193) # Padding to maintain 988 total
                    ], dim=0)
                    #print(stats.shape)
                    #print(_data.shape)
                    _data = self.prepare_seq_data(_data, s).squeeze(0) # Shape 1 x 300 x 12
                    #print(_data.shape)
                    _data = torch.cat((_data.cpu(), stats.cpu(), rel_data.cpu()), dim=1).to(device='cpu', dtype=torch.float32)
                    #print(_data.shape)
                    _data = torch.cat((_data.cpu(), sector_one_hot, industry_one_hot), dim=1).to(device='cpu', dtype=torch.float32)
                    #print(_data.shape)
                    #print(_data.shape)
                    self.inference_data[date][company] = [_data, summary, price]


    def __len__(self):
        return len(self.dataloader)

    def __getitem__(self, ind):
        return self.dataloader[ind]


# Helper functions for data preprocessing

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

def _convert_reward_to_one_hot_batch(values, bin_edges, correct=False, with_smoothing=0, smoothing=0.2):
    """
    Converts continuous reward values into a one-hot encoding based on dynamic bin edges.

    Args:
        values (Tensor): Continuous reward values (batch_size,).
        bin_edges (Tensor): Edges of the bins (num_bins + 1,).
        correct (bool): If True, adjusts the input values.
        with_smoothing (int): If True, applies smoothing to the one-hot encoding.
        smoothing (float): Smoothing factor for label smoothing (default=0.2).

    Returns:
        Tensor: One-hot encoded rewards (batch_size, num_bins).
    """
    #bin_edges = torch.load('Bin_Edges_200')
    num_bins = len(bin_edges) - 1  # Number of bins derived from bin_edges

    # Adjust values if the correct flag is set
    if correct:
        values = values - 1

    # Compute bin indices for each value
    bin_indices = torch.bucketize(values, bin_edges, right=False) - 1  # bucketize returns 1-based indices
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)  # Ensure indices are within [0, num_bins-1]

    # Create one-hot encoding
    one_hot = torch.zeros(values.size(0), num_bins, device=values.device)
    one_hot.scatter_(1, bin_indices.unsqueeze(1), 1.0)

    if with_smoothing:
        smooth_mass = smoothing / 2.0  # Spread smoothing equally to neighbors
        center_mass = 1.0 - smoothing

        # Reset the center bin with smoothed value
        one_hot = one_hot * center_mass

        # Add smoothing to neighboring bins
        for i in range(values.size(0)):
            center_bin = bin_indices[i]
            if center_bin > 0:  # Left neighbor
                one_hot[i, center_bin - 1] += smooth_mass
            if center_bin < num_bins - 1:  # Right neighbor
                one_hot[i, center_bin + 1] += smooth_mass

    return one_hot

def convert_reward_to_one_hot_batch(values, num_bins, bounds=0.2, correct=True, with_smoothing=0, smoothing=0.2):
    # Adjust values if correct flag is set
    if correct:
        values = values - 1

    # Scale values to bin indices
    scaled_values = (values + bounds) / (bounds * 2) * (num_bins - 1)
    bin_indices = scaled_values.long()

    # Clip bin indices to be within [0, num_bins-1]
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)


    # Create one-hot encoding
    #print(values.shape)
    one_hot = torch.zeros(values.size(0), num_bins, device=values.device)
    one_hot.scatter_(1, bin_indices.unsqueeze(1), 1.0)
    if with_smoothing:
        smooth_mass = smoothing / 2.0  # Spread smoothing equally to neighbors
        center_mass = 1.0 - smoothing

        # Reset the center bin with smoothed value
        one_hot = one_hot * center_mass

        # Add smoothing to neighboring bins
        for i in range(values.size(0)):
            center_bin = bin_indices[i]
            if center_bin > 0:  # Left neighbor
                one_hot[i, center_bin - 1] += smooth_mass
            if center_bin < num_bins - 1:  # Right neighbor
                one_hot[i, center_bin + 1] += smooth_mass



    return one_hot

def smooth_one_hot_reward(targets, num_bins, smoothing=0.1):
    """
    Converts regression targets into a smoothed one-hot encoding across bins.

    Args:
        targets (Tensor): Tensor of shape (batch_size,) with continuous values.
        num_bins (int): Total number of bins to discretize the targets.
        smoothing (float): Smoothing factor for label smoothing (0 means pure one-hot).

    Returns:
        Tensor: Smoothed one-hot target of shape (batch_size, num_bins).
    """
    assert 0.0 <= smoothing <= 1.0, "Smoothing factor must be between 0 and 1"

    # Compute the bin index for each target
    bin_width = 1.0 / num_bins
    target_bins = torch.clamp((targets / bin_width).long(), 0, num_bins - 1)

    # Initialize the smoothed target distribution
    batch_size = targets.size(0)
    smoothed_targets = torch.zeros(batch_size, num_bins, device=targets.device)

    # Define smoothing weights
    smooth_mass = smoothing / 2.0  # Spread smoothing equally to neighbors
    center_mass = 1.0 - smoothing  # Remaining weight on the target bin

    for i in range(batch_size):
        # Get the target bin index
        target_bin = target_bins[i]

        # Distribute mass to center, left neighbor, and right neighbor
        smoothed_targets[i, target_bin] += center_mass
        if target_bin > 0:  # Left neighbor
            smoothed_targets[i, target_bin - 1] += smooth_mass
        if target_bin < num_bins - 1:  # Right neighbor
            smoothed_targets[i, target_bin + 1] += smooth_mass

    return smoothed_targets

def prepare_direct_pred_data(data, device='cuda', dtype=torch.float32):
    data = data.squeeze(2)
    _data = set_nan_inf(data).to(device)
    _data = gauss_normalize(_data, dim=1)
    __data = cube_normalize(_data, dim=1)
    a_data = get_abs_price_dim(data).to(device)
    _data = torch.cat((_data, __data, a_data), dim=2).to(device, dtype=dtype)
    return _data
