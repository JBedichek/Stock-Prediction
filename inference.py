import torch
import torch.nn as nn
import datetime
from utils import *
from Models import *
from Training import QTrainingData, GenerateDataDict
from tqdm import tqdm
import os

class stock_inference:
    def __init__(self, model_pth, s_pth, p_pth, sectors_pth, device='cuda', start_date=(2024, 5, 1), 
                 end_date=(2024, 5, 1), n=5, seq_len=50, num_bins=160, load_dataset=False, data_dict_pth='DataDict_2024-06-26'):
        # Data dict -> date : company : data(tensor)
        self.num_bins = num_bins
        self.start_date = datetime.date(start_date[0], start_date[1], start_date[2])
        self.end_date = datetime.date(end_date[0], end_date[1], end_date[2])
        self.model = torch.load(model_pth).eval().to(device)
        if load_dataset:
            self.dataset = pic_load('InferenceDataset')
            print('Loaded Dataset...')
        else:
            
            self.dataset = QTrainingData(s_pth, p_pth, seq_len, n=n, full=False, load=1, data_dict_pth=data_dict_pth,
                                        inference=(self.start_date, self.end_date), sectors_pth=sectors_pth,
                                    )
            save_pickle(self.dataset, 'InferenceDataset')
        self.companies = self.dataset.companies
        self.i_dataset = self.dataset.inference_data
        self.n = n
        self.cum_money = 1000

    def get_company_date_price(self, company, date):
        day_data = self.i_dataset[date][company][2].item()
        #print(day_data)
        return day_data
    
    def get_expected_q(self, pred, device='cuda'):
        '''
        Takes a vector of logits, corresponding to probability bins, and outputs a mean
        price prediction.
        '''
        softmax = nn.Softmax(dim=0)
        #reward_values = torch.linspace(-0.2, 0.2, self.num_bins).repeat(2, 1).to(device)
        reward_values = torch.linspace(-0.2, 0.2, self.num_bins, device='cuda:0')
        #expected_reward = torch.sum(reward_values*softmax(pred), dim=1)
        expected_reward = torch.sum(reward_values*softmax(pred), dim=0)
        #r_val = torch.max(expected_reward).item()
        return expected_reward

    def get_top_q_buy(self, top_n, date, show=True, low=False, entropy=True):
        '''
        Iterates through all companies in the dataset on a given date, gets
        the model predictions, and selects the top_n best companies to buy based
        on those predictions.
        '''
        with torch.no_grad():
            pred = {}
            best_pred = {}
            entropies = {}
            for company in tqdm(self.companies):
                try:
                    data = self.i_dataset[date][company]
                    if data is None:
                        pass
                    else:
                        data, summary, price = data[0].unsqueeze(0), data[1].unsqueeze(0), data[2]
                        data = set_nan_inf(data)
                        price_pred = self.model(data.to('cuda'), summary.to('cuda'))
                        price_pred = price_pred.squeeze(0)
                        n = price_pred.size(0)
                        prices = []
                        _entropies = []
                        for i in range(4):
                            prices.append(self.get_expected_q(price_pred[:,i], self.num_bins))
                            _entropies.append(get_distribution_entropy(price_pred[:,i]))
                        price_pred = prices[self.n-2]
                        pred[company] = price_pred
                        entropies[company] = _entropies[self.n-2]
                        best_pred[price_pred] = company
                except Exception as e:
                    pass
        
        b_p = sorted(list(best_pred.keys()))
        if low:
            b_p = b_p[:top_n]
        else:
            b_p = b_p[-top_n:]
        n_companies = [best_pred[p] for p in b_p]

        if entropy:
            entropy_values = [entropies[company] for company in n_companies]
            min_entropy_index = entropy_values.index(min(entropy_values))
            n_companies = [n_companies[min_entropy_index]]
        if show:
            #print('TOP PROFIT PREDICTIONS')
            if entropy:
                #print(n_companies[0], ' ', f'{100*pred[n_companies[0]]:.5f}')
                print(n_companies)
            else:
                for p in b_p:
                    c = best_pred[p]
                    #print(p)
                    print(c, ' ', f'{100*p:.5f}%')
        if n_companies is not None:
            return n_companies
        else: 
            print(best_pred)
            raise ValueError
        
    def get_date_profit(self, date, companies, low=False):
        if len(companies) == 0:
            print('ERROR')
            return 
        ind = self.dataset.iterable_dates.index(date)
        buy_prices = [self.get_company_date_price(c, date) for c in companies]
        sell_date = self.dataset.iterable_dates[ind+(self.n-1)]
        sell_prices = [self.get_company_date_price(c, sell_date) for c in companies]
        profit = []
        for i in range(len(buy_prices)):
            profit.append(sell_prices[i]/buy_prices[i])
        profit = sum(profit)/len(profit)
        if low:
            profit = 1+1-profit # This corresponds to selling short
        return profit
    
    def run_trading_sim(self, n, date, period_len, show=True, low=False, entropy=True):
        ind = self.dataset.iterable_dates.index(date)
        dates = self.dataset.iterable_dates[ind:ind+period_len]
        profits = []
        counter = 0
        for i in range(len(dates)):
            if (i+self.n) % self.n == 0:
                companies = self.get_top_q_buy(n, dates[i], show=show, low=low, entropy=entropy)
                profit = self.get_date_profit(dates[i], companies, low=low)
                profits.append(profit)
                self.cum_money = self.cum_money*profit
                counter += 1
                print(f'\r{counter}: Point Profit: {profit} | Average Profit: {sum(profits)/len(profits)} | Account: ${self.cum_money:.2f}')
        return profits
    
    def get_price_volume_bounds(self, date):
        '''
        This calculates the upper and lower bounds, as well as the mean and 
        std deviation of the total money traded for companies on the stock market
        on a given date (price * volume). Downstream, it is used to generate a z-score
        and unit cube placement of individual companies to get a relative indicator of
        market cash flow on the stock. 
        '''
        price_volumes = []
        for company in self.companies:
            try:
                day_data = self.dataset.data[date][company]
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
    
    def get_seq_pv_stats(self):
        #ind = self.dataset.iterable_dates.index(date)
        dates = self.dataset.iterable_dates
        mins, maxs, means, stds = [], [], [], []
        for date in dates:
            min, max, mean, std = self.get_price_volume_bounds(date)
            mins.append(min.item())
            maxs.append(max.item())
            means.append(mean.item())
            stds.append(std.item())
        min = sum(mins)/len(mins)
        max = sum(maxs)/len(maxs)
        mean = sum(means)/len(means)
        std = sum(stds)/len(stds)
        print(min, max, mean, std)
    

def main():
   # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    l_model_pth = 'L1_Dist600_E22'
    b_model_pth = 'Dist600__E12'
    model_pth = 'inf_model'
    gen_model = 1

    if gen_model:
        model = Composed_Dist_Pred(b_model_pth, l_model_pth)
        torch.save(model, model_pth)

    summaries_pth = 'i_summary_embs.pickle'
    sectors_pth = 'i_sector_dict'
    prices_pth = 'unnorm_price_series_i_-10y-2024-07-01__.pickle'
    data_dict_pth = 'DataDict_inf_10y'
    load_dataset = 0
    low = 0
    entropy = 0
    a = stock_inference(model_pth, summaries_pth, prices_pth, start_date=(2024, 1, 3), end_date=(2024, 6, 26), 
            n=2, seq_len=600, load_dataset=load_dataset, num_bins=320, sectors_pth=sectors_pth)
    a.run_trading_sim(1, datetime.date(2024, 4, 5), 55, low=low, entropy=entropy)
if __name__ == '__main__':
    main()