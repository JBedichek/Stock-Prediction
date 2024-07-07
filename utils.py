import pickle
import torch


def get_distribution_entropy(tensor):
    return (-torch.sum(tensor * torch.log(tensor))).item()

def gaussian_noise(tensor, noise_level=5e-5):
    noise = torch.randn_like(tensor) * noise_level
    return tensor+noise

def save_pickle(data, name):
    with open(name, 'wb') as out:
        pickle.dump(data, out)
    print(f'Saved {name}...')

def pic_load(pic):
    with open(pic, 'rb') as infile:
        return pickle.load(infile)

def cube_normalize(tensor, n_min=-1, n_max=1, dim=1):
    '''
    Takes a tensor and projects all elements onto a unit
    cube, size -1 to 1. 
    '''
    min, max = tensor.min(dim=dim, keepdim=True)[0], tensor.max(dim=dim, keepdim=True)[0]
    return (tensor-min)/(max-min)*(n_max-n_min) + n_min

def set_nan_inf(data):
    '''
    This eliminates any NaN or inf in a tensor by
    setting it to one, so downstream operations are
    minimally disrupted.
    '''
    if torch.isinf(data).any():
        inf_mask = torch.isinf(data)
        data[inf_mask] = 1
    if torch.isnan(data).any():
        nan_mask = torch.isnan(data)
        data[nan_mask] = 1
    return data

def get_abs_price_dim(price_seq):
    '''
    
    '''
    # Shape (batch, seq, dim)
    # 2.465073550667833 16111693133.838446 99981918.3880809 557781883.0235642
    a = 1
    _min, _max, mean, std = torch.tensor(2.465073550667833), torch.tensor(16111693133.838446), torch.tensor(99981918.3880809), torch.tensor(557781883.0235642)
    close = price_seq[:, :, -2]
    volume = price_seq[:, :, -1]
    price_volume = close*volume
    gauss_price_volume = (price_volume-mean)/std
    cube_price_volume = 2*(price_volume-_min)/(_max-_min)-1
    added_data = torch.cat((gauss_price_volume.unsqueeze(2), cube_price_volume.unsqueeze(2)), dim=2).to(torch.float32)
    return added_data

def get_expected_price(pred, num_bins, device='cuda'):
        '''
        Takes a vector of logits, corresponding to probability bins, and outputs a mean
        price prediction.
        '''
        softmax = torch.nn.Softmax(dim=0)
        #reward_values = torch.linspace(-0.2, 0.2, self.num_bins).repeat(2, 1).to(device)
        reward_values = torch.linspace(-0.2, 0.2, num_bins).to(device)
        #expected_reward = torch.sum(reward_values*softmax(pred), dim=1)
        expected_reward = torch.sum(reward_values*softmax(pred), dim=0)
        #r_val = torch.max(expected_reward).item()
        return expected_reward