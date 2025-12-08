from transformers import Adafactor
import gc 
from Models import RobertaEncoder
from Training import QTrainingData
import torch

def format_test_data_series(data):
    # Take out news articles to test Q network on stock data
    _data = {}
    for date, info in data.items():
        if info[0]['APH'].numel() != 0:
            _data[date] = info[0]
    return _data

def preprocess_q_data(data, device='cuda'):
    data = set_nan_inf(data)
    data = data.squeeze(2).to(device)
    a = gauss_normalize(data[:,:,-5:], dim=1).to(device, dtype=torch.float32)
    data[:,:,-5:] = a
    return data.to(torch.float32).permute(1,0,2)

def Train_MC_Distributional_DQN(model, dataloader, epochs, save_name, lr, bs, num_bins, epsilon=0.5, device='cuda', hold_weight=0.04, dtype=torch.float16):
    # Data format: key(Datetime.date) : {company : value(vector of features)}
    dqn = model
    for param in dqn.parameters():
        param = param.to(dtype)
    #optim = torch.optim.Adam(dqn.parameters(), lr=lr)
    optim = Adafactor(dqn.parameters())
    #optim = Adafactor(dqn.parameters(), relative_step=False, scale_parameter=False, lr=lr)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 1550, lr/3)
    loss_fn = torch.nn.CrossEntropyLoss()
    epsilon = epsilon # For random decision making
    no_hold_mask = torch.tensor([True, True, False], requires_grad=False).to(device)
    hold_mask = torch.tensor([True, False, True], requires_grad=False).to(device)
    softmax = nn.Softmax(dim=1)
    for epoch in range(epochs):
        running_loss = 0
        steps = 0
        running_reward = 0
        running_profit = 0
        running_r_profit = 0
        running_r_profit2 = 0
        running_r_profit3 = 0
        print(f"\nEpoch {epoch+1}:")
        for i, n_step_data_list in enumerate(dataloader):
            if i > 0:
                q_vals = torch.zeros((num_bins, 1)).to(device)
                mask = no_hold_mask
                r_mask = no_hold_mask
                r_mask2 = no_hold_mask
                r_mask3 = no_hold_mask
                rewards = torch.tensor([0]).to(device)
                buy_price = torch.tensor([0]).to(device)
                sell_price = torch.tensor([0]).to(device)
                r_buy_price = torch.tensor([0])
                r_sell_price = torch.tensor([0])
                r_buy_price2 = torch.tensor([0])
                r_sell_price2 = torch.tensor([0])
                r_buy_price3 = torch.tensor([0])
                r_sell_price3 = torch.tensor([0])
                counter = 0
                profit = 0
                r_profit = 0
                r_profit2 = 0
                r_profit3 = 0
                total_r_profit = 0
                for data in n_step_data_list:
                    data = data.squeeze(2)
                    #print(data.shape)
                    _data = set_nan_inf(data)
                    _data = _data.squeeze(2).to(device)
                    r_action = random.randint(0, 1)
                    r_action2 = random.randint(0, 1)
                    r_action3 = random.randint(0, 1)
                    do_r_action = torch.rand(1).item() < epsilon
                    _data = gauss_normalize(_data, dim=1)
                    __data = cube_normalize(_data)
                    _data = torch.cat((_data, __data), dim=2).to(device, dtype=dtype)
                    #print(_data)
                    #print(_data.shape)
                    #_data[:,:,-5:] = a
                    #print(_data[0][0:10])
                    if torch.isinf(_data).any():
                        inf_mask = torch.isinf(_data)
                        _data[inf_mask] = 0
                        print(_data)
                    if torch.isnan(_data).any():
                        nan_mask = torch.isnan(_data)
                        _data[nan_mask] = 0
                        print(_data)
                    #print(_data)
                    pred = dqn(_data, mask)
                    #print(pred)
                    reward_values = torch.linspace(-0.2, 0.2, num_bins).repeat(2, 1).to(device)
                    a = reward_values*softmax(pred)
                    expected_reward = torch.sum(reward_values*softmax(pred), dim=1)
                    if i % 50 == 0:
                        #print(reward_values)
                        print(expected_reward)
                    #if i % 50 == 0:
                        #print(pred)
                    action = torch.argmax(expected_reward, dim=0).item()
                    if do_r_action:
                        action = random.randint(0, 1)
                    q_vals = torch.cat((q_vals, pred[action, :].unsqueeze(1)), dim=1)               
                    if torch.equal(mask, no_hold_mask):
                        counter += 1
                        if action == 0:
                            rewards = torch.cat((rewards, torch.tensor([0], dtype=torch.float32).to(device)), dim=0)
                        if action == 1:
                            mask = hold_mask
                            buy_price = torch.cat((buy_price, data[0][-1][-2].to(device).unsqueeze(0))) # Get closing price
                            rewards = torch.cat((rewards, torch.tensor([0], dtype=torch.float32).to(device)), dim=0)
                    else:
                        counter += 1
                        if action == 0:
                            #rewards = torch.cat((rewards, data[0][-1][-2].to(device).unsqueeze(0)/buy_price[-1]*hold_weight), dim=0)
                            rewards = torch.cat((rewards, torch.tensor([0], dtype=torch.float32).to(device)), dim=0)
                        if action == 1:
                            mask = no_hold_mask
                            sell_price = torch.cat((sell_price, data[0][-1][-2].to(device).unsqueeze(0)), dim=0)
                            rewards = torch.cat((rewards, (torch.tensor([(sell_price[-1]/buy_price[-1])-1], dtype=torch.float32).to('cuda'))), dim=0)
                            profit += torch.tensor([(sell_price[-1]/buy_price[-1])-1]).item()
                    if torch.equal(r_mask, no_hold_mask):
                        if r_action == 1:
                            r_mask = hold_mask
                            r_buy_price = torch.cat((r_buy_price, data[0][-1][-2].unsqueeze(0))) # Get closing price
                    else:
                        if r_action == 1:
                            r_mask = no_hold_mask
                            r_sell_price = torch.cat((r_sell_price, data[0][-1][-2].unsqueeze(0)), dim=0)
                            r_profit += torch.tensor([(r_sell_price[-1]/r_buy_price[-1])-1]).item()
                    if torch.equal(r_mask2, no_hold_mask):
                        if r_action2 == 1:
                            r_mask2 = hold_mask
                            r_buy_price2 = torch.cat((r_buy_price2, data[0][-1][-2].unsqueeze(0))) # Get closing price
                    else:
                        if r_action2 == 1:
                            r_mask2 = no_hold_mask
                            r_sell_price2 = torch.cat((r_sell_price2, data[0][-1][-2].unsqueeze(0)), dim=0)
                            r_profit2 += torch.tensor([(r_sell_price2[-1]/r_buy_price2[-1])-1]).item()
                    if torch.equal(r_mask3, no_hold_mask):
                        if r_action3 == 1:
                            r_mask3 = hold_mask
                            r_buy_price3 = torch.cat((r_buy_price3, data[0][-1][-2].unsqueeze(0))) # Get closing price
                    else:
                        if r_action3 == 1:
                            r_mask3 = no_hold_mask
                            r_sell_price3 = torch.cat((r_sell_price3, data[0][-1][-2].unsqueeze(0)), dim=0)
                            r_profit3 += torch.tensor([(r_sell_price3[-1]/r_buy_price3[-1])-1]).item()
                target_q = convert_reward_to_one_hot(0, num_bins).unsqueeze(1)
                # Format reward into one hot for cross entropy 
                #if torch.sum(rewards).item() == 0:
                #    rewards[-1] = -0.005
                for i in range(len(rewards)-1):
                    target_q = torch.cat((target_q, convert_reward_to_one_hot(torch.sum(rewards[i+1:]).item(), num_bins).unsqueeze(1)), dim=1)
                loss = loss_fn(q_vals, target_q)
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(dqn.parameters(), 1.0)
                    optim.step()
                    #scheduler.step()
                    optim.zero_grad()
                    running_loss += loss.item()
                    steps += 1
                    running_reward += sum(rewards)
                    running_profit += profit
                    running_r_profit += r_profit
                    running_r_profit2 += r_profit2
                    running_r_profit3 += r_profit3
                    total_r_profit = running_r_profit+running_r_profit2+running_r_profit3
                if steps % 1 == 0 and steps > 0:
                    a = f'\rRunning Epoch Loss: {(running_loss/steps):.7f} Epoch Steps: {steps}/{len(dataloader)} Epoch Average Profit: {(running_profit/steps):.5f}, Epoch Random Profit: {(total_r_profit/3/steps):.5f}, Point Profit: {profit:.3f}, Epsilon: {epsilon:.3f}'
                    print(a, end='')
                    #tqdm.write(a, end=' ')
                #if profit == 0 and epsilon < 0.25:
                #    epsilon += 0.005
                #if steps > :
                #    epsilon = 0
                if steps % 150 == 0 and epsilon > 0.00:
                    epsilon *= 0
                if steps % 2000 == 0:
                    torch.save(dqn, save_name+f'_{steps+8000}')
                    print(f'saved model ({steps}) . . . Loss: {(running_loss/steps):.7f}')

def Train_MC_DQN(model, data, epochs, context_len, save_name, trade_period_len, lr, dtype):
    # Data format: key(Datetime.date) : {company : value(vector of features)}
    data_class = QTrainingData(data, trade_period_len, context_len)
    dqn = model.to('cuda')
    optim = Adafactor(dqn.parameters(), lr=lr, weight_decay=0.1)
    loss_fn = torch.nn.MSELoss()
    epsilon = 0.5 # For random decision making
    for epoch in range(epochs):
        running_loss = 0
        steps = 0
        running_reward = 0
        print(f"Epoch {epoch+1}:")
        for date in data_class.iterable_dates:
            for company in data_class.companies:
                n_step_data_list = data_class.get_n_company_chunks(company, date)
                q_vals = torch.tensor([0]).to('cuda')
                mask = data_class.no_hold_mask
                rewards = torch.tensor([0]).to('cuda')
                buy_price = torch.tensor([0]).to('cuda')
                sell_price = torch.tensor([0]).to('cuda')
                counter = 0
                r_action = torch.rand(1).item() < epsilon
                for data in n_step_data_list:
                    # Action == 0 -> hold, 1 -> buy, 2 -> sell
                    if torch.isinf(data).any():
                        inf_mask = torch.isinf(data)
                        data[inf_mask] = 1
                    if torch.isnan(data).any():
                        nan_mask = torch.isnan(data)
                        data[nan_mask] = 1
                    n_input = cube_normalize(data).to('cuda').unsqueeze(0).float()
                    pred = dqn(n_input, mask)
                    action = torch.argmax(pred).item()
                    if r_action:
                        if action == 1:
                            action = 0
                        if action == 0:
                            action = 1
                    q_vals = torch.cat((q_vals, pred[action].unsqueeze(0)), dim=0)
                    if torch.equal(mask, data_class.no_hold_mask):
                        counter += 1
                        if action == 0:
                            rewards = torch.cat((rewards, torch.tensor([0], dtype=torch.float32).to('cuda')), dim=0)
                        if action == 1:
                            mask = data_class.hold_mask
                            buy_price = torch.cat((buy_price, data[0][3].to('cuda').unsqueeze(0))) # Get closing price
                            rewards = torch.cat((rewards, torch.tensor([0], dtype=torch.float32).to('cuda')), dim=0)
                    else:
                        counter += 1
                        if action == 0:
                            rewards = torch.cat((rewards, torch.tensor([0], dtype=torch.float32).to('cuda')), dim=0)
                        if action == 1:
                            mask = data_class.no_hold_mask
                            sell_price = torch.cat((sell_price, data[0][3].to('cuda').unsqueeze(0)), dim=0)
                            rewards = torch.cat((rewards, (torch.tensor([(sell_price[-1]/buy_price[-1])-1], dtype=torch.float32).to('cuda'))), dim=0)
                target_q = torch.tensor([0], dtype=torch.float32).to('cuda')
                for i in range(len(rewards)-1):
                    target_q = torch.cat((target_q, rewards[i+1].unsqueeze(0)))
                loss = loss_fn(q_vals, target_q).float()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(dqn.parameters(), 1.0)
                optim.step()
                optim.zero_grad()
                running_loss += loss.item()
                steps += 1
                running_reward += sum(rewards)
                if steps % 1 == 0:
                    print(f'\rRunning Epoch Loss: {(running_loss/steps):.6f} Epoch Steps: {steps} Epoch Average Profit: {(running_reward/steps):.6f}, Point Profit: {sum(rewards):.3f}', end=' ')
                if steps % 100 == 0 and epsilon > 0.12 and steps < 10000:
                    epsilon = epsilon*0.96
                if steps > 10000:
                    epsilon = 0.03
    torch.save(dqn, save_name)

def Batch_Train_MC_DQN(model, dataloader, epochs, context_len, save_name, trade_period_len, lr):
    # Data format: key(Datetime.date) : {company : value(vector of features)}
    #data_class = QTrainingData(data, trade_period_len, context_len)
    accumulation = 4
    dqn = model.to('cuda')
    #optim = torch.optim.Adam(dqn.parameters(), lr=lr)
    optim = Adafactor(dqn.parameters())
    loss_fn = torch.nn.MSELoss()
    epsilon = 0.5 # For random decision making
    no_hold_mask = torch.tensor([True, True, False]).to('cuda')
    hold_mask = torch.tensor([True, False, True]).to('cuda')
    for epoch in range(epochs):
        running_loss = 0
        steps = 0
        running_reward = 0
        print(f"Epoch {epoch+1}:")
        for i, n_step_data_list in enumerate(dataloader):
            q_vals = torch.tensor([0]).to('cuda')
            mask = no_hold_mask
            rewards = torch.tensor([0]).to('cuda')
            buy_price = torch.tensor([0]).to('cuda')
            sell_price = torch.tensor([0]).to('cuda')
            counter = 0
            for data in n_step_data_list:
                # Action == 0 -> hold, 1 -> buy, 2 -> sell
                r_action = torch.rand(1).item() < epsilon
                n_input = cube_normalize(data).to('cuda').float()
                pred = dqn(n_input, mask)
                action = torch.argmax(pred).item()
                if r_action:
                    if action == 1:
                        action = 0
                    if action == 0:
                        action = 1
                q_vals = torch.cat((q_vals, pred[action].unsqueeze(0)), dim=0)
                if torch.equal(mask, no_hold_mask):
                    counter += 1
                    if action == 0:
                        rewards = torch.cat((rewards, torch.tensor([0], dtype=torch.float32).to('cuda')), dim=0)
                    if action == 1:
                        mask = hold_mask
                        #print(data[0][3].to('cuda').unsqueeze(0))
                        buy_price = torch.cat((buy_price, data[0][0][3].to('cuda').unsqueeze(0))) # Get closing price
                        rewards = torch.cat((rewards, torch.tensor([0], dtype=torch.float32).to('cuda')), dim=0)
                else:
                    counter += 1
                    if action == 0:
                        rewards = torch.cat((rewards, torch.tensor([0], dtype=torch.float32).to('cuda')), dim=0)
                    if action == 1:
                        mask = no_hold_mask
                        sell_price = torch.cat((sell_price, data[0][0][3].to('cuda').unsqueeze(0)), dim=0)
                        #print(sell_price)
                        rewards = torch.cat((rewards, (torch.tensor([(sell_price[-1]/buy_price[-1])-1], dtype=torch.float32).to('cuda'))), dim=0)
            target_q = torch.tensor([0], dtype=torch.float32).to('cuda')
            for i in range(len(rewards)-1):
                target_q = torch.cat((target_q, rewards[i+1].unsqueeze(0)))
            loss = loss_fn(q_vals, target_q).float()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(dqn.parameters(), 1.0)
            if steps % accumulation == 0:
                optim.step()
                optim.zero_grad()
            running_loss += loss.item()
            steps += 1
            running_reward += sum(rewards)
            if sum(rewards) == 0:
                epsilon += 0.05
            if steps % 1 == 0:
                print(f'\rRunning Epoch Loss: {(running_loss/steps):.6f} Epoch Steps: {steps}/{len(dataloader)} Epoch Average Profit: {(running_reward/steps)/bs:.6f}, Point Profit: {sum(rewards)/bs:.3f}, Epsilon: {epsilon}', end=' ')
            if steps % 50 == 0 and epsilon > 0.07:
                epsilon = epsilon*0.95
    torch.save(dqn, save_name)

def Run_DDP_Training(world_size, gl_model, dataloader, epochs, save_name, lr, bs, num_bins, t_max, weight_decay, grd_nrm, dtype=torch.float16):
    mp.spawn(DDP_Train_Direct_Predictor,
            args=(world_size, gl_model, dataloader, epochs, save_name, lr, bs, num_bins, t_max, weight_decay, grd_nrm, dtype),
            nprocs=world_size,
            join=True)

def DDP_Train_Direct_Predictor(world_size, gl_model, dataloader, epochs, save_name, lr, bs, num_bins, t_max, weight_decay, grd_nrm, dtype=torch.float16):
    #os.environ[]
    dist.init_process_group(backend="ncll")
    rank = dist.get_rank()
    device = f"cuda:{rank}"
    gl_model.device = device
    gl_model = DDP(gl_model, device_ids=[rank])
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataloader=dataloader,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataloader,
        batch_size=bs,
        num_workers=0,
        sampler=sampler
    )
    Train_Direct_Predictor_gl(rank, gl_model, dataloader, epochs, save_name, lr, bs, num_bins, t_max, weight_decay, grd_nrm, epsilon=0.5, device=device, dtype=dtype)
    dist.destroy_process_group()

def Train_Direct_Predictor(v_model, gl_model, dataloader, epochs, save_name, lr, bs, num_bins, t_max, weight_decay, grd_nrm, epsilon=0.5, device='cuda', dtype=torch.float16):
    for param in v_model.parameters():
        param = param.to(dtype)
    for param in gl_model.parameters():
        param = param.to(dtype)
    optim_v = torch.optim.AdamW(v_model.parameters(), lr=lr)
    optim_gl = torch.optim.AdamW(gl_model.parameters(), lr=lr)
    scheduler_v = torch.optim.lr_scheduler.CosineAnnealingLR(optim_v, T_max=t_max, eta_min=1e-13)
    scheduler_gl = torch.optim.lr_scheduler.CosineAnnealingLR(optim_gl, T_max=t_max, eta_min=1e-13)
    #optim = Adafactor(dqn.parameters(), weight_decay=0.1)
    #optim = Adafactor(dqn.parameters(), relative_step=False, scale_parameter=False, lr=lr)
    loss_fn_v = torch.nn.MSELoss()
    loss_fn_gl = torch.nn.CrossEntropyLoss()
    epsilon = epsilon # For random decision making
    #for i, n_step_data_list in enumerate(dataloader):
    #    if i < 10:
    #        valid_data.append(n_step_data_list)
    for epoch in range(epochs):
        running_loss_v = 0
        running_loss_gl = 0
        steps = 0
        print(f"\nEpoch {epoch+1}:")
        for i, n_step_data_list in enumerate(dataloader):
            if i > 0:
                counter = 0
                profit = 0
                for data in n_step_data_list:
                    data, summary = data[0], data[1]
                    data = data.squeeze(1, 3)
                    summary = summary.squeeze(1)
                    if counter == 0:
                        try:
                            _data = set_nan_inf(data)
                            _data = _data.squeeze(2)
                            _data = torch.flip(_data, [1])
                            f_data = get_abs_price_dim(data)
                            _data = gauss_normalize(_data, dim=1)
                            __data = cube_normalize(_data)
                            #print(f_data.shape, _data.shape, __data.shape)
                            _data = torch.cat((_data.cpu(), __data.cpu(), f_data.cpu()), dim=2).to(device, dtype=dtype)
                            if torch.isinf(_data).any():
                                inf_mask = torch.isinf(_data)
                                _data[inf_mask] = 0
                                #print(_data)
                            if torch.isnan(_data).any():
                                nan_mask = torch.isnan(_data)
                                _data[nan_mask] = 0
                                #print(_data)
                            pred_v = v_model(_data, summary)
                            pred_gl = gl_model(_data.to('cuda:1'), summary.to('cuda:1'))
                            buy_price = data[:,-1,-2].to(device)
                                #print(buy_price.shape)
                        except Exception as e:
                        #    print(e)
                            break
                    counter += 1
                    if counter == (len(n_step_data_list)):
                        sell_price = data[:,-1,-2].to(device)
                profit = sell_price/buy_price
                #print(profit.shape)
                #profit = profit.unsqueeze(0)
                gl_mask = profit >= 1
                batch = profit.shape[0]
                gl = torch.zeros((batch, 2), dtype=torch.float32)
                gl[gl_mask] = torch.tensor([1, 0], dtype=torch.float32) # Profitable
                gl[~gl_mask] = torch.tensor([0, 1], dtype=torch.float32) # Not profitable
                #print(gl)
                gl = gl.to('cuda:1')
                #print(gl.shape)
                var = torch.abs(profit-1).unsqueeze(0)
                #print(var)
                #if steps%400 == 0:
                #    print(f'{pred_v[0]} {profit[0]} {pred_gl[0]}')
                #print(pred_v.shape, pred_gl.shape)
                loss_v = loss_fn_v(pred_v.to(dtype=torch.float32), var.to(dtype=torch.float32))
                loss_gl = loss_fn_gl(pred_gl, gl)
                if not torch.isnan(loss_v):
                    loss_v.backward()
                    torch.nn.utils.clip_grad_norm_(v_model.parameters(), grd_nrm)
                    optim_v.step()
                    scheduler_v.step()
                    optim_v.zero_grad()
                    running_loss_v += loss_v.item()
                if not torch.isnan(loss_gl):
                    loss_gl.backward()
                    torch.nn.utils.clip_grad_norm_(gl_model.parameters(), grd_nrm)
                    optim_gl.step()
                    scheduler_gl.step()
                    optim_gl.zero_grad()
                    running_loss_gl += loss_gl.item()
                steps += 1
                if steps % 1 == 0 and steps > 0:
                    a = f'\rRunning Epoch V Loss: {(running_loss_v/steps):.6f} Running Epoch GL Loss: {(running_loss_gl/steps):.6f} Epoch Steps: {steps}/{len(dataloader)} Step Loss V: {loss_v.item():.6f} Step Loss GL: {loss_gl.item():.6f}'
                    print(a, end='')
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
        wandb.log({'V_loss':(running_loss_v/steps), 'GL Loss':(running_loss_gl/steps)})
        torch.save(v_model, 'V_'+save_name+f'_E{epoch+1}')
        torch.save(gl_model, 'GL_'+save_name+f"E{epoch+1}")
        print(f'saved model ({steps})')

def Train_Direct_Predictor_gl(rank, gl_model, dataloader, epochs, save_name, lr, bs, num_bins, t_max, weight_decay, grd_nrm, epsilon=0.5, device='cuda', dtype=torch.float16):
    for param in gl_model.parameters():
        param = param.to(dtype)
    #print(dataloader, len(dataloader))
    optim_gl = torch.optim.AdamW(gl_model.parameters(), lr=lr)
    scheduler_gl = torch.optim.lr_scheduler.CosineAnnealingLR(optim_gl, T_max=t_max, eta_min=lr/10)
    loss_fn_gl = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss_gl = 0
        steps = 0
        print(f"\nEpoch {epoch+1}:")
        for i, n_step_data_list in enumerate(dataloader):
            if i > 0:
                counter = 0
                profit = 0
                for data in n_step_data_list:
                    data, summary = data[0], data[1]
                    data = data.squeeze(1, 3)
                    summary = summary.squeeze(1)
                   # print(data.shape)
                    if counter == 0:
                        #try:
                        _data = set_nan_inf(data)
                        _data = _data.squeeze(2)
                        _data = torch.flip(_data, [1])
                        f_data = get_abs_price_dim(data)
                        _data = gauss_normalize(_data, dim=1)
                        __data = cube_normalize(_data)
                        _data = torch.cat((_data.cpu(), __data.cpu(), f_data.cpu()), dim=2).to(device, dtype=dtype)
                        if torch.isinf(_data).any():
                            inf_mask = torch.isinf(_data)
                            _data[inf_mask] = 0
                        if torch.isnan(_data).any():
                            nan_mask = torch.isnan(_data)
                            _data[nan_mask] = 0
                        pred_gl = gl_model(_data.to(device), summary.to(device, dtype=dtype))
                        #print(pred_gl)
                        buy_price = data[:,-1,-2].to(device)
                        #except Exception as e:
                        #   print(e)
                        #    pass
                            #break
                    counter += 1
                    if counter == (len(n_step_data_list)):
                        sell_price = data[:,-1,-2].to(device)
                        #print(sell_price)
                profit = sell_price/buy_price
                gl_mask = profit >= 1
                batch = profit.shape[0]
                gl = torch.zeros((batch, 2), dtype=torch.float32)
                gl[gl_mask] = torch.tensor([1, 0], dtype=torch.float32) # Profitable
                gl[~gl_mask] = torch.tensor([0, 1], dtype=torch.float32) # Not profitable
                gl = gl.to(device)
                loss_gl = loss_fn_gl(pred_gl, gl)
                if not torch.isnan(loss_gl):
                    loss_gl.backward()
                    torch.nn.utils.clip_grad_norm_(gl_model.parameters(), grd_nrm)
                    optim_gl.step()
                    scheduler_gl.step()
                    optim_gl.zero_grad()
                    running_loss_gl += loss_gl.item()
                steps += 1
                if steps % 1 == 0 and steps > 0:
                    a = f'\rRunning Epoch GL Loss: {(running_loss_gl/steps):.6f} Epoch Steps: {steps}/{len(dataloader)}  Step Loss GL: {loss_gl.item():.6f}'
                    print(a, end='')
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
            wandb.log({'GL Loss':(running_loss_gl/steps)})
            torch.save(gl_model, save_name+f"E{epoch+1}")
            print(f'saved model ({steps})')

# From inference class
    def seq_pv_test(self):
        data = self.dataset.get_company_chunk_data(self.companies[28], self.dataset.iterable_dates[20])
        data, sum = data[0], data[1]
        data = data.squeeze(2)
        print(data.shape)
        new = get_abs_price_dim(data)
        print(new, new.shape)

class Ext_DirectPredSumTran(nn.Module):
    def __init__(self, seq_len=350, data_dim=5, num_bins=21, nhead=5, ff=15000, layers=72, sum_emb=76, scale=1, model_pth=None):
        super(Ext_DirectPredSumTran, self).__init__()
        self.base_model = torch.load(model_pth)
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()
        self.num_bins = num_bins
        self.seq_len = seq_len
        self.dim = data_dim
        self.pos_enc = PositionalEncoding(data_dim, seq_len, 'cuda')
        self.act_fn = nn.GELU
        self.scale = 1
        self.sum_emb = sum_emb
        self.seq_dim = self.seq_len*self.dim
        self.layer = nn.TransformerEncoderLayer(d_model=data_dim, nhead=nhead, dim_feedforward=ff, activation=self.act_fn(),
                                                batch_first=True, dropout=0.1)
        
        self.sum_inp = nn.Sequential(nn.Linear(sum_emb+self.seq_dim, self.seq_dim),
                                    self.act_fn(),
                                    nn.Linear(self.seq_dim, self.seq_dim),
                                    self.act_fn())
        self.tran_layer = nn.ModuleList([self.layer, self.sum_inp])
        self.tran = nn.ModuleList([self.tran_layer for i in range(layers)])
        self.linear = nn.Sequential(
            nn.Linear(seq_len*data_dim, self.scale*seq_len*data_dim),
            nn.GELU(),
            nn.Linear(self.scale*seq_len*data_dim, self.scale*seq_len*data_dim),
            nn.GELU(),
            nn.Linear(self.scale*seq_len*data_dim, 1),
            )
                      
    def forward(self, x, s):
        #print(x.shape, s.shape)
        batch_size = x.shape[0]
        #x = x + self.pos_enc(x)
        x = self.base_model.encode(x, s)
        #print(x.shape)
        for t, lin in self.tran:
            x = t(x)
            #print(x.shape)
            x = torch.cat((torch.reshape(x, (batch_size, self.seq_len*self.dim)), s), dim=1)
            #print(x.shape)
            x = lin(x)
            #print(x.shape)
            x = torch.reshape(x, (batch_size, self.seq_len, self.dim))
            #print(x.shape)
        x = torch.reshape(x, (batch_size, self.seq_dim))
        x = self.linear(x)
        return x

class NewsEmbeddingVAE(nn.Module):
    def __init__(self, act_fn=str, lat_act_fn=str, latent_dim=400):
        super(NewsEmbeddingVAE, self).__init__()
        if act_fn == 'g':
            self.act_fn = nn.GELU
        if act_fn == 'lr':
            self.act_fn = nn.LeakyReLU
        if lat_act_fn == 'g':
            self.lat_act_fn = nn.GELU
        if lat_act_fn == 'lr':
            self.lat_act_fn = nn.LeakyReLU
        if lat_act_fn == 's':
            self.lat_act_fn == nn.Sigmoid
        self.encoder = nn.Sequential(
            nn.Linear(768, 750),
            self.act_fn(),
            nn.Linear(750, 712),
            self.act_fn(),
            nn.Linear(712, 670),
            self.act_fn(),
            nn.Linear(670, 650),
            self.act_fn(),
            nn.Linear(650, 600),
            self.act_fn(),
            nn.Linear(600, 500),
            self.act_fn(),
            nn.Linear(500, 450),
            self.act_fn(),
            nn.Linear(450, latent_dim),
            self.lat_act_fn()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 450),
            self.act_fn(),
            nn.Linear(450, 500),
            self.act_fn(),
            nn.Linear(500, 600),
            self.act_fn(),
            nn.Linear(600, 650),
            self.act_fn(),
            nn.Linear(650, 670),
            self.act_fn(),
            nn.Linear(670, 712),
            self.act_fn(),
            nn.Linear(712, 750),
            self.act_fn(),
            nn.Linear(750, 768),
            self.act_fn()
        )
    
    def forward(self, emb):
        x = self.encoder(emb)
        return self.decoder(x)

class LatentNewsEncoder(nn.Module):
    # This processes the latent embeddings generated by the initial news embedding VAE
    def __init__(self, num_articles=4, d_output=768):
        super(LatentNewsEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.main_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.ff = nn.Linear(4*768, 768)
    
class PricesVAE(nn.Module):
    def __init__(self, seq_len):
        super(PricesVAE, self).__init__()
        self.seq_len = seq_len
        self.pos_enc = PositionalEncoding(5, seq_len, device='cuda')
        self.act_fn = nn.GELU
        dim = 5*seq_len
        self.encoder = nn.Sequential(
            # 5*400 = 2000
            nn.Linear(dim, int(dim*0.95)),
            self.act_fn(),
            nn.Linear(int(dim*0.95), int(dim*0.9)),
            self.act_fn(),
            nn.Linear(int(dim*0.9), int(dim*0.85)),
            self.act_fn(),
            nn.Linear(int(dim*0.85), int(dim*0.8)),
            self.act_fn(),
            nn.Linear(int(dim*0.8), int(dim*0.75)),
            self.act_fn(),
            nn.Linear(int(dim*0.75), int(dim*0.7)),
            self.act_fn(),
            nn.Linear(int(dim*0.7), int(dim*0.65)),
            self.act_fn(),
            nn.Linear(int(dim*0.65), int(dim*0.6)),
            self.act_fn(),
            nn.Linear(int(dim*0.6), int(dim*0.55)),
            self.act_fn(),
            nn.Linear(int(dim*0.55), int(dim*0.5)),
            self.act_fn(),
            nn.Linear(int(dim*0.5), int(dim*0.45)),
            self.act_fn(),
            nn.Linear(int(dim*0.45), int(dim*0.4)),
            self.act_fn(),
            #nn.Linear(int(dim*0.4), int(dim*0.35)),
            #self.act_fn()
            )
        self.decoder = nn.Sequential(
            #nn.Linear(int(dim*0.35), int(dim*0.4)),
            #self.act_fn(),
            nn.Linear(int(dim*0.4), int(dim*0.45)),
            self.act_fn(),
            nn.Linear(int(dim*0.45), int(dim*0.5)),
            self.act_fn(),
            nn.Linear(int(dim*0.5), int(dim*0.55)),
            self.act_fn(),
            nn.Linear(int(dim*0.55), int(dim*0.6)),
            self.act_fn(),
            nn.Linear(int(dim*0.6), int(dim*0.65)),
            self.act_fn(),
            nn.Linear(int(dim*0.65), int(dim*0.7)),
            self.act_fn(),
            nn.Linear(int(dim*0.7), int(dim*0.75)),
            self.act_fn(),
            nn.Linear(int(dim*0.75), int(dim*0.8)),
            self.act_fn(),
            nn.Linear(int(dim*0.8), int(dim*0.85)),
            self.act_fn(),
            nn.Linear(int(dim*0.85), int(dim*0.9)),
            self.act_fn(),
            nn.Linear(int(dim*0.9), int(dim*0.95)),
            self.act_fn(),
            nn.Linear(int(dim*0.95), dim),
            self.act_fn()
            )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x + self.pos_enc(x)
        x = torch.reshape(x, (batch_size, self.seq_len*5))
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.reshape(x, (batch_size, 400, 5))

class NestQTran2(nn.Module):
    def __init__(self, base_model, layers=10, f_dim=8000, device='cuda'):
        super(NestQTran2, self).__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False
            param = param.half()
        self.base_model.eval()
        self.dim = self.base_model.seq_len *self.base_model.dim
        self.f_dim = f_dim
        self.act_fn = nn.GELU
        self.network = nn.ModuleList([nn.Linear(self.dim, self.dim).half() for i in range(layers)]).to(device)
        self.in_layer = nn.Sequential(nn.Linear(self.dim, self.f_dim),
                                nn.GELU()).half().to(device)
        self.out_layer = nn.Sequential(nn.Linear(self.f_dim, 3*self.base_model.num_bins),
                                nn.GELU()).half().to(device)
    def forward(self, x, mask):
        x = self.base_model.encode(x)
        x = self.in_layer(x)
        for i, layer in self.network: 
            x = self.act_fn(layer(x))
        x = self.out_layer(x)
        x = torch.reshape(x, (3, self.base_model.num_bins))
        s = x[mask, :]
        return s

class GL_DirectPredSumTran(nn.Module):
    def __init__(self,seq_len=350, data_dim=5, num_bins=21, nhead=5, ff=15000, layers=72, sum_emb=76, scale=1):
        super(GL_DirectPredSumTran, self).__init__()
        self.num_bins = num_bins
        self.seq_len = seq_len
        self.dim = data_dim
        
        self.act_fn = nn.GELU
        self.scale = scale
        self.sum_emb = sum_emb
        self.seq_dim = self.seq_len*self.dim

        self.layers = nn.ModuleList([c_transformer_layer(static_dim=sum_emb, seq_dim=self.seq_dim, act_fn=self.act_fn,data_dim=self.dim, nhead=nhead, 
                                            dim_ff=ff, dropout=0.1) for i in range(layers)])  
        #print(self.layers[0].tran_layer)
        #print(sum(p.numel() for p in self.layers[0].parameters()))
        self.linear = nn.Sequential(
            nn.Linear(seq_len*data_dim, self.scale*seq_len*data_dim),
            nn.GELU(),
            nn.Linear(self.scale*seq_len*data_dim, 2),
            )
        #device = self.linear.device
        #print(device)
        self.pos_encoding = PositionalEncoding(data_dim, seq_len)
        self._encoding = nn.Parameter(self.pos_encoding.encoding, requires_grad=False)   

    def pos_encode(self, x):
        batch_size, seq_len, data_dim = x.size()
        return self._encoding[:seq_len, :]


    def forward(self, x, s):
        #print(x.shape, s.shape)
        #print(self.pos_enc.encoding.device, x.device)
        batch_size = x.shape[0]
        x = x + self.pos_encode(x)
        res = 0
        init_res = 0
        res_2 = 0
        res_3 = 0
        res_4 = 0
        res_5 = 0
        res_6 = 0
        res_7 = 0
        res_8 = 0
        res_9 = 0
        res_10 = 0
        res_11 = 0
        res_12 = 0
        res_13 = 0
        res_14 = 0
        res_15 = 0
        res_16 = 0
        res_17 = 0
        res_18 = 0
        res_19 = 0
        #res_20 = 0
        #res_21 = 0
        #res_22 = 0
        #res_23 = 0
        #res_24 = 0
        #res_25 = 0
        i = 0
        for layer in self.layers:
            n = i+3
            x = (layer(x, s) + (res + init_res + res_2 + res_3+res_4+res_5+res_6+res_7+res_8+res_9+res_10+
                                res_11+res_12+res_13+res_14))/(sqrt(n))
            res = x 
            if i == 0:
                init_res = x
            if i == 1:
                res_2 = x
            if i == 2:
                res_3 = x
            if i == 3:
                res_4 = x
            if i == 4:
                res_5 = x
            if i == 5:
                res_6 = x
            if i == 6:
                res_7 = x
            if i == 7:
                res_8 = x
            if i == 8:
                res_9 = x
            if i == 9:
                res_10 = x
            if i == 10:
                res_11 = x
            if i == 11:
                res_12 = x
            if i == 12:
                res_13 = x
            if i == 13:
                res_14 = x
            i += 1
        x = torch.reshape(x, (batch_size, self.seq_dim))
        x = self.linear(x)
        return x

class DirectPredSumTran(nn.Module):
    def __init__(self, seq_len=350, data_dim=5, num_bins=21, nhead=5, ff=15000, layers=72, sum_emb=76, scale=1):
        super(DirectPredSumTran, self).__init__()
        self.num_bins = num_bins
        self.seq_len = seq_len
        self.dim = data_dim
        self.pos_enc = PositionalEncoding(data_dim, seq_len, 'cuda')
        self.act_fn = nn.GELU
        self.scale = 1
        self.sum_emb = sum_emb
        self.seq_dim = self.seq_len*self.dim
        self.layer = nn.TransformerEncoderLayer(d_model=data_dim, nhead=nhead, dim_feedforward=ff, activation=self.act_fn(),
                                                batch_first=True, dropout=0.1)
        
        self.sum_inp = nn.Sequential(nn.Linear(sum_emb+self.seq_dim, self.seq_dim),
                                    self.act_fn(),
                                    nn.Linear(self.seq_dim, self.seq_dim),
                                    self.act_fn())
        self.tran_layer = nn.ModuleList([self.layer, self.sum_inp])
        self.tran = nn.ModuleList([self.tran_layer for i in range(layers)])
        self.linear = nn.Sequential(
            nn.Linear(seq_len*data_dim, self.scale*seq_len*data_dim),
            nn.GELU(),
            nn.Linear(self.scale*seq_len*data_dim, self.scale*seq_len*data_dim),
            nn.GELU(),
            nn.Linear(self.scale*seq_len*data_dim, 1),
            )
                      
    def forward(self, x, s):
        #print(x.shape, s.shape)
        batch_size = x.shape[0]
        #x = x + self.pos_enc(x)
        for t, lin in self.tran:
            x = t(x)
            #print(x.shape)
            x = torch.cat((torch.reshape(x, (batch_size, self.seq_len*self.dim)), s), dim=1)
            #print(x.shape)
            x = lin(x)
            #print(x.shape)
            x = torch.reshape(x, (batch_size, self.seq_len, self.dim))
            #print(x.shape)
        x = torch.reshape(x, (batch_size, self.seq_dim))
        x = self.linear(x)
        return x
    
    def encode(self, x, s):
        #print(x.shape, s.shape)
        batch_size = x.shape[0]
        #x = x + self.pos_enc(x)
        for t, lin in self.tran:
            x = t(x)
            #print(x.shape)
            x = torch.cat((torch.reshape(x, (batch_size, self.seq_len*self.dim)), s), dim=1)
            #print(x.shape)
            x = lin(x)
            #print(x.shape)
            x = torch.reshape(x, (batch_size, self.seq_len, self.dim))
            #print(x.shape)
        #x = torch.reshape(x, (batch_size, self.seq_dim))
        #x = self.linear(x)
        return x

class DirectPredTran(nn.Module):
    def __init__(self, seq_len=350, data_dim=5, num_bins=21, nhead=5, ff=15000, layers=72, sum_emb=77):
        super(DirectPredTran, self).__init__()
        self.num_bins = num_bins
        self.seq_len = seq_len
        self.dim = data_dim
        self.pos_enc = PositionalEncoding(data_dim, seq_len, 'cuda')
        self.act_fn = nn.GELU
        self.scale = 1
        self.layer = nn.TransformerEncoderLayer(d_model=data_dim, nhead=nhead, dim_feedforward=ff, activation=self.act_fn(),
                                                batch_first=True, dropout=0.15)
        self.network = nn.TransformerEncoder(self.layer, num_layers=layers)
        #self.network_2 = nn.TransformerEncoder(self.layer, num_layers=int(layers/2))
        #self.sum_inp = nn.Sequential(nn.Linear(sum_emb, 100),
        #                            self.act_fn())
        self.linear = nn.Sequential(
            nn.Linear(seq_len*data_dim, self.scale*seq_len*data_dim),
            nn.GELU(),
            nn.Linear(self.scale*seq_len*data_dim, self.scale*seq_len*data_dim),
            nn.GELU(),
            nn.Linear(seq_len*data_dim, self.scale*seq_len*data_dim),
            nn.GELU(),
            nn.Linear(self.scale*seq_len*data_dim, self.scale*seq_len*data_dim),
            nn.GELU(),
            nn.Linear(self.scale*seq_len*data_dim, 1),
            )
                      
    def forward(self, x):
        #print(x.shape)
        batch_size = x.shape[0]
        x = x + self.pos_enc(x)
        x = self.network(x)
        #(x.shape, x.flatten().shape)
        x = x.squeeze(0)
        x = self.linear(torch.reshape(x, (batch_size, self.seq_len*self.dim)))
        return x
      
class MixedLayeredDAE(nn.Module):
    def __init__(self, seq_len, emb_dim=76):
        super(MixedLayeredDAE, self).__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.pos_enc = PositionalEncoding(5, seq_len, device='cuda')
        self.act_fn = nn.GELU()
        #self.act_fn = nn.LeakyReLU()
        self.dim = 5*seq_len + emb_dim
        self.wid = [1, 0.97, 0.94, 0.92, 0.9]
        self.encoder = nn.ModuleList([nn.Linear(int(self.wid[i]*self.dim), int(self.wid[i+1]*self.dim)) for i in range(3)])
        self.decoder = nn.ModuleList([nn.Linear(int(self.wid[i+1]*self.dim), int(self.wid[i]*self.dim)) for i in reversed(range(3))])
    
    def forward(self, x, emb):
        # x ->(batch_size, seq_len, 5)
        # emb -> (batch_size, 768)
        batch_size = x.shape[0]
        x += self.pos_enc(x)
        x = torch.reshape(x, (batch_size, self.seq_len*5))
        emb = emb.squeeze(1)
        x = torch.cat((x, emb), dim=1)
        res = 0
        #print(x.shape)
        for layer in self.encoder:
            x = self.act_fn(layer(x))+res
            res = torch.mean(x)
        #print(x.shape)
        for layer in self.decoder:
            x = self.act_fn(layer(x))
        r_x = torch.reshape(x[:, :self.seq_len*5], (batch_size, self.seq_len, 5)).to(dtype=torch.float32)
        r_emb = torch.reshape(x[:, self.seq_len*5:], (batch_size, self.emb_dim)).to(dtype=torch.float32)
        return r_x, r_emb

    def encode(self, x, emb):
        # x ->(batch_size, seq_len, 5)
        # emb -> (batch_size, 768)
        print(emb.shape, x.shape)
        batch_size = x.shape[0]
        x += self.pos_enc(x)
        x = torch.reshape(x, (batch_size, self.seq_len*5))
        emb = emb.squeeze(1)
        x = torch.cat((x, emb), dim=1)
        for layer in self.encoder:
            x = self.act_fn(layer(x))
        return x
    
class PricesSumVAE(nn.Module):
    def __init__(self, seq_len):
        super(PricesSumVAE, self).__init__()
        self.seq_len = seq_len
        self.pos_enc = PositionalEncoding(5, seq_len, device='cuda')
        self.act_fn = nn.GELU
        dim = 5*seq_len +768
        self.dim = dim
        self.encoder = nn.Sequential(
            # 5*400 = 2000
            nn.Linear(dim, int(dim*0.975)),
            self.act_fn(),
            nn.Linear(int(dim*0.975), int(dim*0.95)),
            self.act_fn(),
            nn.Linear(int(dim*0.95), int(dim*0.925)),
            self.act_fn(),
            nn.Linear(int(dim*0.925), int(dim*0.9)),
            self.act_fn(),
            nn.Linear(int(dim*0.9), int(dim*0.875)),
            self.act_fn(),
            nn.Linear(int(dim*0.875), int(dim*0.85)),
            self.act_fn(),
            nn.Linear(int(dim*0.85), int(dim*0.825)),
            self.act_fn(),
            nn.Linear(int(dim*0.825), int(dim*0.8)),
            self.act_fn(),
            nn.Linear(int(dim*0.8), int(dim*0.775)),
            self.act_fn(),
            nn.Linear(int(dim*0.775), int(dim*0.75)),
            self.act_fn(),
            nn.Linear(int(dim*0.75), int(dim*0.725)),
            self.act_fn(),
            nn.Linear(int(dim*0.725), int(dim*0.7)),
            self.act_fn(),
            nn.Linear(int(dim*0.7), int(dim*0.675)),
            self.act_fn(),
            nn.Linear(int(dim*0.675), int(dim*0.65)),
            self.act_fn(),
            nn.Linear(int(dim*0.65), int(dim*0.625)),
            self.act_fn(),
            nn.Linear(int(dim*0.625), int(dim*0.6)),
            self.act_fn(),
            nn.Linear(int(dim*0.6), int(dim*0.575)),
            self.act_fn(),
            nn.Linear(int(dim*0.575), int(dim*0.55)),
            self.act_fn(),
            nn.Linear(int(dim*0.55), int(dim*0.525)),
            self.act_fn(),
            nn.Linear(int(dim*0.525), int(dim*0.5)),
            self.act_fn(),
            nn.Linear(int(dim*0.5), int(dim*0.475)),
            self.act_fn(),
            nn.Linear(int(dim*0.475), int(dim*0.45)),
            self.act_fn(),
            #nn.Linear(int(dim*0.45), int(dim*0.425)),
            #self.act_fn(),
            #nn.Linear(int(dim*0.425), int(dim*0.4)),
            #self.act_fn(),
            #nn.Linear(int(dim*0.4), int(dim*0.375)),
            #self.act_fn(),
            #nn.Linear(int(dim*0.375), int(dim*0.35)),
            #self.act_fn(),
            )
        self.decoder = nn.Sequential(
            #nn.Linear(int(dim*0.35), int(dim*0.375)),
            #self.act_fn(),
            #n.Linear(int(dim*0.375), int(dim*0.4)),
            #self.act_fn(),
            #nn.Linear(int(dim*0.4), int(dim*0.425)),
            #self.act_fn(),
            #nn.Linear(int(dim*0.425), int(dim*0.45)),
            #self.act_fn(),
            nn.Linear(int(dim*0.45), int(dim*0.475)),
            self.act_fn(),
            nn.Linear(int(dim*0.475), int(dim*0.5)),
            self.act_fn(),
            nn.Linear(int(dim*0.5), int(dim*0.525)),
            self.act_fn(),
            nn.Linear(int(dim*0.525), int(dim*0.55)),
            self.act_fn(),
            nn.Linear(int(dim*0.55), int(dim*0.575)),
            self.act_fn(),
            nn.Linear(int(dim*0.575), int(dim*0.6)),
            self.act_fn(),
            nn.Linear(int(dim*0.6), int(dim*0.625)),
            self.act_fn(),
            nn.Linear(int(dim*0.625), int(dim*0.65)),
            self.act_fn(),
            nn.Linear(int(dim*0.65), int(dim*0.675)),
            self.act_fn(),
            nn.Linear(int(dim*0.675), int(dim*0.7)),
            self.act_fn(),
            nn.Linear(int(dim*0.7), int(dim*0.725)),
            self.act_fn(),
            nn.Linear(int(dim*0.725), int(dim*0.75)),
            self.act_fn(),
            nn.Linear(int(dim*0.75), int(dim*0.775)),
            self.act_fn(),
            nn.Linear(int(dim*0.775), int(dim*0.8)),
            self.act_fn(),
            nn.Linear(int(dim*0.8), int(dim*0.825)),
            self.act_fn(),
            nn.Linear(int(dim*0.825), int(dim*0.85)),
            self.act_fn(),
            nn.Linear(int(dim*0.85), int(dim*0.875)),
            self.act_fn(),
            nn.Linear(int(dim*0.875), int(dim*0.9)),
            self.act_fn(),
            nn.Linear(int(dim*0.9), int(dim*0.925)),
            self.act_fn(),
            nn.Linear(int(dim*0.925), int(dim*0.95)),
            self.act_fn(),
            nn.Linear(int(dim*0.95), dim),
            self.act_fn()
            )

    def forward(self, x, emb):
        # x ->(batch_size, seq_len, 5)
        # emb -> (batch_size, 768)
        batch_size = x.shape[0]
        x += self.pos_enc(x)
        x = torch.reshape(x, (batch_size, self.seq_len*5))
        emb = emb.squeeze(1)
        x = torch.cat((x, emb), dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        r_x = torch.reshape(x[:, :self.seq_len*5], (batch_size, self.seq_len, 5)).to(dtype=torch.float32)
        r_emb = torch.reshape(x[:, self.seq_len*5:], (batch_size, 768)).to(dtype=torch.float32)
        return r_x, r_emb

    def encode(self, x, emb):
        batch_size = x.shape[0]
        x += self.pos_enc(x)
        x = torch.reshape(x, (batch_size, self.seq_len*5))
        emb = emb.squeeze(1)
        x = torch.cat((x, emb), dim=1)
        return self.encoder(x)

class EmbDAE(nn.Module):
    def __init__(self, dim):
        super(EmbDAE, self).__init__()
        self.act_fn = nn.GELU()
        self.w = [1, 0.92, 0.86, 0.78, 0.7, 0.62, 0.54, 0.46, 0.38, 0.33, 0.27, 0.23, 0.18, 0.14, 0.1]
        self.encoder = nn.ModuleList([nn.Linear(int(self.w[i]*dim), int(self.w[i+1]*dim)) for i in range(len(self.w)-1)])
        self.decoder = nn.ModuleList([nn.Linear(int(self.w[i+1]*dim), int(self.w[i]*dim)) for i in reversed(range(len(self.w)-1))])
    
    def forward(self, x):
        for layer in self.encoder:
            x = self.act_fn(layer(x))
        for layer in self.decoder:
            x = self.act_fn(layer(x))
        return x
    
    def encode(self, x):
        for layer in self.encoder:
            x = self.act_fn(layer(x))
        return x

class MarketDayDAE(nn.Module):
    def __init__(self, t_dim, num_t):
        super(MarketDayDAE, self).__init__()
        #self.pos_enc = PositionalEncoding(5, seq_len, device='cuda')
        self.act_fn = nn.GELU()
        self.t_dim = t_dim
        self.num_t = num_t
        dim = self.t_dim*self.num_t
        self.dim = dim
        self.w = [1, 0.92, 0.86, 0.78, 0.7, 0.62]
        self.encoder = nn.ModuleList([nn.Linear(int(self.w[i]*dim), int(self.w[i+1]*dim)) for i in range(len(self.w)-1)])
        self.decoder = nn.ModuleList([nn.Linear(int(self.w[i+1]*dim), int(self.w[i]*dim)) for i in reversed(range(len(self.w)-1))])

    def forward(self, x):
        #res = torch.zeros()
        for layer in self.encoder:
            x = self.act_fn(layer(x))
        for layer in self.decoder:
            x = self.act_fn(layer(x))
        return x

class Dist_DirectPredSumTran(nn.Module):
    def __init__(self,seq_len=350, data_dim=5, num_bins=21, num_days=5, nhead=5, ff=15000, layers=72, sum_emb=76, scale=1):
        super(Dist_DirectPredSumTran, self).__init__()
        self.num_bins = num_bins
        self.seq_len = seq_len
        self.dim = data_dim
        self.num_preds = num_days-1
        self.act_fn = nn.GELU
        self.scale = scale
        self.sum_emb = sum_emb
        self.seq_dim = self.seq_len*self.dim

        self.layers = nn.ModuleList([c_transformer_layer(static_dim=sum_emb, seq_dim=self.seq_dim, act_fn=self.act_fn,data_dim=self.dim, nhead=nhead, 
                                            dim_ff=ff, dropout=0.1) for i in range(layers)])  
        self.linear = nn.Sequential(
            nn.Linear(seq_len*data_dim, int(self.scale*seq_len*data_dim)),
            nn.GELU(),
            nn.Linear(int(self.scale*seq_len*data_dim), int(self.scale*seq_len*data_dim)),
            nn.GELU(),
            nn.Linear(int(self.scale*seq_len*data_dim), int(self.scale*seq_len*data_dim)),
            nn.GELU(),
            nn.Linear(int(self.scale*seq_len*data_dim), int(self.scale*seq_len*data_dim)),
            nn.GELU(),
            nn.Linear(int(self.scale*seq_len*data_dim), num_bins*self.num_preds),
            )
        self.pos_encoding = PositionalEncoding(data_dim, seq_len)
        self._encoding = nn.Parameter(self.pos_encoding.encoding, requires_grad=False)   

    def pos_encode(self, x):
        batch_size, seq_len, data_dim = x.size()
        return self._encoding[:seq_len, :]
    

    def encode(self, x, s):
        batch_size = x.shape[0]
        x = x + self.pos_encode(x)
        res = 0
        init_res = 0
        res_2 = 0
        res_3 = 0
        res_4 = 0
        res_5 = 0
        res_6 = 0
        res_7 = 0
        res_8 = 0
        res_9 = 0
        res_10 = 0
        res_11 = 0
        res_12 = 0
        res_13 = 0
        res_14 = 0
        i = 0
        for layer in self.layers:
            n = i+3
            x = (layer(x, s) + (res + init_res + res_2 + res_3+res_4+res_5+res_6+res_7+res_8+res_9+res_10+
                                res_11+res_12+res_13+res_14))/(sqrt(n))
            res = x 
            if i == 0:
                init_res = x
            if i == 1:
                res_2 = x
            if i == 2:
                res_3 = x
            if i == 3:
                res_4 = x
            if i == 4:
                res_5 = x
            if i == 5:
                res_6 = x
            if i == 6:
                res_7 = x
            if i == 7:
                res_8 = x
            if i == 8:
                res_9 = x
            if i == 9:
                res_10 = x
            if i == 10:
                res_11 = x
            if i == 11:
                res_12 = x
            if i == 12:
                res_13 = x
            if i == 13:
                res_14 = x
            i += 1
        transformer_activation = x
        x = torch.reshape(x, (batch_size, self.seq_dim))
        x = self.linear(x) # 1 160 4
        #print(x.shape)
        softmax = nn.Softmax(dim=1)
        x = softmax(x)
        linear_output = x.flatten()
        return (transformer_activation, linear_output) # Shape (200, 52), ()
    
    def forward(self, x, s):
        #print(x.shape, s.shape)
        #print(self.pos_enc.encoding.device, x.device)
        batch_size = x.shape[0]
        #print(x.shape, self.pos_encode(x).shape)
        x = x + self.pos_encode(x)
        res = 0
        init_res = 0
        res_2 = 0
        res_3 = 0
        res_4 = 0
        res_5 = 0
        res_6 = 0
        res_7 = 0
        res_8 = 0
        res_9 = 0
        res_10 = 0
        res_11 = 0
        res_12 = 0
        res_13 = 0
        res_14 = 0
        res_15 = 0
        res_16 = 0
        res_17 = 0
        res_18 = 0
        res_19 = 0
        #res_20 = 0
        #res_21 = 0
        #res_22 = 0
        #res_23 = 0
        #res_24 = 0
        #res_25 = 0
        i = 0
        for layer in self.layers:
            n = i+3
            x = (layer(x, s) + (res + init_res + res_2 + res_3+res_4+res_5+res_6+res_7+res_8+res_9+res_10+
                                res_11+res_12+res_13+res_14))/(sqrt(n))
            res = x 
            if i == 0:
                init_res = x
            if i == 1:
                res_2 = x
            if i == 2:
                res_3 = x
            if i == 3:
                res_4 = x
            if i == 4:
                res_5 = x
            if i == 5:
                res_6 = x
            if i == 6:
                res_7 = x
            if i == 7:
                res_8 = x
            if i == 8:
                res_9 = x
            if i == 9:
                res_10 = x
            if i == 10:
                res_11 = x
            if i == 11:
                res_12 = x
            if i == 12:
                res_13 = x
            if i == 13:
                res_14 = x
            i += 1
        x = torch.reshape(x, (batch_size, self.seq_dim))
        x = self.linear(x)
        x = torch.reshape(x, (batch_size, self.num_bins, self.num_preds))
        return x

class Layer_Dist_DirectPredSumTran(nn.Module):
    def __init__(self,seq_len=350, data_dim=5, num_bins=21, num_days=5, nhead=5, ff=15000, layers=72, sum_emb=76, scale=1,num_lin_layers=8):
        super(Layer_Dist_DirectPredSumTran, self).__init__()
        self.num_lin_layers = num_lin_layers
        self.num_bins = num_bins
        self.seq_len = seq_len*2
        self.dim = data_dim
        #self.dim = data_dim*2
        self.num_preds = num_days-1
        self.act_fn = nn.GELU
        self.act = nn.GELU()
        self.scale = scale
        self.sum_emb = sum_emb
        self.seq_dim = self.seq_len*self.dim
        self.layer_act_dim = 160*4
        self.layers = nn.ModuleList([c_transformer_layer(static_dim=self.layer_act_dim, seq_dim=self.seq_dim, act_fn=self.act_fn,data_dim=self.dim, nhead=nhead, 
                                            dim_ff=ff, dropout=0.1) for i in range(layers)])  
        self.summary_module_dim = 300
        self.sum_scale = 3
        self.linear_in = nn.Sequential(
            nn.Linear(self.seq_len*self.dim+self.summary_module_dim+4*160, int(self.scale*self.seq_len*data_dim)),
            nn.GELU(),
            )
        self.linear_layers = nn.ModuleList([nn.Linear(int(self.scale*self.seq_len*data_dim), int(self.scale*self.seq_len*data_dim)) for i in range(self.num_lin_layers)]) 
        self.linear_out = nn.Linear(int(self.scale*self.seq_len*data_dim), num_bins*self.num_preds)
        self.summary_module = nn.Sequential(
            nn.Linear(sum_emb, int(self.sum_scale*sum_emb)),
            nn.GELU(),
            nn.Linear(int(self.sum_scale*sum_emb), int(self.sum_scale*sum_emb)),
            nn.GELU(),
            nn.Linear(int(self.sum_scale*sum_emb), int(self.sum_scale*sum_emb)),
            nn.GELU(),
            nn.Linear(int(self.sum_scale*sum_emb), self.summary_module_dim),
            nn.GELU()
        )
        self.pos_encoding = PositionalEncoding(self.dim, int(self.seq_len/2))
        self._encoding = nn.Parameter(self.pos_encoding.encoding, requires_grad=False)   

    def pos_encode(self, x):
        batch_size, seq_len, data_dim = x.size()
        return self._encoding[:seq_len, :]

    def forward(self, x, s, a_lin):
        # x -> previous 200 days
        # s -> summary embedding
        # a_seq -> first 200 days from previous model
        # a_lin -> flattened linear activation from  previous model
        #print(x.shape, s.shape)
        #print(self.pos_enc.encoding.device, x.device)
        batch_size = x.shape[0]
        #x = x + self.pos_encode(x)
        #print(x[:,200:,:].shape, self.pos_encode(x[:,200:,:]).shape)
        x[:,200:,:] = x[:,200:,:] + self.pos_encode(x[:,200:,:])
        i = 0
        #res = 0
        for layer in self.layers:
            x = layer(x, a_lin)
        x = torch.reshape(x, (batch_size, self.seq_dim))
        s = self.summary_module(s.squeeze(1))
        #print(x.shape, s.shape)
        x = torch.cat((x, s, a_lin), dim=-1)
        x = self.linear_in(x)
        #print(x, type(x))
        l_res = 0
        l_res_0 = 0
        i = 0
        for layer in self.linear_layers:
            x = layer(x) + l_res + l_res_0
            x = self.act(x)
            l_res = x
            if i == 0:
                l_res_0 = x
                i += 1
        x = self.linear_out(x)
        x = torch.reshape(x, (batch_size, self.num_bins, self.num_preds))
        return x
    

class V_DirectPredSumTran(nn.Module):
    def __init__(self, seq_len=350, data_dim=5, num_bins=21, nhead=5, ff=15000, layers=72, sum_emb=76, scale=1):
        super(V_DirectPredSumTran, self).__init__()
        self.num_bins = num_bins
        self.seq_len = seq_len
        self.dim = data_dim
        self.pos_enc = PositionalEncoding(data_dim, seq_len, 'cuda')
        self.act_fn = nn.GELU
        self.scale = 1
        self.sum_emb = sum_emb
        self.seq_dim = self.seq_len*self.dim
        self.layer = nn.TransformerEncoderLayer(d_model=data_dim, nhead=nhead, dim_feedforward=ff, activation=self.act_fn(),
                                                batch_first=True, dropout=0.1)
        
        self.sum_inp = nn.Sequential(nn.Linear(sum_emb+self.seq_dim, self.seq_dim),
                                    self.act_fn(),
                                    nn.Linear(self.seq_dim, self.seq_dim),
                                    self.act_fn())
        self.tran_layer = nn.ModuleList([self.layer, self.sum_inp])
        self.tran = nn.ModuleList([self.tran_layer for i in range(layers)])
        self.linear = nn.Sequential(
            nn.Linear(seq_len*data_dim, self.scale*seq_len*data_dim),
            nn.GELU(),
            nn.Linear(self.scale*seq_len*data_dim, self.scale*seq_len*data_dim),
            nn.GELU(),
            nn.Linear(self.scale*seq_len*data_dim, 1),
            )
                      
    def forward(self, x, s):
        #print(x.shape, s.shape)
        batch_size = x.shape[0]
        x = x + self.pos_enc(x)
        for t, lin in self.tran:
            x = t(x)
            #print(x.shape)
            x = torch.cat((torch.reshape(x, (batch_size, self.seq_len*self.dim)), s), dim=1)
            #print(x.shape)
            x = lin(x)
            #print(x.shape)
            x = torch.reshape(x, (batch_size, self.seq_len, self.dim))
            #print(x.shape)
        x = torch.reshape(x, (batch_size, self.seq_dim))
        x = self.linear(x)
        return x
    
    def encode(self, x, s):
        #print(x.shape, s.shape)
        batch_size = x.shape[0]
        #x = x + self.pos_enc(x)
        for t, lin in self.tran:
            x = t(x)
            #print(x.shape)
            x = torch.cat((torch.reshape(x, (batch_size, self.seq_len*self.dim)), s), dim=1)
            #print(x.shape)
            x = lin(x)
            #print(x.shape)
            x = torch.reshape(x, (batch_size, self.seq_len, self.dim))
            #print(x.shape)
        #x = torch.reshape(x, (batch_size, self.seq_dim))
        #x = self.linear(x)
        return x