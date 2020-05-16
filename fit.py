import torch
from torch import nn, optim
from torch.utils import data
from utils import construct_params
#    return torch.exp(args[0] * x**2+args[1]*x+args[2])



def fit(test_func, length, train_x, train_y, test_x, test_y):

    allparams = construct_params(length)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class Solver(nn.Module):
        def __init__(self):
            super().__init__()
            for param in allparams:
                setattr(self, param, torch.nn.Parameter(torch.tensor(1.0)))

        def forward(self, x):
            cur_params = []
            for param in allparams:
                cur_params.append(getattr(self,param))
            return test_func(x, *cur_params)
            #return torch.cos( self.a * x[:, 0]) + torch.exp(self.b * x[:, 1])

    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    #print(train_x.device)
    #print(train_y.device)
    train_dataset = data.TensorDataset(train_x, train_y)
    train_loader = data.DataLoader(train_dataset, batch_size = 16, shuffle = True)
    #train_loader.to(device)
    #print(train_X)
    #print(train_Y)


    test_x = torch.from_numpy(test_x).to(device)
    test_y = torch.from_numpy(test_y).to(device)
    #test_x.to(device)
    #test_y.to(device)
    #test_dataset = data.TensorDataset(test_x, test_y)
    #test_loader = data.DataLoader(test_dataset, batch_size = 16, shuffle = True)


    model = Solver()
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    #test_result = model.forward(test_X)
    #print(test_result)
    #print("----------------------")
    #opt.zero_grad()
    #result = model(test_X)
    #print(result)

    for epoch in range(100):
        model.train()
        for inp, out in train_loader:
            opt.zero_grad()
            pred_out = model(inp)
            loss = loss_fn(pred_out, out)
            loss.backward()
            opt.step()
        if torch.isnan(loss) or loss.item() < 1e-2:
           break
        #print(f'Epoch: {epoch}, Loss: {loss}')

    model.eval()
    y_pred = model(test_x)
    error = loss_fn(y_pred, test_y)
    error =error.item()
    #print(error)

    params_ret = []
    for param in allparams:
        #print(param)
        param_val = getattr(model, param).item()
        #print(param_val)
        params_ret.append(param_val)


    return params_ret, error
    #print(f'Final a:{model.a}, b: {model.b}')


#train_x, train_y = generate_data(200, test_func1)
#test_x, test_y = generate_data(200, test_func1)

#params, error = fit(test_func, 3, train_x, train_y, test_x, test_y)

#print(params)
#print(error)



