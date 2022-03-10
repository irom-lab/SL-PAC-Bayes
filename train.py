import torch
import numpy as np
from model import NSModel, SModel
import warnings
warnings.filterwarnings('ignore')
from utils import *
from copy import deepcopy
from torchvision import datasets
from torchvision.transforms import ToTensor

N = 30000
delta = 0.009
deltap = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = torch.nn.CrossEntropyLoss()
num_epochs = 100
num_evaluations = 1000

train_data = datasets.MNIST(root = 'data', train = True, transform = ToTensor(), download = True)
test_data = datasets.MNIST(root = 'data', train = False, transform = ToTensor())
prior_data, post_data = torch.utils.data.random_split(train_data, [train_data.data.size()[0] - N, N])

loaders = {'prior' : torch.utils.data.DataLoader(prior_data, batch_size=1000, shuffle=True),
           'post' : torch.utils.data.DataLoader(post_data, batch_size=1000, shuffle=True),
           'test'  : torch.utils.data.DataLoader(test_data, batch_size=test_data.data.size()[0], shuffle=True)}



prior = SModel()
prior.init_logvar(-10)
optimizer = torch.optim.Adam(prior.parameters(), lr = 0.01)

print("Trainig prior")
for epoch in range(num_epochs):
    for x, y in loaders['prior']:
        x, y = x.to(device), y.to(device)
        prior.init_xi()
        output = prior(x)

        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Batch loss =',float(loss), "           ", end='\r')
print()

prior.init_logvar(-5)
model = SModel()
model.load_state_dict(deepcopy(prior.state_dict()))
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

print("Trainig posterior")
for epoch in range(num_epochs):
    for x, y in loaders['post']:
        x, y = x.to(device), y.to(device)
        model.init_xi()
        output = model(x)

        reg = PAC_Bayes_regularizer(model, prior, N, delta, device)
        loss = criterion(output, y) + torch.sqrt(reg/2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Batch loss =',float(loss), "           ", end='\r')
print()


print("Evaluating Bound")
error_rates = []
with torch.no_grad():
    for _ in range(num_evaluations):
        model.init_xi()
        for x, y in loaders['test']:
            output = model(x)
            pred_y = torch.max(output, dim=1)[1].data.squeeze()
            error_rate = 1 - (pred_y == y).sum().item() / float(y.size(0))
            error_rates.append(float(error_rate))


# Computing sample convergence bound
avg_error_rate = np.mean(error_rates)
sample_convergence_reg = np.log(2/deltap)/num_evaluations
error_rate_bound = kl_inv_l(avg_error_rate, sample_convergence_reg) if avg_error_rate < 1 else 1
print("Bound on the expected error rate for networks sampled from posterior:", error_rate_bound)

# Computing kl-inverse PAC-Bayes bound
reg = float((model.calc_kl_div(prior, device) + np.log(2*np.sqrt(N)/delta)) / N)
pac_bayes_bound = kl_inv_l(error_rate_bound, reg) if error_rate_bound < 1 else 1
print("PAC-Bayes guarantee on error rate for new samples", pac_bayes_bound)
