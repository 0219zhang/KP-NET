
import os
import copy
import math

import torch
from torch.nn import init
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd
import argparse
from sklearn.model_selection import ParameterGrid

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import data_preprocess as dp


class SparseLinear(nn.Module):
    """
    adj: input * output
    """
    def __init__(self, adj, bias):
        super(SparseLinear, self).__init__()
        self.weight = Parameter(torch.empty(adj.T.shape))
        if bias:
            self.bias = Parameter(torch.empty(self.weight.size(0)))
        else:
            self.register_parameter('bias', None)
        self.adj = adj
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        if not isinstance(self.adj, (torch.Tensor)):
            self.adj = torch.FloatTensor(self.adj)

    def forward(self, input):
        if self.adj.device != self.weight.device:
            self.adj = self.adj.to(self.weight.device)
        return F.linear(input, self.weight * self.adj.T, self.bias)


class ClinicalBenefitCLF(nn.Module):

    def __init__(self, maps, drop_p):
        super(ClinicalBenefitCLF, self).__init__()
        # embeding
        clfs = []
        for map in maps:
            clfs += [
                nn.Dropout(p=drop_p),
                SparseLinear(adj=map, bias=True),
                nn.BatchNorm1d(map.size(1)),
                nn.ReLU(True)
            ]
        # classifier
        clfs += [
            nn.Linear(maps[-1].size(1), 2),
            nn.Softmax(dim=1)
        ]
        self.clf = nn.Sequential(*clfs)

    def forward(self, x):
        return self.clf(x)


class MlpDataSet(Dataset):

    def __init__(self, x, y):
        super(MlpDataSet, self).__init__()
        self.x, self.y = x, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        dat = self.x[index, :]
        tar = self.y[index]
        return dat, tar


def train_model(
    model, dataloader, epochs, early_stop, learn_rate, weight_decay,
    lr_step_size, lr_gamma, device, writer
):
    model_wt = copy.deepcopy(model.state_dict())
    best_auc = .0
    no_improve = 0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learn_rate, weight_decay=weight_decay
    )
    optimizer_step = StepLR(
        optimizer=optimizer, step_size=lr_step_size, gamma=lr_gamma
    )

    for epoch in tqdm(range(epochs), leave=False):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = .0
            y_true, y_pred = [], []
            with torch.set_grad_enabled(phase == 'train'):
                for x, y in dataloader[phase]:
                    x, y = x.to(device), y.to(device)
                    output = model(x)
                    loss = criterion(output, y)
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    epoch_loss += loss.item() * len(y)
                    y_true.append(y)
                    y_pred.append(output)

            with torch.no_grad():
                y_true = torch.cat(y_true, dim=0).cpu().numpy()
                y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
                auc = roc_auc_score(y_true, y_pred[:, 1])

                writer.add_scalar('AUC/'+phase, auc, epoch)
                writer.add_scalar(
                    'Loss/'+phase, epoch_loss/len(y_true), epoch
                )

                if phase == 'valid':
                    if auc > best_auc:
                        best_auc = auc
                        model_wt = copy.deepcopy(model.state_dict())
                        no_improve = 0
                    else:
                        no_improve += 1
        optimizer_step.step()
        if no_improve == early_stop:
            break
    with torch.no_grad():
        model.load_state_dict(model_wt)
        model.eval()
        test_true, test_pred = [], []
        for x, y in dataloader['test']:
            x = x.to(device)
            output = model(x)
            test_true.append(y)
            test_pred.append(output)
        test_true = torch.cat(test_true, dim=0).numpy()
        test_pred = torch.cat(test_pred, dim=0).cpu().numpy()
        test_auc = roc_auc_score(test_true, test_pred[:, 1])
        test_result = pd.DataFrame({
            'ytrue': test_true, 'ypred': test_pred[:, 1]
        })

        if 'exter' in dataloader.keys():
            exter_true, exter_pred = [], []
            for x, y in dataloader['exter']:
                x = x.to(device)
                output = model(x)
                exter_true.append(y)
                exter_pred.append(output)
            exter_true = torch.cat(exter_true, dim=0).numpy()
            exter_pred = torch.cat(exter_pred, dim=0).cpu().numpy()
            exter_auc = roc_auc_score(exter_true, exter_pred[:, 1])
            exter_result = pd.DataFrame({
                'ytrue': exter_true, 'ypred': exter_pred[:, 1]
            })
            return model_wt, test_auc, exter_auc, test_result, exter_result
        else:
            return model_wt, test_auc, test_result


def main(mutation_ratio, cna_ratio, dirname):
    # file path
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--drop_p', default=[0.1, 0.2, 0.3])
    parser.add_argument('--learn_rate', type=list, default=[0.01])
    parser.add_argument('--epochs', type=list, default=[300])
    parser.add_argument('--early_stop', type=list, default=[50])
    parser.add_argument('--weight_decay', type=list, default=[1e-5])
    parser.add_argument('--lr_step_size', type=list, default=[20, 30])
    parser.add_argument('--lr_gamma', type=list, default=[0.95, 0.9])
    parser.add_argument('--bs', type=list, default=[8])
    args = parser.parse_args()

    # Grid research
    param_grid = list(ParameterGrid({
        'drop_p': args.drop_p,
        'learn_rate': args.learn_rate,
        'epochs': args.epochs,
        'early_stop': args.early_stop,
        'weight_decay': args.weight_decay,
        'lr_step_size': args.lr_step_size,
        'lr_gamma': args.lr_gamma,
        'bs': args.bs
    }))
    pd.DataFrame(param_grid).to_csv(
        dirname+'/param_grid.csv', index=None
    )

    # torch.dtype
    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda:1')

    # read data and pheno
    DiscX, Discy, ValidX, Validy, maps = dp.data_preprocess(
        mutation_ratio=mutation_ratio, cna_ratio=cna_ratio,
        dirname=dirname
    )

    # Exter Data
    ValidX, Validy = torch.FloatTensor(ValidX.values), torch.LongTensor(Validy)
    for n in range(len(maps)):
        maps[n] = torch.FloatTensor(maps[n].values).to(device)

    # split data
    trainx, testx, trainy, testy = train_test_split(
        DiscX, Discy, test_size=0.2, random_state=219, shuffle=True,
        stratify=Discy
    )
    trainx, validx, trainy, validy = train_test_split(
        trainx, trainy, test_size=len(testy), random_state=219, shuffle=True,
        stratify=trainy
    )

    trainx, trainy = torch.FloatTensor(trainx.values), torch.LongTensor(trainy)
    validx, validy = torch.FloatTensor(validx.values), torch.LongTensor(validy)
    testx, testy = torch.FloatTensor(testx.values), torch.LongTensor(testy)

    test_aucs, exter_aucs = [], []
    best_test_auc = .0
    for epo, params in tqdm(enumerate(param_grid), total=len(param_grid)):
        drop_p = params['drop_p']
        learn_rate = params['learn_rate']
        epochs = params['epochs']
        early_stop = params['early_stop']
        weight_decay = params['weight_decay']
        lr_step_size = params['lr_step_size']
        lr_gamma = params['lr_gamma']
        bs = params['bs']

        # dataloader
        dataloader = {
            'valid': DataLoader(MlpDataSet(validx, validy), batch_size=bs),
            'test': DataLoader(MlpDataSet(testx, testy), batch_size=bs),
            'exter': DataLoader(MlpDataSet(ValidX, Validy))
        }
        if trainx.size(0) % bs == 1:
            dataloader['train'] = DataLoader(
                MlpDataSet(trainx, trainy), batch_size=bs, shuffle=True,
                drop_last=True
            )
        else:
            dataloader['train'] = DataLoader(
                MlpDataSet(trainx, trainy), batch_size=bs, shuffle=True
            )

        # tensorboard path
        wr_path = './path/'+dirname+'/param'+str(epo)
        writer = SummaryWriter(wr_path)

        # set seed
        torch.manual_seed(219)
        # model
        model = ClinicalBenefitCLF(maps, drop_p=drop_p).to(device)

        # trainmodel and return result
        wt, test_auc, exter_auc, *_ = train_model(
            model=model, dataloader=dataloader, epochs=epochs,
            early_stop=early_stop, learn_rate=learn_rate,
            weight_decay=weight_decay, lr_step_size=lr_step_size,
            lr_gamma=lr_gamma, device=device, writer=writer
        )
        if test_auc > best_test_auc:
            torch.save(wt, dirname+'/best_wt.pt')
            best_test_auc = test_auc

        test_aucs.append(test_auc)
        exter_aucs.append(exter_auc)

    result = pd.DataFrame({
        'param': range(len(test_aucs)), 'test': test_aucs, 'exter': exter_aucs
    })
    result.to_csv(dirname+'/AUC.csv', index=None)
    data = pd.concat([result, pd.DataFrame(param_grid)], axis=1)
    data = data.sort_values(by='test', ascending=True)
    data.to_csv(dirname+'/Result.csv', index=None)


if __name__ == '__main__':
    mutation_ratio = 0.05
    cna_ratio = 0.1
    dirnames = 'result'
    main(
        mutation_ratio=mutation_ratio, cna_ratio=cna_ratio, dirname=dirnames
    )
