
import math

import torch
from torch.nn import init
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import pandas as pd
from captum.attr import DeepLift
from captum.attr import LayerConductance

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


def deeplift_fun():
    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda:1')

    # 读取测试集数据及maps
    *_, ValidX, Validy, maps = dp.data_preprocess(
        mutation_ratio=0.05, cna_ratio=0.1, dirname='DeepLIFT'
    )

    ValidX = torch.FloatTensor(ValidX.values).to(device)
    Validy = torch.LongTensor(Validy).to(device)
    model_maps = []
    for map in maps:
        map = torch.FloatTensor(map.values)
        model_maps.append(map)

    # 初始化模型, 并加载训练好的模型参数
    model = ClinicalBenefitCLF(maps=model_maps, drop_p=0.2).to(device)
    wt = torch.load('result/best_wt.pt')
    model.load_state_dict(wt)
    model.eval()

    # 使用整合梯度算法计算变量重要性, 初始化IG模型, 构建评估对象
    deeplift = DeepLift(model)
    ValidX = ValidX.requires_grad_().to(device)

    # 计算输入节点的VIM
    input_attr = deeplift.attribute(
        ValidX, target=1, return_convergence_delta=False
    )
    input_attr = pd.DataFrame(input_attr.detach().cpu().numpy().T)

    input_data = pd.read_csv('DeepLIFT/input_type.csv')
    input_data.loc[:, 'node'] = [
        'N'+str(x) for x in range(input_data.shape[0])
    ]
    input_data = pd.concat([input_data, input_attr], axis=1)
    input_data
    input_data.to_csv('DeepLIFT/input_vim.csv', index=None)

    input2gene = model_maps[0] * wt['clf.1.weight'].cpu().T
    input2gene = pd.DataFrame(
        input2gene.numpy(),
        index=['i'+str(x) for x in range(input2gene.shape[0])],
        columns=['g'+str(x) for x in range(input2gene.shape[1])]
    )
    input2gene
    input2gene.to_csv('DeepLIFT/input2gene_weight.csv')

    # 计算hidden0(基因层)的节点重要性
    cond_gene = LayerConductance(model, model.clf[1])
    gene_value = cond_gene.attribute(ValidX, target=1).detach().cpu().numpy()
    gene_value = pd.DataFrame(gene_value.T)
    gene_name = maps[0].columns.values
    gene_name = pd.DataFrame({
        'gene': gene_name, 'Node': ['N'+str(x) for x in range(len(gene_name))]
    })
    gene_data = pd.concat([gene_name, gene_value], axis=1)
    gene_data
    gene_data.to_csv('DeepLIFT/gene_vim.csv', index=None)

    gene2pathc = model_maps[1] * wt['clf.5.weight'].cpu().T
    gene2pathc = pd.DataFrame(
        gene2pathc.numpy(),
        index=['g'+str(x) for x in range(gene2pathc.shape[0])],
        columns=['pc'+str(x) for x in range(gene2pathc.shape[1])]
    )
    gene2pathc
    gene2pathc.to_csv('DeepLIFT/gene2pathc_weigth.csv')

    # 计算hidden1(PathwayC)的节点重要性 [5]
    cond_pathc = LayerConductance(model, model.clf[5])
    pathc_value = cond_pathc.attribute(ValidX, target=1).detach().cpu().numpy()
    pathc_value = pd.DataFrame(pathc_value.T)

    pathc_name = maps[1].columns.values
    pathc_name = pd.DataFrame({
        'pathc': pathc_name,
        'Node': ['N'+str(x) for x in range(len(pathc_name))]
    })
    pathc_data = pd.concat([pathc_name, pathc_value], axis=1)
    pathc_data
    pathc_data.to_csv('DeepLIFT/pathwayC_vim.csv', index=None)

    pathc2pathb = model_maps[2] * wt['clf.9.weight'].cpu().T
    pathc2pathb = pd.DataFrame(
        pathc2pathb.numpy(),
        index=['pc'+str(x) for x in range(pathc2pathb.shape[0])],
        columns=['pb'+str(x) for x in range(pathc2pathb.shape[1])]
    )
    pathc2pathb
    pathc2pathb.to_csv('DeepLIFT/pathc2pathb_weigth.csv')

    # 计算hidden2(PathwayB)的节点重要性 [9]
    cond_pathb = LayerConductance(model, model.clf[9])
    pathb_value = cond_pathb.attribute(ValidX, target=1).detach().cpu().numpy()
    pathb_value = pd.DataFrame(pathb_value.T)

    pathb_name = maps[2].columns.values
    pathb_name = pd.DataFrame({
        'pathb': pathb_name,
        'Node': ['N'+str(x) for x in range(len(pathb_name))]
    })
    pathb_data = pd.concat([pathb_name, pathb_value], axis=1)
    pathb_data
    pathb_data.to_csv('DeepLIFT/pathwayB_vim.csv', index=None)

    pathb2patha = model_maps[3] * wt['clf.13.weight'].cpu().T
    pathb2patha = pd.DataFrame(
        pathb2patha.numpy(),
        index=['pb'+str(x) for x in range(pathb2patha.shape[0])],
        columns=['pa'+str(x) for x in range(pathb2patha.shape[1])]
    )
    pathb2patha
    pathb2patha.to_csv('DeepLIFT/pathb2patha_weigth.csv')

    # 计算hidden3(PathwayA)的节点重要性 [13]
    cond_patha = LayerConductance(model, model.clf[13])
    patha_value = cond_patha.attribute(ValidX, target=1).detach().cpu().numpy()
    patha_value = pd.DataFrame(patha_value.T)

    patha_name = maps[3].columns.values
    patha_name = pd.DataFrame({
        'patha': patha_name,
        'Node': ['N'+str(x) for x in range(len(patha_name))]
    })
    patha_data = pd.concat([patha_name, patha_value], axis=1)
    patha_data
    patha_data.to_csv('DeepLIFT/pathwayA_vim.csv', index=None)

    patha2output = wt['clf.16.weight'].T.cpu().numpy()
    patha2output = pd.DataFrame(
        patha2output,
        index=['pa'+str(x) for x in range(patha2output.shape[0])],
        columns=['o1', 'o2']
    )
    patha2output
    patha2output.to_csv('DeepLIFT/patha2output.csv')


if __name__ == '__main__':
    deeplift_fun()
