
import os

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

from sklearn.model_selection import train_test_split


def annotate_cna_gene(data):
    '''
    data的列名是ID, chrom, loc.start, loc.end, num.mark, seg.mean
    '''
    r_script = '''
    anno <- function(data) {
        library(CNTools)
        cnseg <- CNSeg(data)
        data("geneInfo")
        rdByGene <- getRS(cnseg, by = "gene", imput = FALSE, XY = T,
                        geneMap = geneInfo, what = "median")
        cna_annotate <- rdByGene@rs
        return(cna_annotate)
    }
    '''
    ro.r(r_script)

    pandas2ri.activate()
    anno_data_r = ro.r['anno'](data)
    anno_data_py = pandas2ri.ri2py(anno_data_r)
    pandas2ri.deactivate()
    return anno_data_py


def drop_duplicated(x):
    x0, x1 = x.min(), x.max()
    if all([x0 < -0.5, x1 > 0.5]):
        x = [0] * len(x)
    if all([x0 > -0.5, x1 < 0.5]):
        x = [0] * len(x)
    if x0 < -0.5:
        x = [x0] * len(x)
    if x1 > 0.5:
        x = [x1] * len(x)
    return x


def pre_data():
    clinical1 = pd.read_csv('melanoma_profile/clinical1.csv')
    clinical1 = clinical1.loc[clinical1.cancer_type.values == 'Melanoma', :]
    clinical1 = clinical1.loc[clinical1.time.values == 'pre-treatment', :]
    ids = []
    for n in clinical1.individual_id.values:
        if 'MEL-IPI_' in n:
            n = n.replace('MEL-IPI_', '')
            ids.append(n)
        else:
            ids.append(n)
    clinical1.individual_id = ids

    clinical = pd.read_csv('melanoma_profile/clinical2.csv')
    clinical = clinical.loc[clinical.cancer_type.values == 'Melanoma', :]

    clinical = clinical.loc[
        np.in1d(clinical.patient_id.values, clinical1.individual_id.values), :
    ]
    clinical = clinical.loc[clinical.RECIST.values != 'X', :]
    clinical = clinical.loc[clinical.drug_type.values == 'anti-CTLA-4', :]

    mutation = pd.read_table('melanoma_profile/Mutation.txt', sep='\t')
    mutation = mutation.loc[mutation.Variant_Type.values == 'SNP', :]
    mutation = mutation.loc[
        mutation.Variant_Classification.values != 'Silent', :
    ]

    # 将突变数据整理成标准数据格式
    patients, genes = clinical.pair_id.values, mutation.Hugo_Symbol.unique()
    muta = pd.DataFrame(
        np.zeros((len(patients), len(genes))), patients, genes
    )
    for n in patients:
        gene = mutation.Hugo_Symbol.values[mutation.pair_id.values == n]
        muta.loc[n, gene] = 1

    # 读取CNA数据
    cna = pd.read_csv('melanoma_profile/CNA.csv')
    cna = cna.loc[
        np.in1d(cna.loc[:, 'sample'].values, clinical.pair_id.values), :
    ]
    cna = cna.loc[cna.total_copy_ratio.values != .0, :]
    cna = cna.loc[np.log2(cna.total_copy_ratio).abs().values > 0.5, :]
    cna = cna[[
        'sample', 'Chromosome', 'Start.bp', 'End.bp', 'n_probes',
        'total_copy_ratio'
    ]]
    cna.total_copy_ratio = np.log2(cna.total_copy_ratio).values
    cna.columns = [
        'ID', 'chrom', 'loc.start', 'loc.end', 'num.mark', 'seg.mean'
    ]

    anno_cna = annotate_cna_gene(cna)

    anno_cna = anno_cna.loc[:, ~np.in1d(
        anno_cna.columns.values, ['chrom', 'start', 'end', 'geneid']
    )]

    dup_gene = anno_cna.genename.values[anno_cna.genename.duplicated()]
    unique_cna = \
        anno_cna.loc[~np.in1d(anno_cna.genename.values, dup_gene), :]

    unique_cna.index = unique_cna.genename.values
    unique_cna = unique_cna.loc[:, unique_cna.columns.values != 'genename']

    dup_cna = pd.DataFrame()
    for n in dup_gene:
        dup_dat = anno_cna.loc[anno_cna.genename.values == n, ]
        dup_datx = dup_dat.iloc[:, 1:].apply(drop_duplicated, axis=0)
        rowname = [n] + [n+'_'+str(x+1) for x in range(dup_datx.shape[0]-1)]
        dup_datx.index = rowname
        dup_cna = pd.concat([dup_cna, dup_datx], axis=0)

    dup_cna = dup_cna.loc[~dup_cna.index.str.contains('_'), :]

    cna = pd.concat([unique_cna, dup_cna], axis=0).T

    idx_amp = np.argwhere(cna.values > 0.5)
    cna_amp = pd.DataFrame(
        np.zeros_like(cna, dtype=np.float32), cna.index.values,
        cna.columns.values
    )
    for n in range(cna_amp.shape[0]):
        gene_idx = idx_amp[idx_amp[:, 0] == n, 1]
        cna_amp.iloc[n, gene_idx] = 1

    idx_del = np.argwhere(cna.values < -0.5)
    cna_del = pd.DataFrame(
        np.zeros_like(cna, dtype=np.float32), cna.index.values,
        cna.columns.values
    )
    for n in range(cna_del.shape[0]):
        gene_idx = idx_del[idx_del[:, 0] == n, 1]
        cna_del.iloc[n, gene_idx] = 1

    clinical = clinical.sort_values(by='pair_id')
    muta = muta.sort_index(axis='index')
    cna_del = cna_del.sort_index(axis='index')
    cna_amp = cna_amp.sort_index(axis='index')

    assert (clinical.pair_id.values == muta.index.values).all()
    assert (clinical.pair_id.values == cna_del.index.values).all()
    assert (clinical.pair_id.values == cna_amp.index.values).all()

    return clinical, muta, cna_amp, cna_del


def data_preprocess(mutation_ratio, cna_ratio, dirname):

    if not os.path.exists(dirname):
        os.makedirs(dirname)
    # mutation_ratio = 0.05
    # cna_ratio = 0.05
    # dirname = 'result1'
    clinical, muta, cna_amp, cna_del = pre_data()

    muta = muta.loc[
        :, muta.apply(np.sum, axis=0)/muta.shape[0] > mutation_ratio
    ]
    cna_amp = cna_amp.loc[
        :, cna_amp.apply(np.sum, axis=0)/cna_amp.shape[0] >= cna_ratio
    ]
    cna_del = cna_del.loc[
        :, cna_del.apply(np.sum, axis=0)/cna_del.shape[0] >= cna_ratio
    ]

    total_input_gene = np.unique(np.concatenate([
        muta.columns.values, cna_amp.columns.values, cna_del.columns.values
    ]))

    # kegg mapping
    g2p = pd.read_csv('KEGG_data/gene2pathway_map.csv', index_col=0)

    # select co gene
    cogene = np.intersect1d(g2p.index.values, total_input_gene)

    # use cogene select Mutation data and CNA data
    muta = muta.loc[:, np.in1d(muta.columns.values, cogene)]
    cna_amp = cna_amp.loc[:, np.in1d(cna_amp.columns.values, cogene)]
    cna_del = cna_del.loc[:, np.in1d(cna_del.columns.values, cogene)]

    # Adj of mutation and cogene
    muta2gene = pd.DataFrame(
        np.zeros((muta.shape[1], len(cogene))),
        index=muta.columns.values, columns=cogene
    )
    for n in muta2gene.index.values:
        muta2gene.loc[n, n] = 1

    # Adj of CNA amplification and cogene
    cna_amp2gene = pd.DataFrame(
        np.zeros((cna_amp.shape[1], len(cogene))),
        index=cna_amp.columns.values, columns=cogene
    )
    for n in cna_amp2gene.index.values:
        cna_amp2gene.loc[n, n] = 1

    # Adj of CNA deletion and cogene
    cna_del2gene = pd.DataFrame(
        np.zeros((cna_del.shape[1], len(cogene))),
        index=cna_del.columns.values, columns=cogene
    )
    for n in cna_del2gene.index.values:
        cna_del2gene.loc[n, n] = 1

    # concate all the Adj
    input2gene = pd.concat([muta2gene, cna_amp2gene, cna_del2gene], axis=0)

    # concate mutation data and CNA data
    assert (muta.index.values == cna_amp.index.values).all()
    assert (muta.index.values == cna_del.index.values).all()
    data = pd.concat([muta, cna_amp, cna_del], axis=1)

    # input data type
    input_type = ['MUT']*muta.shape[1]+['AMP']*cna_amp.shape[1]
    input_type += ['DEL']*cna_del.shape[1]

    input_gene = np.concatenate(
        [muta.columns.values, cna_amp.columns.values, cna_del.columns.values],
        axis=0
    )

    print('Mutation Gene: {}'.format(muta.shape[1]))
    print('CNA Amplification Gene: {}'.format(cna_amp.shape[1]))
    print('CNA Deletion Gene: {}'.format(cna_del.shape[1]))

    pd.DataFrame({'type': input_type, 'gene': input_gene}).to_csv(
        dirname+'/input_type.csv', index=None
    )

    assert (data.columns.values == input2gene.index.values).all()

    # use cogene to select KEGG gene2pathway
    g2p = g2p.loc[input2gene.columns.values, :]

    assert (input2gene.columns.values == g2p.index.values).all()

    # pc2pb, pb2pa
    pc2pb = pd.read_csv('KEGG_data/pathC2pathB_map.csv', index_col=0)
    pb2pa = pd.read_csv('KEGG_data/pathB2pathA_map.csv', index_col=0)

    pc2pb = pc2pb.loc[g2p.columns.values, :]

    assert(g2p.columns.values == pc2pb.index.values).all()
    assert(pc2pb.columns.values == pb2pa.index.values).all()

    # maps list
    maps = [input2gene, g2p, pc2pb, pb2pa]

    # valid sample id in clinical1 and data1
    assert (data.index.values == clinical.pair_id.values).all()

    y = np.zeros(clinical.shape[0])
    y[clinical.RECIST.values == 'PD'] = 1

    # split data1 into train and test
    DiscX, ValidX, Discy, Validy = train_test_split(
        data, y, test_size=0.2, shuffle=True, random_state=219, stratify=y
    )
    return DiscX, Discy, ValidX, Validy, maps
