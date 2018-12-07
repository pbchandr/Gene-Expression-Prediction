#!/usr/bin/env python2:
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 20:12:08 2018

@author: pbchandr
"""
import pandas as pd
import numpy as np
import random

geq_data = pd.read_csv('./data/tissue/brain_chr_all.csv', sep = '\t')
t1 = geq_data.describe()
geq_data1 = pd.read_csv('./data/tissue/lung_chr_1_2_3_4_19.csv', sep = '\t')

nrow, _ = geq_data.shape

import os
d = './data/tissue/'
print(os.path.isfile(d))

t = os.listdir(d)


tr_split = int(np.ceil(len(unq_genes)*0.8))
unq_genes = np.random.shuffle(unq_genes)
tr_genes = unq_genes[:80]
val_genes = unq_genes[unq_genes ]


from sklearn.model_selection import train_test_split
unq_genes = np.unique(geq_data['gene_id'])
tr_genes, vr_genes = train_test_split(unq_genes, test_size = 0.2)
val_genes, te_genes = train_test_split(vr_genes, test_size = 0.5)


tr_indx = geq_data.index[geq_data['gene_id'].isin(tr_genes)]
val_indx = geq_data.index[geq_data['gene_id'].isin(val_genes)]
te_indx = geq_data.index[geq_data['gene_id'].isin(te_genes)]

trd = data[tr_indx]
vald = data[val_indx]
ted = data[te_indx]

trr = labels[tr_indx]
valr = labels[val_indx]
ter = labels[te_indx]

chrom = "19"
dat = pd.DataFrame()
if chrom == "All":
    dat = geq_data
else:
    chrom = [x for x in chrom.split(',')]
    for ind, chr in enumerate(chrom):
        print 'chr'+chr
        if ind == 1:
            dat = geq_data.loc[geq_data['chromosome_name'] == ('chr'+chr)]
        else:
            dat = dat.append(geq_data.loc[geq_data['chromosome_name'] == ('chr'+chr)])

dat = dat.reset_index(drop = True)


# Get ranks for the remaining data
dat['percentile']= dat.FPKM.rank(pct = True, ascending=False, method ="dense")
dat['rank']= dat.FPKM.rank(ascending=False, method ="dense")

# Get class labels
seq_labels = dat["rank"].values
num_seq = len(seq_labels)
dat = dat[(dat['rank'] > num_seq*2/3) |  (dat['rank'] < num_seq/3)]
seq_labels = seq_labels

seq_labels[seq_labels >= np.median(seq_labels)] = 0
seq_labels[seq_labels != 0] = 1


#from gepred_data_util import perform_one_hot_encoding
#
seq_info = perform_one_hot_encoding(dat["adadasequence"].values, nrow)
nrow, _, _ = seq_info.shape

seq_labels = dat["percentile"].values
seq_labels[seq_labels >= np.median(seq_labels)] = 0
seq_labels[seq_labels != 0] = 1

a, b = np.unique(seq_labels, return_counts=True)
print(np.asarray((a,b)))


# Task 1: Function to get the 
epigen_cols = [col for col in dat.columns if 'chipseq' in col]
for indx, col in enumerate(epigen_cols):
    if indx > 2:
        break
    epigen_info = dat[col]
    for i in range(0, nrow):
        seq_info[i] = np.append(seq_info[i], np.reshape(np.asarray(map(float, epigen_info[i])), [1, -1]), axis = 0)



t1 = np.reshape(np.asarray(map(float, dat['chipseq1'][1])), [1, -1])
t2 = np.vstack([seq_info[1,:,:], t1])


from gepred_data_util import get_data
data, labels, info = get_data(gex_gepi_fname = './data/Brain/brain.csv', 
                              add_epigenetics = True, 
                              req_flatten = False,
                              chromosome = 'All')





geq_data = geq_data[(geq_data['TPM'] >= 1) |  (geq_data['TPM'] == 0)]
    
pos_class = geq_data.loc[geq_data['TPM'] >= 1]
pos_class = pos_class.reset_index(drop=True)
print('pos_class', pos_class.shape)

neg_class = geq_data.loc[geq_data['TPM'] == 0]
neg_class = neg_class.reset_index(drop=True)

rndm_neg_class_indx = np.arange(neg_class.shape[0])
np.random.shuffle(rndm_neg_class_indx)
print('rndm_neg_class_indx', len(rndm_neg_class_indx))

neg_class = neg_class.iloc[rndm_neg_class_indx[np.arange(pos_class.shape[0])]]
print('neg_class', neg_class.shape)

geq_data = pos_class.append(neg_class, ignore_index=True)
seq_labels = geq_data['TPM'].values
seq_labels[seq_labels != 0] = 1