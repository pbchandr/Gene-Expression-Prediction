#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: pbchandr
"""
import copy
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split

def saturated_mutageneis(seq, nucleotides=['A', 'C', 'T', 'G']):
    """
    Function to generate Saturated Mutagenesis
    (https://en.wikipedia.org/wiki/Saturation_mutagenesis)
    - Gen all possible SNP mutations for each base pair in the sequence
    - Each sample has 5000 bps with nucleotide inforation.
    - Since we are dealing with nucleotide sequence, there can be only 4 nucleotides at each position
        -As the base sequence is given, we can have only 3 other mutations for each position.
        -This gives a total of 5000*3 + 1 number of mutated sequences
    """
    data = np.array([list(seq),]*(len(seq)*(len(nucleotides)-1)+1))
    i = 0
    j = 0
    while i <= (len(data)-len(nucleotides)+1):
        n1 = copy.copy(nucleotides)
        n1.remove(data[i, j])
        data[(i+1):(i + len(n1)+1), j] = n1
        i = i + len(nucleotides)-1
        j = j + 1
    return data

def perform_one_hot_encoding(data):
    """
    Function to perform one hot encoding of the sequence which is the input to the system
        - Sequences are converted into 4*5000 dim vector
        - 4 dimensions:- Order of dimension is A-1, G-2, C-3, T-4import collections
        - The value in each cell is either 0 or 1 representing corresponding nucleotide.
    """
    
    nseq = len(data)
    print nseq
    info = np.zeros(shape=(nseq, 4, len(data[0])))
    num_iter = 1
    for i in range(0, nseq):
        indices = [j for j, a in enumerate(data[i]) if a == 'A']
        info[i, 0, indices] = 1

        indices = [j for j, a in enumerate(data[i]) if a == 'G']
        info[i, 1, indices] = 1

        indices = [j for j, a in enumerate(data[i]) if a == 'C']
        info[i, 2, indices] = 1

        indices = [j for j, a in enumerate(data[i]) if a == 'T']
        info[i, 3, indices] = 1
        
        if i%math.ceil(nseq/10) == 0:
            print (num_iter*10, 'percent complete')
            num_iter = num_iter + 1

    return info

def get_data(gex_gepi_fname, req_flatten=False, add_epigenetics=False, chromosome = "All"):
    """Function to fetch the data, preprocess it and then split into train, test, and validation set"""
    
    # Read the gene expression with epigenetic information
    dat = pd.read_csv(gex_gepi_fname, sep = '\t')
    print('Dat shape', dat.shape)
        
    
    # Filter the data based on chromosome
    geq_data = pd.DataFrame()
    if chromosome == "All":
        geq_data = dat
    else:
        chrom = [x for x in chromosome.split(',')]
        for ind, chr in enumerate(chrom):
            print 'chr'+chr
            if ind == 1:
                geq_data = dat.loc[dat['chromosome_name'] == ('chr'+chr)]
            else:
                geq_data = geq_data.append(dat.loc[dat['chromosome_name'] == ('chr'+chr)])

    # Get class labels based on the percentile of FPKM values
    geq_data['percentile']= geq_data.FPKM.rank(pct = True, ascending=False)
    geq_data['rank']= geq_data.FPKM.rank(ascending=False)

    # Get class labels
#    num_seq = np.max(geq_data['rank'])
#    geq_data = geq_data[(geq_data['rank'] > num_seq*2/3) |  (geq_data['rank'] < num_seq/3)]
#    seq_labels = geq_data['rank'].values
#    seq_labels[seq_labels > num_seq*2/3] = 0
#    seq_labels[seq_labels != 0] = 1
    
    geq_data = geq_data[(geq_data['TPM'] >= 2) |  (geq_data['TPM'] == 0)]
    
    pos_class = geq_data.loc[geq_data['TPM'] >= 2]
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
    seq_labels[seq_labels == 0] = 0
    
    
    # Free memory
    del dat
    geq_data = geq_data.reset_index(drop = True)
    print('geq_data', geq_data.shape)
    
    
    # Perform one-hot encoding for the upstream sequences.
    seq_info = perform_one_hot_encoding(geq_data["sequence"].values)
    print('seq_info', seq_info.shape)
    
    # Add epigenetic information if required
    if add_epigenetics:
        epigen_cols = [col for col in geq_data.columns if 'chipseq' in col]
        print('epigen_cols', len(epigen_cols))
        nseq, nr, nc = seq_info.shape
        
        gex_gepi_data = np.zeros([nseq, nr + len(epigen_cols), nc])

        for i in range(0, nseq):
            gex_gepi_data[i,0:4,:] = seq_info[i,:,:]
    
        for indx, col in enumerate(epigen_cols):
            epigen_info = geq_data[col].values
            for i in range(0, nseq):
                gex_gepi_data[i, 4+indx, :] = np.reshape(np.asarray(map(float, epigen_info[i])), [1, -1])
            print(col + ' complete')
        print('Gex Gepi Data', gex_gepi_data.shape)
    else:
       gex_gepi_data = seq_info 
            
    
    # Neural Networks: CNN takes 3d data but feed forward network takes only 2D data and hence need flattening.
    if req_flatten:
        gex_gepi_data = gex_gepi_data.reshape(gex_gepi_data.shape[0], -1)
    
    print gex_gepi_data.shape
    seq_labels = np.eye(2)[np.int32(seq_labels)]
    
    return(gex_gepi_data, seq_labels, geq_data)
    
#def encode_epigenetics_info(epigenetics_data, geq_data, nrow):
#    """
#    Function to encode epigenetics inforation as inputs.
#        - For each seq of length 5000, each position contains (0,1,2) value..
#        - 0 - No epigenetics present, 1 - present, 2 - peak
#    """
#    # If max location of upstream is greater than the epigenetic position, append zeros to the end of remaining positions.    
#    if max(geq_data['upstream_end']) > np.shape(epigenetics_data)[0]:
#        epigenetics_data = np.append(epigenetics_data, np.zeros(shape=(max(geq_data['upstream_end']) - np.shape(epigenetics_data)[0], 1)))
#    
#    epigenetics_info = np.zeros(shape=(nrow, 1, 5000))
#    for i in range(nrow):
#        epigenetics_info[i, 0, :] = np.transpose(epigenetics_data[range((geq_data['upstream_start'][i]-1), geq_data['upstream_end'][i])])
#    return epigenetics_info
#
#
#
#def preprocess_data(gex_gepi_fname, req_flatten=False, add_epigenetics=False, chromosome = "All"):
#    """
#    Function to fetch the data, preprocess it and then split into train, test, and validation set
#    """
#    dat = pd.read_csv(gex_gepi_fname, sep = '\t')
#    nrow, _ = dat.shape
#    geq_data = dat
#    
#    # Filter the data based on chromosome
#    if chromosome == "All":
#        geq_data = dat
#    else:
#        chrom = [x for x in chromosome.split(',')]
#        for ind, chr in enumerate(chrom):
#            print 'chr'+chr
#            if ind == 1:
#                geq_data = dat.loc[dat['chromosome_name'] == ('chr'+chr)]
#            else:
#                geq_data = geq_data.append(dat.loc[dat['chromosome_name'] == ('chr'+chr)])
#    
#    # Frre memory
#    del dat
#    geq_data = geq_data.reset_index(drop = True)
#    # Perform one-hot encoding for the upstream sequences.
#    seq_info = perform_one_hot_encoding(geq_data["sequence"].values, nrow)
#    
#    # Add epigenetic Get encoding of epigenetics information
#    if add_epigenetics:
#        geq_data.loc[:, geq_data.columns.str.startswith('chipseq')]
#        dat = pd.read_csv(epigen_fname)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
#        epigenetics_data = dat.values
##        epigenetics_data = pd.read_csv(epigen_fname)
#        for i in range(0,6):
#            epigenetics_info = encode_epigenetics_info(epigenetics_data[:, i], geq_data, nrow)
#
#            # Merge epigenetics and Sequence data
#            seq_info = np.concatenate((seq_info, epigenetics_info), axis=1)
#            print np.shape(seq_info)
#        del epigenetics_data
#        del dat
#        
#
#    # Get the output label which is the percentile value and convert into binary labels.
##    seq_orig_labels = geq_data["percentile"].values
##    seq_labels = np.copy(seq_orig_labels)
##    seq_labels[seq_labels < np.median(seq_orig_labels)] = 0
##    seq_labels[seq_labels != 0] = 1
#    
#    print seq_info.shape
#    seq_orig_labels = geq_data["percentile"].values
#    max_seq_labels = max(seq_orig_labels) - 0.1
#    indx_fltr = np.any([seq_orig_labels <= 0.3, seq_orig_labels > max_seq_labels], axis=0)
#    seq_info = seq_info[indx_fltr,]
#    geq_data = geq_data.ix[indx_fltr]
#    seq_labels = seq_orig_labels[indx_fltr]
#    seq_labels[seq_labels <= 0.3] = 0
#    seq_labels[seq_labels >= max_seq_labels] = 1
#
#    # Neural Networks: CNN takes 3d data but feed forward network takes only 2D data and hence need flattening.
#    if req_flatten:
#        seq_info = seq_info.reshape(seq_info.shape[0], -1)
#    
#    print seq_info.shape
#    seq_labels = np.eye(2)[np.int32(seq_labels)]
#    seq_orig_labels = np.eye(2)[np.int32(seq_orig_labels)]
#
#    return(seq_info, seq_labels, seq_orig_labels, geq_data)
