#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: pbchandr
"""

import argparse

import csv
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from gepred_data_util import get_data
from gepred_nnmodels_mod import Cnn
import sys

def evaluate(sess, model, data, labels, batch_size, write_result=False):
    """Evaluate and print results"""
    predictions = []

    for ptr in range(0, len(data), batch_size):
        prediction = sess.run(model.logit,
                              feed_dict={model.input_x: data[ptr: ptr + batch_size],
                                         model.input_y: labels[ptr: ptr + batch_size],
                                         model.dropout_keep_prob: 1.0})
        predictions.extend(prediction)

    predictions = np.asarray(predictions)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(labels, axis=1)
    
    if write_result:       
        predFile = open('pred.csv', 'w')  
        with predFile:
            writer = csv.writer(predFile)
            writer.writerows(y_pred)
        
        trueFile = open('true.csv', 'w')  
        with trueFile:
            writer = csv.writer(trueFile)
            writer.writerows(y_true)
    

    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average="micro")

    if write_result:
        print("MAX FOUND. Writing results to file")
        print("P:{:.5f}, R:{:.5f}, F1:{:.5f}".format(precision, recall, fscore))
        print(confusion_matrix(y_true, y_pred))

    return fscore
    
def train_step(sess, model, data, labels, batch_size, dropout_rate):
    """Train data"""
    avg_cost = 0.0
    total_batch = int(len(data)/batch_size)
    for ptr in range(0, len(data), batch_size):
        # Run backprop and cost during training
        _, epoch_cost = sess.run([model.optimizer, model.cost],
                                 feed_dict={model.input_x: data[ptr:ptr + batch_size],
                                            model.input_y: labels[ptr:ptr + batch_size],
                                            model.dropout_keep_prob: 1-dropout_rate})
        # Compute average loss across batches
        avg_cost += epoch_cost / total_batch
    return avg_cost

def train(args):
    """ Training method"""
        
    # Step 1: Load the train and validation data
    if os.path.isfile(args.gex_gepi_data_file):
        data, labels, _ = get_data(gex_gepi_fname = args.gex_gepi_data_file, 
                                   add_epigenetics = args.add_epigenetic_info, 
                                   req_flatten = args.req_flatten_data,
                                   chromosome = args.chromosome_name)
        trd, tvd, trr, tvr = train_test_split(data, labels, test_size=0.2, random_state=42)
        vald, ted, valr, ter = train_test_split(tvd, tvr, test_size=0.5, random_state=42)
    else:
        files = os.listdir(args.gex_gepi_data_file)
        tr_genes, val_genes, te_genes = [],[],[]
        for indx, data_file in enumerate(files):
            print(indx, args.gex_gepi_data_file+data_file)
            data, labels, geq_data = get_data(gex_gepi_fname = args.gex_gepi_data_file+data_file, 
                                       add_epigenetics = args.add_epigenetic_info, 
                                       req_flatten = args.req_flatten_data,
                                       chromosome = args.chromosome_name)
            if indx == 0:
                unq_genes = np.unique(geq_data['gene_id'])
                tr_genes, vr_genes = train_test_split(unq_genes, test_size = 0.2)
                val_genes, te_genes = train_test_split(vr_genes, test_size = 0.5)
            
                trd = data[geq_data.index[geq_data['gene_id'].isin(tr_genes)]]
                vald = data[geq_data.index[geq_data['gene_id'].isin(val_genes)]]
                ted_1 = data[geq_data.index[geq_data['gene_id'].isin(te_genes)]]
    
                trr = labels[geq_data.index[geq_data['gene_id'].isin(tr_genes)]]
                valr = labels[geq_data.index[geq_data['gene_id'].isin(val_genes)]]
                ter_1 = labels[geq_data.index[geq_data['gene_id'].isin(te_genes)]]
            else:
                trd = np.append(trd, data[geq_data.index[geq_data['gene_id'].isin(tr_genes)]], axis=0)
                vald = np.append(vald, data[geq_data.index[geq_data['gene_id'].isin(val_genes)]], axis=0)
                ted_2 = data[geq_data.index[geq_data['gene_id'].isin(te_genes)]]
    
                trr = np.append(trr, labels[geq_data.index[geq_data['gene_id'].isin(tr_genes)]], axis=0)
                valr = np.append(valr, labels[geq_data.index[geq_data['gene_id'].isin(val_genes)]], axis=0)
                ter_2 = labels[geq_data.index[geq_data['gene_id'].isin(te_genes)]]
            
    # Split to train and test and load the data
    
    print ('Training shape: ', trd.shape)
    print ('Validation shape: ', vald.shape)
    print ('Test shape: ', ted.shape)
#    print ('Test shape: ', ted_1.shape)
#    print ('Test shape: ', ted_2.shape)
    
    # Delete unused data
    del data, labels
    
    trd = trd.transpose(0,2,1)
    vald = vald.transpose(0,2,1)
    ted = ted.transpose(0,2,1)
    
#    ted_1 = ted_1.transpose(0,2,1)
#    ted_2 = ted_2.transpose(0,2,1)

    # Step 2: Build the graph and cnn object
    _, seq_w, seq_h = trd.shape
    num_classes = len(np.unique(trr))

    
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=session_conf) as sess:
        model = Cnn(sequence_width = seq_w, sequence_height = seq_h, num_classes = num_classes, 
                    num_conv_filters = args.num_cnn_dims, conv_filter_sizes = args.filter_sizes, 
                    conv_strides = args.strides, pooling_size = args.pool_size, 
                    num_fc_layers = args.num_fc_layers, num_fc_kernels = args.num_fc_dims,
                    lrn_rate=args.learn_rate)
        
        # Train cycle
        sess.run(tf.global_variables_initializer())
        max_tr_f1 = 0.0
        max_val_f1 = 0.0
        
        
        for epoch in range(args.train_epochs):
            avg_cost = train_step(sess, model, trd, trr, args.batch_size, args.dropout_rate)
            print("\nEpoch:", '%02d' % (epoch+1), "cost=", "{:.5f}".format(avg_cost))
            
            if epoch % args.eval_interval == 0:
                train_f1 = evaluate(sess, model, trd, trr, args.batch_size)
                val_f1 = evaluate(sess, model, vald, valr, args.batch_size)
                print("-Training : {:.5f}".format(train_f1), "Val : {:.5f}".format(val_f1))
                
                if train_f1 > max_tr_f1:
                    if val_f1 > (max_val_f1 - 0.01):
                        max_tr_f1 = train_f1
                        # max_val_f1 = val_f1 if val_f1 > max_val_f1 else max_val_f1
                        max_val_f1 = val_f1
                        if args.save is not None:
                            # Write model checkpoint to disk
#                            te1_f1 = evaluate(sess, model, ted_1, ter_1, args.batch_size)
#                            te2_f1 = evaluate(sess, model, ted_2, ter_2, args.batch_size)
#                            
#                            print("-Testing : {:.5f}".format(te1_f1))
#                            print("-Testing : {:.5f}".format(te2_f1))
#                           
                            te_f1 = evaluate(sess, model, ted, ter, args.batch_size)                           
                            print("-Testing : {:.5f}".format(te_f1))
                            
                            print("Saving model to {}\n".format(args.save))
                            # saver.save(sess, args.save)
                    if train_f1 > 0.99:
                        print("Optimization Finished!")
                        sys.exit(1)

                        
                        

        print("Optimization Finished!")


def main():
    """ Main: This method is used to parse all the arguements and call train function """
    parser = argparse.ArgumentParser()

    # Input data file
    parser.add_argument('--gex_gepi_data_file', type=str, default='./data/tissue/',
                        help='Gene Expression with epigenetic data input file location')
    parser.add_argument('--add_epigenetic_info', type=bool, default=False, 
                        help = 'Epigenetic details to be included or not')
    parser.add_argument('--req_flatten_data', type=bool, default=False, 
                        help='Should the data be flattened')
    parser.add_argument('--chromosome_name', type=str, default='All', 
                        help='Chromosomes that should be included for analysis, comma delimited')
    # Hyperparameters
    parser.add_argument('--filter_sizes', type=str, default='100,500',
                        help='sizes of filters, comma delimited. Number of filter sizes should be equal to the number of layers')
    parser.add_argument('--num_cnn_dims', type=str, default='128,128',
                        help='Number of filters, comma delimited. i.e. dimensions for weights in each filters.')
    parser.add_argument('--strides', type=str, default='1,1',
                        help='Strides for filters, comma delimited. i.e steps to move for sliding window.')
    parser.add_argument('--pool_size', type=int, default=500,
                        help='Pooling size')
    parser.add_argument('--num_fc_layers', type=int, default=1,
                        help='Number of fully connected layers to be used after convolution layer')
    parser.add_argument('--num_fc_dims', type=str, default='512',
                        help='Number of kernels for fully connected layers, comma delimited.')
    parser.add_argument('--dropout_rate', type=float, default=1.0,
                        help='Droupout percent for handling overfitting. 1.0 to keep all and 0 to keep none')
    
    
    # Settings
    parser.add_argument('--train_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--eval_interval', type=int, default=1, help='Evaluate once in _ epochs')
    parser.add_argument('--batch_size', type=int, default=144, help='Batch size of training')
    parser.add_argument('--learn_rate', type=float, default=0.001, help='learning rate')
    

    # Model save paths
    parser.add_argument('--save', type=str, default="model/cnn", help="path to save model")
    args = parser.parse_args()
    print(args)
    train(args)

if __name__ == '__main__':
    main()
