#!/bin/bash
python -u gepred_train_model_mod.py --gex_gepi_data_file='./data/tissue/brain_chr_all.csv' --add_epigenetic_info=True --chromosome_name='All' --filter_sizes='500' --num_cnn_dims=128 --strides=1 --pool_size=500 --num_fc_layers=1 --num_fc_dims=512 --dropout_rate=0.25 --train_epochs=20 --eval_interval=1 --batch_size=150 --learn_rate=0.001 > trash/brain-all-new-epigene-fs500-ncd128-ps500-fc512-dr25-bs150-te20.txt

