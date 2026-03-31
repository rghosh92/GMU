# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:11:03 2026

@author: User
"""

batch_size = 200
init_rate = 0.0005
init_rate_crank = 0.01
labelnoise = 0
input_noise = 0 
epsilons = [ .01]
gmu_epsilons = [0.001,0.00001,0]

# test_transforms = ['scale','translate','rotate','']


step_size = 10
gamma_learning = 0.8
total_epoch = 200
total_epoch_crank = 100
decay_normal = 0    
decay_regress = 0
decay_errors = 0
dropping = 0 
multiplier = int(1)


layers = [25,50]
kernels = [5,5]
layers_crank = []

training_size = 10000
mode = 'regress_batch'
use_bn = True
rescale = 1.0
rank_convert = False
input_channels = 1