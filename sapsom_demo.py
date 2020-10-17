# -*- coding: utf-8 -*-
"""
Created on 20.04.2020

@author: Martin Stetter

Tests an experimental "state action prediction som setup
"""

import torch
import numpy as np
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision import transforms
# import time
import gym
import matplotlib.pyplot as plt
import sapsom


##############################################################################  
### 0. Create a new sapsom with 16x16 repreentational units
##############################################################################

sap = sapsom.sapsom(16, 16, use_gpu = False)
# diagnostics utility class
sd = sapsom.SapsomDiagnoser(sap)
 
##############################################################################
### 1. Pretrain representation - optional
##############################################################################

sap.set_pretrain_options(8, 0.1, 0.3, 0.01, 1000)
pred_errs, topo_meas = sap.pretrain_representation(verbose = True)

# report 
print(" Mean PE: %.5f  \quad Mean TM: %.3f" 
      %(np.mean(pred_errs[-50:]), np.mean(topo_meas[-50:])))


# visualize training success        
plt.figure()
plt.plot(pred_errs,label="pred. errors")
plt.plot(topo_meas,label="topo. measure")
plt.xlabel("epochs")
plt.ylabel("error measures")
plt.legend()

# draw SOM grids 
sd.draw_grids()

# draw SOM grids plus a collection of system states
sd.draw_grids_states(n_episodes=50, render = False)

# play an animation of states and SOM winner codebooks for an episode
sd.play_states_mapping()

##############################################################################
### 2. Train representation under adaptive neighbourhood 
##############################################################################

sap.som_left.eta_0 = sap.som_right.eta_0 = 0.05
pred_errs, topo_meas = sap.train_representation(n_episodes = 500, verbose = True)

# report 
print(" Mean PE: %.5f  \quad Mean TM: %.3f" 
      %(np.mean(pred_errs[-50:]), np.mean(topo_meas[-50:])))


# visualize training success        
plt.figure()
plt.plot(pred_errs,label="pred. errors")
plt.plot(topo_meas,label="topo. measure")
plt.xlabel("epochs")
plt.ylabel("error measures")
plt.legend()


# draw SOM grids 
sd.draw_grids()

# draw SOM grids plus a collection of system states
sd.draw_grids_states(n_episodes=50, render = False)

# play an animation of states and SOM winner codebooks for an episode
sd.play_states_mapping()


##############################################################################
### 3. Train predictions
##############################################################################
sap.som_left.reset_T()
sap.som_right.reset_T()
# choose learning rates
sap.som_left.eta_0 = sap.som_right.eta_0 = 0.05
sap.som_left.gamma = sap.som_right.gamma = 0.05
norm_diff, mode_dist, percent_correct = sap.train_prediction(1000, render = False)

print("Norm diff: %.4f  Mode dist.:%.2f  frac. correct:%.3f" 
                          %(np.mean(norm_diff[-100:]), 
                           np.mean(mode_dist[-100:]),
                           np.mean(percent_correct[-100:])                           
                           ) ) 

# visualize training success        
plt.figure()
plt.plot(norm_diff,label="|p-p_hat|")
plt.plot(mode_dist,label="mode distance")
plt.plot(percent_correct,label="frac. correct")
plt.xlabel("epochs")
plt.ylabel("measures")
plt.legend()

# Transition matrix conditioned on left pushes
plt.figure(1)
plt.imshow(sap.som_left.T.cpu())

# Transition matrix conditioned on right pushes
plt.figure(2)
plt.imshow(sap.som_right.T.cpu())

# play true and predicted next SOM activations for one episode 
plt.figure(3)
sd.play_probs_error()

# mplot true and predicted state changes for next time steps for some episodes
plt.figure(4)
sd.quivernext_real_pred(n_episodes = 10, pred = 'mode')

# plot predicted state changes following left vs. right push for a number of episodes
plt.figure(5)
sd.quivernext_left_right(n_episodes = 10, pred = 'mode')


##############################################################################
### 3. Imagination/planning: predict sequences under certain actions
##############################################################################

# prediction only here: switch off learningchoose learning rates
sap.som_left.eta_0 = sap.som_right.eta_0 = 0.0
sap.som_left.gamma = sap.som_right.gamma = 0.0

seq_len = 10

# all left pushes
action_sequence = [0]*seq_len

# all right pushes
action_sequence = [1]*seq_len


# random pushes
action_sequence = torch.randint(0,2,(seq_len,)).numpy()

# oscillation

action_sequence = [0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0]

action_sequence
sap.env.render()

# predict state sequence under action sequence specified
start_state = sap.env_reset()

# Play rendered video of cartpole under true time evolution
sap.play(start_state, action_sequence)


# Play rendered video of cartpole under predicted time evolution
plt.pause(1)
states = sap.predict_sequence(start_state, action_sequence, 
                              pred='mode',render = True)

sap.env.close()


##############################################################################
### 4. Inference: reach and maintain target state    
##############################################################################

# 1st arg = means of sequence of target states presented to the agent 
# second arg = precision of sequence of target states presented to the agent
# Only means and precisions (capped to [0, 1]) are communicated to the agent
# example
# sap.set_targetstate( torch.tensor([0.0, 0.0, 0.0, 0.0]), 
#                      torch.tensor([0.0, 0.0, 1.0, 1.0]) )
# In the presented sequence, x and x dot had means zero and varied strongly
# theta and theta dot had mean zero and varied barely
# this corresponds to balancing the pole



# require all 4 coordinates to be satisisfied
sap.set_targetstate( torch.tensor([0.0, 0.0, 0.0, 0.0]), 
                     torch.tensor([1.0, 1.0, 1.0, 1.0]) )

# require only theta and theta_dot to be satisified
sap.set_targetstate( torch.tensor([0.0, 0.0, 0.0, 0.0]), 
                     torch.tensor([0.0, 0.0, 1.0, 1.0]) )
 
sap.set_targetstate( torch.tensor([0.0, 0.0, 0.2, 1.0]),
                     torch.tensor([0.0, 0.0, 1.0, 1.0]) )
 
print(sap.nr_target)
print(sap.nc_target)
print(sap.state_target)

sap.ai_state_mode()

sap.env_close()


##############################################################################
### 5. Save trained sapsom network for later use   
##############################################################################
sap.save_state('./sapsom_cartpole_net.pkl') 



##############################################################################  
### 6. Load sapsom network for later use
##############################################################################
sap = sapsom.sapsom(16, 16, use_gpu=False)    
sap.load_state('./sapsom_cartpole_net.pkl')   
# sap.load_state('./sapsom_cartpole_16_300.pkl') 
# sap.load_state('./sapsom_cartpole_16_1000.pkl')  
# connect it to sapsom diagnoser
sd = sapsom.SapsomDiagnoser(sap)

### ... 




























