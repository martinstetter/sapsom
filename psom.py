# -*- coding: utf-8 -*-
"""
Created on Thu Apr  15 2020

@author: Martin Stetter

Predictive SOM
Each state neuron keeps a conditional probability table, which learns
the conditional distribution of successor neurons
For a neuron s_t which is winner at time t, the table (vector) keeps the 
probabilities p(s_(t+1) | s_t)

Apart from that follows the implementation of SOM class contained in ssom.py

Dependencies: 
Requires ssom.py
"""

import torch
import numpy as np

import pickle

import ssom

class PSOM(ssom.SOM):
    """
    Implements a predictive self-organizing map
    predictions are kept within-layer as 
    joint un-normalized contingency tablesprobability table H(s_(t+1), s_t) 
    In this first implementation,H(s_(t+1), s_t) contains the absolute 
    frequency of cases, where s_(t+1) follows s_t
    Dimension of H: (n_units x n_units)
    (transparency beats efficiency here ...)
    """ 
    
    # constructor
    def __init__(self, n_rows, n_cols, n_inputs, use_gpu = True):
        ssom.SOM.__init__(self, n_rows, n_cols, n_inputs, use_gpu)
        
        # for debugging only
        

        self.p_t = torch.ones(self.n_units, 1,  device = self.device)
        self.p_t = self.p_t / self.n_units
        
        self.p_tp1 = self.p_t.clone()
        self.p_hat_tp1  = self.p_t.clone()
        
        # State transition matrix
        # column s_t approximates p(s_(t+1) |s_t)
        # at the beginning 
        self.T = torch.ones(self.n_units, self.n_units, 
                             device = self.device) / self.n_units
        # learning rate for update of T
        self.gamma = 0.1
      

    """ 
    Resets state transition matrix
    """        
    def reset_T(self):
        self.T = torch.ones(self.n_units, self.n_units, 
                             device = self.device) / self.n_units
    
    """
    Uses update method of som0 for standard learning step, 
    but in addition updates the prediction contingency table
    
    Here: entry T(new_win, old_win)++, i.e., hard s_t -> s_t+1
    update_T: if True, state transition matrix is being updated
              otherwise a standard SOM learning step is performed 
    transition from winner to next winner
    """
    def update_pred(self, x):
        

        # # calculate p(s_t)
        self.p_t = self.activations.reshape(self.n_units, 1)
        # print(self.p_t.sum())
        norm = torch.sum(self.p_t)
        if not (norm == 0): # treats only very first step
            self.p_t = self.p_t / norm
            
        # used for count-increment learning rule
        n_t = self.n_win
            
        # perform SOM learning step (including determining winners)
        super(PSOM, self).update(x)
                
        # calculate p(s_(t+1)) 
        self.p_tp1 = self.activations.reshape(self.n_units,1)
        self.p_tp1 = self.p_tp1 / torch.sum(self.p_tp1)
   
        self.p_hat_tp1 = self.T.mm(self.p_t)
        # update state transition matrix
        diff_p = (self.p_tp1 - self.p_hat_tp1).squeeze()
        # here: minimise MSE (p - p_hat)
        self.T = self.T + self.gamma * \
            torch.ger( diff_p, self.p_t.squeeze())
                              
        # calculate performace measures    
        norm_diff = torch.norm(diff_p).cpu().item()
        # true mode location
        n_mode = torch.argmax(self.p_tp1)
        nr, nc = self.get_coordinates(n_mode)  
        # predicted mode location
        n_hat_mode = torch.argmax(self.p_hat_tp1)
        nr_hat, nc_hat = self.get_coordinates(n_hat_mode)  
        mode_dist = np.sqrt((nr_hat - nr)**2 + (nc_hat - nc)**2)
        if (mode_dist == 0):
            mode_correct = 1
        else:
            mode_correct = 0
        # norm = torch.repeat(self.T.sum(0))
        return norm_diff, mode_dist, mode_correct           
        
     
    """ 
    Predicts the most likely next psom-state given input x.
    
    Determines winner ID (mode of activation driven by x) 
    and predicts most likely next state (mode of conditional pdf)
    if allow_self = True, the same state as the current winner state
    may be predicted, otherwise a different state is enforced for t+1
    """
    def predict_bymode(self, x, allow_self = False):
        # determine winning unit given x using parent's update method
        # NB: update also enforces a learning step (online learning philosophy)
        super(PSOM, self).update(x)

        if (max(self.T[:, self.n_win]) < 1.0/(self.n_units-1)):
            # return random unit, if no next state has ever been experienced
            print('oops, no next state learned')
            n_win_next = torch.randint(self.n_units, (1,)).item()
        elif allow_self:
            # determine next winner simply as argmax
            n_win_next = torch.argmax(self.T[:,self.n_win]).item()

        else:
            # exclude actual winner from being predicted as next state
            # do so by setting p(s_t, s_t) to zero prior to determining mode
            T_temp = self.T[self.n_win, self.n_win]
            self.T[self.n_win, self.n_win] = 0
            n_win_next = torch.argmax(self.T[:, self.n_win]).item()
            # restore original table entry
            self.T[self.n_win, self.n_win] = T_temp  
        
        nr_next, nc_next = self.get_coordinates(n_win_next)
        # determine next predicted input as codebook vector of next winner        
        state_next = self.W[n_win_next,:]       
        return nr_next, nc_next, state_next        
        
    
    """
    Predicts expected location of mode of next activation and expected
    next state 
    nr_next, nc_next: expected mode of next activation
    NB: due to expectation mght be non-integer locations (population code of 
    activations)
    state_next: expected next input state
    """
    def predict_byexpect(self, x):
        # determine winning unit given x using parent's update method
        # NB: update also enforces a learning step (online learning philosophy)
        super(PSOM, self).update(x)
        
        p_t = self.activations / torch.sum(self.activations)
        
        # print(self.activations)
        # print(self.n_win)
        
        # estimate distribution of next state
        p_hat_tp1 = self.T.mm(p_t)       
        # calculate expectations
        nr_next = torch.sum(p_hat_tp1 * self.row).item()
        nc_next = torch.sum(p_hat_tp1 * self.col).item()
        # state_next = p_hat_tp1.t().mm(self.W).squeeze()
        state_next = p_hat_tp1.t().mm(self.W).squeeze()
        return nr_next, nc_next, state_next
    
    
    
    """ 
    Saves the PSOM's state to file
    
    Uses a pickle dump to store all relevant instance variables to file
    with name filename
    to load the state, create a new PSOM object and call its load_state
    with the appropriate filename - see load_state function
    """    
    def save_state(self, filename):   
        with open(filename, 'wb') as f: 
            self.save_state_stream(f)
            
    """
    saves network state to stream f (to be opened elsewhere!!)
    """ 
    def save_state_stream(self, f)    :
        super(PSOM, self).save_state_stream(f)
        pickle.dump([self.T], f)
        

    """ 
    Loads the PSOM's state from file
    
    to load the state, create a new SOM0 object and call its load_state
    with the appropriate filename,
    e.g., som = psom.PSOM(0, 0, 0, use_gpu = (True/False))
          som.load_state(saved_psom_file)
    """
    def load_state(self, filename):   
        with open(filename, 'rb') as f: 
            self.load_state_stream(f)

    """ 
    loads psom state from stream f
    """        
    def load_state_stream(self, f):
        super(PSOM, self).load_state_stream(f)
        [self.T] = pickle.load(f)
        self.T = self.T.to(self.device)   
        self.p_t = torch.ones(self.n_units, 1,  device = self.device)
        self.p_t = self.p_t / self.n_units
        
        self.p_tp1 = self.p_t.clone()
        self.p_hat_tp1  = self.p_t.clone()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        