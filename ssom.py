# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:04:59 2020

@author: Martin Stetter
Sensory Self-organizing Map
Implements a standard Kohonen self-organizing map for direct representation of
sensory data.

Implements the following extension - even at the expense of performance:
(a) It is explicitly designed for online learning: learns after
each individual new input. For the learning step, a variable 
neighbourhood range sigma is calculated from current input 

supports save and load operations via pickle dump
supports gpu
"""

import torch
import numpy as np

import matplotlib.pyplot as plt
import torchvision.utils as vutils
import pickle


class SOM():
    """
    Implements a standard 2D Kohonen self-organizing map with the 
    following extension:
    
    (a) It is explicitly designed for online learning: learns after
    each individual new input. For the learning step, a variable 
    neighbourhood range sigma is calculated from current input   
    
    neuron grid has dimensions n_rows times n_cols
    
    n_inputs: dimension of input signals 
    input must be a 1D ROW tensor of length n_inputs 
    
    """ 
    
    # constructor
    def __init__(self, n_rows, n_cols, n_inputs, use_gpu = True):
        
        # use gpu, if possible
        if  (torch.cuda.is_available() and use_gpu):
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_units = n_rows*n_cols
        self.n_inputs = n_inputs
        
        # list of geometric locations of each unit in SOM grid
        self.row, self.col = torch.meshgrid(torch.arange(self.n_rows), 
                                            torch.arange(self.n_cols) )
        
        self.row = self.row.to(self.device).reshape(self.n_units, 1)
        self.col = self.col.to(self.device).reshape(self.n_units, 1) 
        # current activation, normalized to max(activations) = 1
        self.activations = torch.zeros(self.n_units, 1, 
                                       device = self.device)
        # weight matrix       
        
        # stores current winner's id and co-ordinates
        self.n_win    = 0
        self.nr_win   = 0
        self.nc_win   = 0
        
        # for debugging only
        self.nr2_win = 0
        self.nc2_win = 0


        # empirical values of typical neighbourhood range and learning rate
        # CAUTION: sigma_0 should be chosen larger either in initial
        # phase of learning, or generally, for small input dimensions
        # maybe: np.power(n_inputs, -3)*np.max ..... (empirically)
        self.sigma_0 = 0.25*np.max([self.n_rows, self.n_cols])
        self.eta_0   = 0.1
        
        # activation pattern

        # has n_units rows (vertical) and n_inputs columns (horizontal)
        self.W = 0.1*torch.randn(self.n_units, self.n_inputs,
                                 device = self.device, dtype=torch.float32)


    """
    Performs a complete som learning step under pretraining conditions
    
    Winning unit is the one with minimal eukliden distance between its
    weight vector and input x
    
    x must be a 1D row tensor of size n_inputs and dtype float32
    
    sigma: neighbourhood width, determined from outside in pretraining
    eta: learning step, pretrained from outside in pretraining
    
    returns: pred_error: distance x-w_winner
             topo_meas:  distance bmu and second bmu in topographic space
    """
    def update_pretrain(self, x, sigma, eta):

        # prepare x
        x = x.to(self.device)
        X = x.repeat(self.n_units, 1)

        # calculate distance vectors: for each unit, the corresponding
        # row represents the distance vector
        v_dist = X - self.W
        
        # for all units, calculate eucliden distance between
        # units' weight vectors and input
        dists = torch.sqrt(torch.pow(v_dist, 2).sum(1))
        # caution: like this square_dists is a row tensor
                
        # determine winner unit 
        dist_winner = torch.min(dists).item()
        pred_error = dist_winner
        # index of best matching unit
        self.n_win  = torch.argmin(dists).item()
        # location of winner on grid
        self.nr_win = self.row[self.n_win].item()
        self.nc_win = self.col[self.n_win].item()
        
        # calculate topographic preservation measure
        dists[self.n_win] = 1e3
        n_2_win = torch.argmin(dists).item()
        # print(n_2_win)
        topo_meas = 0
        self.nr2_win = self.row[n_2_win].item()
        self.nc2_win = self.col[n_2_win].item()
        topo_meas = np.sqrt((self.nr_win - self.nr2_win)**2 + 
                                (self.nc_win - self.nc2_win)**2)

        # calculate gaussian activation profile, which is centered above
        # the winning unit, with width sigma
        # it is at the same time the factor for learning of each neuron
        precision = 1.0 / (sigma**2)              
        row_dists = self.row - self.nr_win
        col_dists = self.col - self.nc_win        
        # print(row_dists, col_dists)
        self.activations = torch.pow(row_dists,2) + torch.pow(col_dists,2)
        self.activations = torch.exp(-0.5 * precision * self.activations)
        
        # finally, the learning step

        h = self.activations.repeat(1, self.n_inputs)       
        # print(h)       
        self.W = self.W + eta * h * v_dist

        return pred_error, topo_meas
        
    """
    Performs a complete som learning step
    
    Winning unit is the one with minimal eukliden distance between its
    weight vector and input x
    
    x must be a 1D row tensor of size n_inputs and dtype float32
    pretrain:   use this eventually in initial learning phase
                if True, update is performed with very large neighbourhood
                sigma = max(nrow, ncol)
                should help unfolding the map properly at the beginning
    """
    def update(self, x):
        # if x is no tensor, convert to float32 tensor
        # if not torch.is_tensor(x):
        #     x = torch.from_numpy(x).float()
        # performs a perception and learning step
        x = x.to(self.device)
        # generate repmat, replicate input over n_units rows
        X = x.repeat(self.n_units, 1)

        # calculate distance vectors: for each unit, the corresponding
        # row represents the distance vector
        v_dist = X - self.W
        # for all units, calculate squared eukliden distance between
        # units' weight vectors and input
        # square_dist = torch.pow(self.W - X, 2).sum(1).reshape(self.n_units,1)
        dists = torch.sqrt(torch.pow(v_dist, 2).sum(1))
        # caution: like this square_dists is a row tensor
        
        # print(square_dists)
        
        # calculate winner unit and adaptive neighbourhood range
        dist_winner = torch.min(dists).item()
        dist_mean   = torch.mean(dists).item()
        # index of best matching unit
        self.n_win  = torch.argmin(dists).item()
        # location of winner on grid
        self.nr_win = self.row[self.n_win].item()
        self.nc_win = self.col[self.n_win].item()
 
        # calculate performace measures
        pred_error = dist_winner
     
        # calculate topographic preservation measure
        dists[self.n_win] = 1e3
        n_2_win = torch.argmin(dists).item()
        # print(n_2_win)
        topo_meas = 0
        self.nr2_win = self.row[n_2_win].item()
        self.nc2_win = self.col[n_2_win].item()
        topo_meas = np.sqrt((self.nr_win - self.nr2_win)**2 + 
                                (self.nc_win - self.nc2_win)**2)
        
        # calculate adaptive neighbourhood range
        # the smaller, the closer the winner is to data, compared to 
        # mean distance of all neurons      
        sigma = self.sigma_0 * dist_winner / dist_mean
        # avoid division by zero and provide lower limit to sigma       
        if sigma < 1:
            sigma = 1

        precision = 1.0 / (sigma**2) 
        # width = -0.5 / (self.sigma_0**2) 
               
        row_dists = self.row - self.nr_win
        col_dists = self.col - self.nc_win
        
        # print(row_dists, col_dists)
        
        # calculate gaussian activation profile, which is centered above
        # the winning unit, with width sigma
        # it is at the same time the factor for learning of each neuron
        self.activations = torch.pow(row_dists,2) + torch.pow(col_dists,2)
        self.activations = torch.exp(-0.5 * precision * self.activations)
        # 2D grid of actual units' activations
        
        # plt.imshow(self.activations)
        
        # finally, the learning step
        # h = self.activations.reshape(self.n_units,1)
        # NB: rehsape takes data from matrix row-wise to build new vector
        # caution: h is pointer to SAME data as activations, so
        # DONT WRITE INTO H HERE
        # use clone() to make a physical copy of a tensor
        
        # print(h)
        # print(h.reshape(self.n_row, self.n_col))
        h = self.activations.repeat(1, self.n_inputs)            
        
        # print(h)       
        self.W = self.W + self.eta_0 * h * v_dist
        return pred_error, topo_meas
    

    """ Returns row and column indes of currently winning neuron
    """
    def get_coordinates(self, n_win):
        nr = self.row[n_win].item()
        nc = self.col[n_win].item()
        # nr = n_win // self.n_cols
        # nc = n_win % self.n_cols
        return nr, nc

    """ 
    Returns matrix with n_row x n_col weight vectors as tiles
    
    n_ch, n_inrow, n_incol: each weight vector is interpreted as a 
    (e.g., image-) matrix of size (n_ch, n_inrows, n_incols)
    Requires n_ch*n_inrows*n_incols = self.n_inputs
    n_inrows*n_incols must be size of weight (and input) vectors  
    """                  
    def get_unfolded_weights(self, n_ch, n_inrow, n_incol):
        
        if (n_ch*n_inrow*n_incol != self.n_inputs):
            raise ValueError('wrong input dimensions')
            
        img_list = torch.zeros(self.n_units, n_ch, n_inrow, n_incol)
        for i in range(self.n_units):
            im = self.W[i,].reshape(n_ch, n_inrow, n_incol).cpu()
            img_list[i,:,:,:] = im
            # print(img_list.size())
            
        unfolded_weights = vutils.make_grid(img_list, nrow = self.n_col,
                                            padding=1, normalize=True)
        return np.transpose(unfolded_weights, (1,2,0))
    

    """
    Draws the component dim2 of weight vector against its component dim1
    together with the topographic grid of corresponding neurons
    uses axes handle axes to plot in
    
    use draw_grid(dim1, dim2) if you just wnt to draw in a new figure
    """    
    def draw_grid(self, dim1, dim2, ax = None):
        # make sure 0 <= dim1, dim2, < n_inputs
#         fig = plt.figure()
        
        if ax is None:
            ax = plt.axes()
            
        W = self.W.reshape(self.n_rows, self.n_cols, self.n_inputs)
        
        for nr in(range(self.n_rows)):
            for nc in range(self.n_cols):
                if (nc < self.n_cols-1):
                    ax.plot([W[nr,nc,dim1], W[nr,nc+1,dim1]],
                              [W[nr,nc,dim2], W[nr,nc+1,dim2]],
                              'b-', linewidth=1) 
                if (nr < self.n_rows-1):    
                    ax.plot([W[nr,nc,dim1], W[nr+1,nc,dim1]],
                              [W[nr,nc,dim2], W[nr+1,nc,dim2]],
                              'b-', linewidth=1)         
        

    """ 
    Saves the SOM's state to file
    
    Uses a pickle dump to store all relevant instance variables to file
    with name filename
    to load the state, create a new SOM0 object and call its load_state
    with the appropriate filename - see load_state function
    """
    def save_state(self, filename):   
        with open(filename, 'wb') as f: 
            self.save_state_stream(f)
        
    """
    saves state to stream f (to be opened elsewhere!!)
    """
    def save_state_stream(self, f):   
        pickle.dump([self.n_rows,
                     self.n_cols,
                     self.n_units,
                     self.n_inputs,
                     self.sigma_0,
                     self.eta_0,
                     self.activations,
                     self.W], f)
      
    """ 
    Loads the SOM's state from file
    
    to load the state, create a new SOM0 object and call its load_state
    with the appropriate filename,
    e.g., som = som0.SOM0(0, 0, 0, use_gpu = (True/False))
          som.load_state(saved_som0_file)
    """
    def load_state(self, filename):   
        with open(filename, 'rb') as f: 
            self.load_state_stream(f)
        
    """
    loads state from stream f (to be opened elsewhere!!)
    """
    def load_state_stream(self, f):   
        [self.n_rows,
        self.n_cols,
        self.n_units,
        self.n_inputs,
        self.sigma_0,
        self.eta_0,                         
        self.activations,
        self.W]  = pickle.load(f)  
        
                # list of geometric locations of each unit in SOM grid
        self.row, self.col = torch.meshgrid(torch.arange(self.n_rows), 
                                            torch.arange(self.n_cols) )        
        self.row = self.row.to(self.device).reshape(self.n_units, 1)
        self.col = self.col.to(self.device).reshape(self.n_units, 1) 
        self.activations = self.activations.to(self.device)
        self.W = self.W.to(self.device).float()
        
        
 
        
        
        
        
        