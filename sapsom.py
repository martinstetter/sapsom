# -*- coding: utf-8 -*-
"""
Created on 20.04.2020

@author: Martin Stetter

Implements the basic principle of State-Action-Prediction SOM
for fast imitation learning 

Dependencies: 
Requires psom.py which in turn requires ssom.py
"""

import torch
import numpy as np

import matplotlib.pyplot as plt
import pickle
# import itertools

import gym
import psom

class sapsom():

    """
    State-action-prediction self-organizing map
    Version details:
    - uses manual setup: one psom per action
    - at present only supports 2 hardwired actions
    - at present only works with cartole gym environment
    - prediction as implemented in psom.
    
    """     
    # constructor
    def __init__(self, n_rows, n_cols, use_gpu = True):
        # som0.SOM0.__init__(self, n_row, n_col, n_inputs, use_gpu)
                # use gpu, if possible
        if  (torch.cuda.is_available() and use_gpu):
            self.device = "cuda:0"
        else:
            self.device = "cpu"
            
        self.a_left   = 0 # action to the left
        self.a_right  = 1
        self.n_inputs = 4
        self.som_left = psom.PSOM(n_rows, n_cols, self.n_inputs, use_gpu = use_gpu)
        self.som_right= psom.PSOM(n_rows, n_cols, self.n_inputs, use_gpu = use_gpu)
        # make SOMs share identical state representation        
        # self.som_right.W = self.som_left.W
        
        # Parameter defaults for pretraining: 
        # Only used when pretraining is applied
        # pretraining with exponential decays over n_epi episodes,
        # as SOMs are usually trained
        self.sigma_0  = 0.5 * np.max([self.som_left.n_rows, self.som_left.n_cols])
        self.sigma_T  = 0.5
        self.eta_0    = 0.3
        self.eta_T    = 0.01
        self.n_T      = 200  # corresponds to T
        self.factor_s = np.exp(-np.log(self.sigma_0/self.sigma_T)/self.n_T)
        self.factor_e = np.exp(-np.log(self.eta_0  /self.eta_T  )/self.n_T)       
        
        # later on stores target state to be reached/maintained
        self.nr_target = 0
        self.nc_target = 0
        self.state_target = torch.zeros(1, self.n_inputs).to(self.device)
        self.use_constraint = torch.ones(1, self.n_inputs).to(self.device)
        
        # create gym environment to play with
        self.env = gym.make('CartPole-v0').unwrapped
        self.env_reset()
 
      
    """
    Copies representational state  (not states according to predictions!)
    from source psom to target psom
    som_from: source psom
    som_to  : target psom
    """
    def clone_soms(self, som_from, som_to):
        som_to.n_win       = som_from.n_win
        som_to.nr_win      = som_from.nr_win 
        som_to.nc_win      = som_from.nc_win
        
        som_to.nr2_win     = som_from.nr2_win 
        som_to.nc2_win     = som_from.nc2_win
        som_to.activations = som_from.activations.clone()
        som_to.W           = som_from.W.clone()

   
    """
    Sets pretraining parameters and takes care of dependencies
    """
    def set_pretrain_options(self, s_0, s_T, e_0, e_T, n_T):
        self.sigma_0  = s_0
        self.sigma_T  = s_T
        self.eta_0    = e_0
        self.eta_T    = e_T
        self.n_T      = n_T  # corresponds to T
        self.factor_s = np.exp(-np.log(self.sigma_0/self.sigma_T)/self.n_T)
        self.factor_e = np.exp(-np.log(self.eta_0  /self.eta_T  )/self.n_T)   
    
    
    """
    Offers the possibility to pretrain the soms' representations following a
    exponential decay scenario as SOMs are usually trained.
    The schedule is specified by the training schedule
    parameters sigma_0, sigma_T etc ...
    Trains only one som and, after training, copies the weight matrix into the
    other
    """
    def pretrain_representation(self, verbose = True):
        sigma = self.sigma_0
        eta = self.eta_0
        # for monitoring progress
        pred_errs = []
        topo_meas = []
        sum_pe = 0
        sum_tm = 0
        epi_steps = 0
        state = self.env_reset()
        done = False
        for t in range(self.n_T):
            # train an episode, monitor mean error measures
            while not done:
                pe, tm = self.som_left.update_pretrain(state, sigma, eta)
                sum_pe += pe
                sum_tm += tm
                epi_steps += 1                                  
                state, done = self.env_step(self.env_random())
        
            # finalize episode, calculate measures
            pred_errs.append(sum_pe/epi_steps)
            topo_meas.append(sum_tm/epi_steps)
            sum_pe = sum_tm = epi_steps = 0
            if verbose:
                print('Episode:%d,  pred_err:%.3f  topo_meas: %.3f' 
                      %(t, pred_errs[-1], topo_meas[-1]))
            # initialize new episode
            done = False
            state = self.env_reset()
            sigma = sigma * self.factor_s
            eta   = eta   * self.factor_e                 
                
        self.clone_soms(self.som_left, self.som_right)
        
        return pred_errs, topo_meas
    
    
    """
    Trains the soms' representation using adaptive online neighbourhood width
    Trains only one som and, after training, copies the weight matrix into the
    other
    """
    def train_representation(self, n_episodes = 100, verbose = True):
        pred_errs = []
        topo_meas = []
        sum_pe = 0
        sum_tm = 0
        epi_steps = 0
        state = self.env_reset()
        done = False
        for t in range(n_episodes):
            # train an episode, monitor mean error measures
            while not done:
                pe, tm = self.som_left.update(state)
                sum_pe += pe
                sum_tm += tm
                epi_steps += 1                                  
                state, done = self.env_step(self.env_random())
        
            # finalize episode, calculate measures
            pred_errs.append(sum_pe/epi_steps)
            topo_meas.append(sum_tm/epi_steps)
            sum_pe = sum_tm = epi_steps = 0
            if verbose:
                print('Episode:%d,  pred_err:%.3f  topo_meas: %.3f' 
                      %(t, pred_errs[-1], topo_meas[-1]))
            # initialize new episode
            done = False
            state = self.env_reset()               
                
        self.clone_soms(self.som_left, self.som_right)
        
        return pred_errs, topo_meas
   
    
    """
    Trains system in interventional mode
    n_T episodes are run under random actions of the agent and the 
    action-conditioned state transition matrices are learned
    """    
    def train_prediction (self, n_episodes, verbose = True, render = False):   
        # initialize training

        # nd = md = pc = 0
        norm_diff = []
        mode_dist = []        
        percent_correct = []
        sum_nd = 0 
        sum_md = 0
        sum_pc = 0
        epi_steps = 0

        for t in range(n_episodes):
            done = False
            state = self.env_reset()
            self.som_left.update(state)
            self.clone_soms(self.som_left, self.som_right)
            # train an episode
            while not done:
                next_action = self.env_random()
                state, done = self.env_step(next_action)
                if render:
                    self.env_render()           
                if (next_action == self.a_left):
                    # update som_left's Transition matrix
                    nd, md, pc = self.som_left.update_pred(state)
                    # assure identical state representations
                    self.clone_soms(self.som_left, self.som_right)

                if (next_action == self.a_right):
                    # update som_right's Transition matrix
                    nd, md, pc = self.som_right.update_pred(state)
                    # assure identical state representations
                    self.clone_soms(self.som_right, self.som_left)
                    
                sum_nd  += nd
                sum_md  += md               
                sum_pc  += pc
                epi_steps += 1

            # finalize episode
            norm_diff.append(sum_nd/epi_steps) 
            mode_dist.append(sum_md/epi_steps) 
            percent_correct.append(sum_pc/epi_steps) 
            sum_nd = sum_md = sum_pc = epi_steps = 0           
            if verbose:
                print('Episode:%d  norm diff:%.3f  mode_dist:%.1f  frac. correct:%.3f' 
                      %(t, norm_diff[-1], mode_dist[-1], percent_correct[-1]) )
            
        return norm_diff, mode_dist, percent_correct
      

    """
    Plays action_sequence on own environment
    """
    def play(self, start_state, action_sequence, render = True):
        device = start_state.device
        states = [start_state]
        self.env_set(start_state)
        self.env_render()
        plt.pause(0.1)
        for a in action_sequence:
            next_state, _ = self.env_step(a)
            states.append(next_state.to(device))
            if render:
                self.env_render()
                plt.pause(0.1)
            
        return states
    
       
    """ 
    Predicts a sequence of states, (basically, imagining the future)
    given a start state and a sequence of actions (should be a list) 
    returns the list of predicted states
    pred = mode: prediction by mode (mostl likely next state, 
                 as described in the paper)
    pred = expect: predict by expected next stae (alternative) 
    """
    def predict_sequence(self, start_state, action_sequence, pred = 'mode',
                         render = True):
        device = start_state.device
        states = [start_state]
        state_pred = start_state
        if (render):
            self.env_set(start_state)
            self.env_render()
            plt.pause(0.1)
        for a in action_sequence:
            if (a == self.a_left):
                if (pred == 'expect'):
                    _, _, state_pred = self.som_left.predict_byexpect(state_pred)
                else:
                    _, _, state_pred = self.som_left.predict_bymode(
                        state_pred, allow_self = False)
            else:
                if (pred == 'expect'):
                    _, _, state_pred = self.som_right.predict_byexpect(state_pred)
                else:
                    _, _, state_pred = self.som_right.predict_bymode(
                        state_pred, allow_self = False)               
            states.append(state_pred.to(device))
            if render:
                self.env_set(state_pred)
                self.env_render()
                plt.pause(0.1)

        return states
    

    """ 
    Predicts a sequence of states, (basically, imagining the future)
    given a start state and a sequence of actions (should be a list) 
    Returns only the final winner coordinates and predicted state
    pred = mode: prediction by mode (mostl likely next state, 
                 as described in the paper)
    pred = expect: predict by expected next state (alternative) 
    """
    def predict(self, start_state, action_sequence, pred = 'mode'):
        # device = start_state.device
        state_pred = start_state
        for a in action_sequence:
            
            # print(a)          
            if (a == self.a_left):
                if (pred == 'expect'):
                    nr_pred, nc_pred, state_pred = self.som_left.predict_byexpect(state_pred)
                else:
                    nr_pred, nc_pred, state_pred = self.som_left.predict_bymode(
                        state_pred, allow_self = False)
            else:
                if (pred == 'expect'):
                    nr_pred, nc_pred, state_pred = self.som_right.predict_byexpect(state_pred)
                else:
                    nr_pred, nc_pred, state_pred = self.som_right.predict_bymode(
                        state_pred, allow_self = False)                

        return nr_pred, nc_pred, state_pred


    """
    Sets the means and precisions (component-wise) of a sequence of 
    target states that are thought to be presented to the agent 
    as a sequence to be imitated.
    Hence: This sequence is not explicitly presented, instead means and precisions
    of the state components are communicated to the agent. This mimicks the 
    sequential storage of states into the SOM by a non-distractible 
    working memory mechanism, resulting in a collection of active SOM states
    which the agent in the following tries to approximate - the targets.
    state: state means vector as given by self.env_step(), i.e., a 4 D float32
    torch tensor
    use_constraint: vector of the 4 precisions (inverse variances)
    """
    def set_targetstate(self, state, use_constraint):
        # determine location of SOM winner unit for target state "state"
        self.som_left.update(state)  
        self.clone_soms(self.som_left, self.som_right)
        # self.state_target = torch.from_numpy(state)
        # target state is codebook of winner triggered by state
        self.state_target = self.som_left.W[self.som_left.n_win,:]
        self.nr_target   = self.som_left.nr_win
        self.nc_target   = self.som_left.nc_win  
        self.use_constraint = use_constraint.to(self.device)
        
   

    """ 
    Trys to reach target state after environment reset for as long as possible
    before stopped by done signal
    
    Uses one-step greedy search in state (input) space, 
    not on predicted latent states
    
    Uses prediction of next step "by mode" (max of p_hat_tplus1)
    """
    def ai_state_mode(self, render = True):
        MAX_STEPS = 200
        
        state = self.env_reset()
        # # make him start off equilibrium
        # state, done = self.env_step(0)
        # state, done = self.env_step(0)        
        # state, done = self.env_step(0)         
        
        x  = [state[0].item()]
        th = [state[2].item()]
        steps_survived = 0

        done = False
        while ((not done) and (steps_survived < MAX_STEPS)):
            if render:
                self.env_render()
            # calculate predicted square distance to target as result of
            # next action = left
            _, _, next_state_left = self.som_left.predict_bymode(state, allow_self=False)
            diff_left = (next_state_left - self.state_target) * self.use_constraint
            # diff_left = next_state_left - self.state_target
            state_dist_left = torch.norm(diff_left).item()
            
            _, _, next_state_right = self.som_right.predict_bymode(state, allow_self=False)
            diff_right = (next_state_right - self.state_target) * self.use_constraint
            # diff_right = next_state_right - self.state_target
            state_dist_right = torch.norm(diff_right).item()           
                                                
            # carry out whatever brings us closer to goal in topographic space
            if (state_dist_left < state_dist_right):
                next_action = self.a_left
            else:
                next_action = self.a_right
            

            state, done = self.env_step(next_action)
            x.append(state[0].item())
            th.append(state[2].item())
            steps_survived += 1
            
        print('steps survived', steps_survived)
        return x, th, steps_survived, state
       
        
      
##############################################################################
### Wrappers for interaction with environment (state data format etc.)
##############################################################################
    def env_reset(self):
        state = self.env.reset()
        return torch.from_numpy(state).float()
    
    def env_set(self, state):
        if torch.is_tensor(state):
            state = state.cpu().numpy()
        self.env.state = tuple(state)
    
    def env_random(self):
        return self.env.action_space.sample()
        
    def env_step(self, action):
        state, _, done, _ = self.env.step(action)
        state = torch.from_numpy(state).float()
        return state, done
    
    def env_render(self):
        self.env.render()
  
    def env_close(self):
        self.env.close()
    
    
    """
    Saves sapsom setup to file
    
    Uses a pickle dump to store all relevant instance variables to file
    with name filename
    to load the state, create a new sapsom object and call its load_state
    with the appropriate filename - see load_state function
    """
    def save_state(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump([self.a_left, 
                         self.a_right, 
                         self.n_inputs,
                         self.sigma_0,
                         self.sigma_T,
                         self.eta_0,
                         self.eta_T,
                         self.n_T], f)
            self.som_left.save_state_stream(f)
            self.som_right.save_state_stream(f)
     
        
    """ 
    Loads the sapsom setup's state from file
    
    to load the state, create a new sapsom object and call its load_state
    with the appropriate filename,
    e.g., sap = SapSomV1.SapSom_V1(0, 0, use_gpu = (True/False))
          sap.load_state(saved_sapsom_file)
    """
    def load_state(self, filename):
        with open(filename, 'rb') as f: 
            [self.a_left, 
            self.a_right, 
            self.n_inputs,
            self.sigma_0,
            self.sigma_T,
            self.eta_0,
            self.eta_T,
            self.n_T] = pickle.load(f)           
            self.som_left.load_state_stream(f)
            self.som_right.load_state_stream(f)
    
    
    
############################# END SAPSOM #####################################    
##############################################################################    
##############################################################################    
##############################################################################    
##############################################################################    
##############################################################################   
##############################################################################


##############################################################################
### utility class that provides diagnostics for a sapsom
##############################################################################
            
class SapsomDiagnoser():

    
    def __init__(self, sap):
        self.sap = sap
        
               
    """ 
    auxiliary function for diagnostics: draws grids of 4 input
    dimension pairs
    """
    def _draw_grids(self):
        # draw grids of 4 combination of input dimensions
        fig, axes = plt.subplots(2, 2)

        self.sap.som_left.draw_grid(0, 1, axes[0, 0])
        axes[0, 0].set_xlim(-2.5, 2.5)
        axes[0, 0].set_ylim(-3, 3)
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('x dot')
       
        self.sap.som_left.draw_grid(2, 3, axes[0, 1])
        axes[0, 1].set_xlim(-0.25, 0.25)
        axes[0, 1].set_ylim(-3, 3)
        axes[0, 1].set_xlabel('theta')
        axes[0, 1].set_ylabel('theta dot')
        
        self.sap.som_left.draw_grid(0, 2, axes[1, 0])
        axes[1, 0].set_xlim(-2.5, 2.5)
        axes[1, 0].set_ylim(-0.25, 0.25)
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('theta')  
        
        self.sap.som_left.draw_grid(1, 3, axes[1, 1])
        axes[1, 1].set_xlim(-3, 3)
        axes[1, 1].set_ylim(-3, 3)
        axes[1, 1].set_xlabel('theta dot')
        axes[1, 1].set_ylabel('x dot') 

        return fig, axes
    
    
##############################################################################
### methods for representation diagnostics
##############################################################################        
        
    """
    Draws map grids for four delected state dimension pairs:
    x-x_dot, theta-theta_dot, x-theta, x_dot-theta_dot
    """    
    def draw_grids(self):    
        # switch off learning for diagnostics
        # eta_0 = self.sap.som_left.eta_0
        # self.sap.som_left.eta_0 = 0
        # self.sap.som_right.eta_0 = 0

        # just draws grids of 4 input dimension pairs
        fig, axes = self._draw_grids()
        
    """
    Draws map grids plus traces of real episodes under random actions
    """
    def draw_grids_states(self, n_episodes = 1, render = True):    
        
        # draws grids plus traces of real episodes in phase space    
        fig, axes = self._draw_grids()
        state = self.sap.env_reset()  
        done = False               
        for i in range(n_episodes):
            # train an episode, monitor mean error measures
            while not done:
                if render: 
                    self.sap.env_render()
                axes[0, 0].plot(state[0], state[1], 'go', markersize=2)
                axes[0, 0].set_title('blue: SOM cd-book')
                # plt.show()
                axes[0, 1].plot(state[2], state[3], 'go', markersize=2)
                axes[0, 1].set_title('green: states')
                # plt.show()
                axes[1, 0].plot(state[0], state[2], 'go', markersize=2)
                # plt.show()
                axes[1, 1].plot(state[1], state[3], 'go', markersize=2) 
                if render:
                    plt.show()
                    plt.pause(0.1)                               
                state, done = self.sap.env_step(self.sap.env_random())
            # initialize new episode
            done = False
            state = self.sap.env_reset()

    """
    Plays one episode under random action, shows input state and 
    corresponding codebook entry of winning unit
    """
    def play_states_mapping(self):
        # switch off learning for diagnostics
        eta_0 = self.sap.som_left.eta_0
        self.sap.som_left.eta_0 = self.sap.som_right.eta_0 = 0
        fig, axes = self._draw_grids()
        state = self.sap.env_reset()
        done = False               
        while not done:
            self.sap.env_render()
            axes[0, 0].plot(state[0], state[1], 'go', markersize=3)
            axes[0, 0].set_title('green:state')
            plt.show()
            axes[0, 1].plot(state[2], state[3], 'go', markersize=3)
            axes[0, 1].set_title('red: winner cd-book')
            plt.show()
            axes[1, 0].plot(state[0], state[2], 'go', markersize=3)
            plt.show()
            axes[1, 1].plot(state[1], state[3], 'go', markersize=3)
            plt.show()
            plt.pause(1e-3)
            self.sap.som_left.update(state)
            som_state = self.sap.som_left.W[self.sap.som_left.n_win].cpu()
            axes[0, 0].plot(som_state[0], som_state[1], 'ro', markersize=3)
            plt.show()
            axes[0, 1].plot(som_state[2], som_state[3], 'ro', markersize=3)
            plt.show()
            axes[1, 0].plot(som_state[0], som_state[2], 'ro', markersize=3)
            plt.show()
            axes[1, 1].plot(som_state[1], som_state[3], 'ro', markersize=3)
            plt.show()
            plt.pause(0.1) 
            state, done = self.sap.env_step(self.sap.env_random())   
             
        # restore learnng rates
        self.sap.som_left.eta_0 = self.sap.som_right.eta_0 = eta_0                
                


# ##############################################################################
# ### methods for prediction diagnostics
# ##############################################################################
    
    """
    Calculates norm of difference p - p_hat, distances of modes and percent 
    correct prediction of mode
    """
    def calc_pred_errors(self, n_episodes = 1):
        # switch off learning temporalily
        eta_0 = self.sap.som_left.eta_0
        self.sap.som_left.eta_0 = self.sap.som_right.eta_0 = 0
        gamma = self.sap.som_left.gamma
        self.sap.som_left.gamma = self.sap.som_right.gamma = 0        
        
        norm_diff, mode_dist, frac_correct = self.sap.train_prediction(n_episodes, render = False)
        mean_nd = np.mean(norm_diff)
        mean_md = np.mean(mode_dist)
        mean_fc = np.mean(frac_correct)
        
        # restore learning rates
        self.sap.som_left.eta_0 = self.sap.som_right.eta_0 = eta_0
        self.sap.som_left.gamma = self.sap.som_right.gamma = gamma  
        return mean_nd, mean_md, mean_fc
    
    
    """
    Animates p (as image), p_hat (as image) and their difference for  
    one episode under random actions
    """
    def play_probs_error(self):
        # switch off learning temporarily
        eta_0 = self.sap.som_left.eta_0
        self.sap.som_left.eta_0 = self.sap.som_right.eta_0 = 0

        n_rows = self.sap.som_left.n_rows
        n_cols = self.sap.som_left.n_cols        

        state = self.sap.env_reset()
        self.sap.env_render()
        done = False           
        while not done:
            # determine activation before next actual step

            # sample randomly next action
            next_action = self.sap.env_random()

            # let network predict next winner
            if (next_action == self.sap.a_left):                  
                _, mode_dist, _ = self.sap.som_left.update_pred(state)
                p = self.sap.som_left.p_tp1.cpu()
                p_hat = self.sap.som_left.p_hat_tp1.cpu()   
                self.sap.clone_soms(self.sap.som_left, self.sap.som_right)
            else:
                _, mode_dist, _ = self.sap.som_right.update_pred(state)
                p = self.sap.som_right.p_tp1.cpu()
                p_hat = self.sap.som_right.p_hat_tp1.cpu()
                self.sap.clone_soms(self.sap.som_right, self.sap.som_left)                           

            plt.subplot(1, 2, 1)
            plt.imshow(p.reshape(n_rows, n_cols))
            plt.title('True p_tplus1')
            plt.subplot(1, 2, 2)
            plt.imshow(p_hat.reshape(n_rows, n_cols))
            plt.title('Predicted p_hat_tplus1')
            print('mode distance: ', mode_dist)
                    
            plt.show()
            plt.pause(0.5)                 

            state, done = self.sap.env_step(next_action)
            self.sap.env_render()
        # restore learning rates    
        self.sap.som_left.eta_0 = self.sap.som_right.eta_0 = eta_0
        

    """
    calculate a quiver for next states under real dynamics and 
    under predictions, respectively  
    concentrate on most important phase space theta vs. theta_dot
    """    
    def quivernext_real_pred(self, n_episodes = 10, pred = 'mode'):
    
        print(pred)
        
        states      = torch.zeros(0, 2, dtype = torch.float32)
        arrows_true = torch.zeros(0, 2, dtype = torch.float32)
        arrows_pred = torch.zeros(0, 2, dtype = torch.float32)
        # arrows_som  = torch.zeros(0, 2, dtype = torch.float32)
        for  i in range(n_episodes):
            print('Episode ', i)
            state = self.sap.env_reset()
            done = False           
            while not done:
                states = torch.cat((states, state[2:4].unsqueeze(0)), 0)

                # next_action = 0  # left only
                # next_action = 1  # right only
                next_action = torch.randint(0,2,(1,)).item()
                
                # envs actions are set with own random number generator, 
                # hence env.seed does not work
                # therefore use torch random generator
                
                # calculate true evolution 
                next_state_true, done = self.sap.env_step(next_action)                                        
                arrow = next_state_true - state
                arrows_true = torch.cat((arrows_true, arrow[2:4].unsqueeze(0)), 0)
               
                # calculate predicted evolution
                # i.e., predict next winner neuron, 
                # take its predicted state as next
                        
                if (next_action == self.sap.a_left): 
                    if (pred == 'expect'):
                            _, _, next_state_pred = self.sap.som_left.predict_byexpect(
                                state) 
                    else:
                        _, _, next_state_pred = self.sap.som_left.predict_bymode(
                            state, allow_self=False)                        
                else:
                    if (pred == 'expect'):
                         _, _, next_state_pred = self.sap.som_right.predict_byexpect(
                                state)                        
                    else:
                        _,  _, next_state_pred = self.sap.som_right.predict_bymode(
                            state, allow_self=False)
                arrow = next_state_pred.cpu() - state
                arrows_pred = torch.cat((arrows_pred, arrow[2:4].unsqueeze(0)), 0) 
                
                # calculate evolution, if next winner neuron had been 
                # predicted correctly
                # practically: present next true state to som, 
                # use its codebook vector
                # self.sap.som_left.update_pred(next_state_true)
                # next_state_som = self.sap.som_left.W[self.sap.som_left.n_win].cpu()
                # arrow = next_state_som - state
                # arrows_som = torch.cat((arrows_som, arrow[2:4].unsqueeze(0)), 0)                     
                
                state = next_state_true
     
        plt.quiver(states     [:,0], states     [:,1], 
                   arrows_true[:,0], arrows_true[:,1], color = 'b', width=0.003, label = 'real')
        plt.quiver(states     [:,0], states     [:,1], 
                   arrows_pred[:,0], arrows_pred[:,1], color = 'r', width=0.003, label = 'pred.')    

        plt.xlabel('theta [rad]')
        plt.ylabel('theta dot')
        plt.legend()
        plt.title('(a) direction of motion in angle phase plane') 
 
    """
    For a number of states, calculate a quiver for predicted dynamics 
    given next action left vs. next action right    
    concentrate on most important phase space theta vs. theta_dot
    """
    def quivernext_left_right(self, n_episodes = 10, pred = 'mode'):           
 

        states       = torch.zeros(0, 2, dtype = torch.float32)
        arrows_left  = torch.zeros(0, 2, dtype = torch.float32)
        arrows_right = torch.zeros(0, 2, dtype = torch.float32)
        for  i in range(n_episodes):
            print('Episode ', i)
            state = self.sap.env_reset()
            done = False           
            while not done:
                states = torch.cat((states, state[2:4].unsqueeze(0)), 0)
                if (pred == 'expect'):
                    _, _, pred_state_left  = self.sap.som_left.predict_byexpect(state)
                    _, _, pred_state_right = self.sap.som_right.predict_byexpect(state)
                else:
                    _, _, pred_state_left  = self.sap.som_left.predict_bymode(state, allow_self=False)
                    _, _, pred_state_right = self.sap.som_right.predict_bymode(state, allow_self=False)                    
                arrow = pred_state_left.cpu() - state
                arrows_left = torch.cat((arrows_left, arrow[2:4].unsqueeze(0)), 0)
                arrow = pred_state_right.cpu() - state
                arrows_right = torch.cat((arrows_right, arrow[2:4].unsqueeze(0)), 0)                                      
                
                # next_action = self.sap.env_random()
                # envs actions are set with own random number generator, 
                # hence env.seed does not work
                # therefore use torch random generator
                next_action = torch.randint(0,2,(1,)).item()
                # next_action = 0  # left only
                # next_action = 1  # right only

                state, done = self.sap.env_step(next_action)
     
        plt.quiver(states      [:,0], states      [:,1], 
                   arrows_left [:,0], arrows_left [:,1], color = 'b',  
                   width=0.003, label = 'action left')
        plt.quiver(states      [:,0], states      [:,1], 
                   arrows_right[:,0], arrows_right[:,1], color = 'r',  
                   width=0.003, label = 'action right')      

        plt.xlabel('theta [rad]')
        plt.ylabel('theta dot')
        plt.legend()
        plt.title('(b) predicted direction of motion in angle phase plane')           

    
    

        
        
        
        
        