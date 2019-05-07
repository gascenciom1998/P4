# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
from itertools import accumulate
import matplotlib.pyplot as plt

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        self.max_vel = 50
        self.min_vel = -50
        self.max_td = 450
        self.min_td = -150
        self.max_hd = 400
        self.min_hd = -100
        self.max_abs = 450
        self.min_abs = 0

        self.bin_td = 50
        self.bin_hd = 50
        self.bin_vel = 5
        self.bin_abs = 50

        self.dim_vel = int((self.max_vel - self.min_vel) / self.bin_vel + 1)
        self.dim_td = int((self.max_td - self.min_td) / self.bin_td + 1)
        self.dim_hd = int((self.max_hd - self.min_hd) / self.bin_hd + 1)
        self.dim_abs = int((self.max_abs - self.min_abs) / self.bin_abs + 1)

        self.Q = np.zeros((self.dim_vel, self.dim_td, self.dim_hd, self.dim_abs, 2))
        self.counts = np.zeros((self.dim_vel, self.dim_td, self.dim_hd, self.dim_abs, 2))

        self.gamma = 0.9
        self.epoch = 1
        self.eps = 0.001
        self.scores = []
        

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.epoch += 1

    def bin_set(self, num_bin, max_bin, min_bin, type_bin):
        if num_bin < min_bin:
            num_bin = min_bin
        elif num_bin > max_bin:
            num_bin = max_bin

        if num_bin < 0:
            num_bin = num_bin * (-1) + max_bin

        return int(num_bin / type_bin)

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        rand_num = npr.rand()
        rand_action = 0 if rand_num < 0.5 else 1

        if self.last_action == None:
            self.last_action = rand_action
            self.last_state  = state
            return rand_action
        
        lv = self.bin_set(self.last_state['monkey']['vel'], self.max_vel, self.min_vel, self.bin_vel)
        ltd = self.bin_set((self.last_state['tree']['top'] - self.last_state['monkey']['top']), self.max_td, self.min_td, self.bin_td)
        lhd = self.bin_set(self.last_state['tree']['dist'], self.max_hd, self.min_hd, self.bin_hd)
        labs = self.bin_set(self.last_state['monkey']['top'], self.max_abs, self.min_abs, self.bin_abs)
        la = self.last_action
        reward = self.last_reward

        cv = self.bin_set(state['monkey']['vel'], self.max_vel, self.min_vel, self.bin_vel)
        ctd = self.bin_set((state['tree']['top'] - state['monkey']['top']), self.max_td, self.min_td, self.bin_td)
        chd = self.bin_set(state['tree']['dist'], self.max_hd, self.min_hd, self.bin_hd)
        cabs = self.bin_set(state['monkey']['top'], self.max_abs, self.min_abs, self.bin_abs)
        
        new_action = 0 if self.Q[cv, ctd, chd, cabs, 0] >= self.Q[cv, ctd, chd, cabs, 1] else 1
        Q_max = self.Q[cv, ctd, chd, cabs, new_action]
        eps = self.eps * (((1 + self.counts[cv, ctd, chd, cabs, new_action]))) if self.counts[cv, ctd, chd, cabs, new_action] > 0 else self.eps
        #eps = self.eps
        new_action = rand_action if rand_num < eps else new_action


        lr = 1 / (2 + self.counts[lv, ltd, lhd, labs, self.last_action])
        #fixed lr = 0.2
        self.counts[cv, ctd, chd, cabs, new_action] += 1
        self.Q[lv, ltd, lhd, labs, la] -= lr * (self.Q[lv, ltd, lhd, labs, la] - (reward + self.gamma * Q_max))

        
        self.last_action = new_action
        self.last_state  = state

        return new_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()

    plt.figure()
    plt.plot(range(len(hist)), hist)
    plt.xlabel("Iterations")
    plt.ylabel("Score")
    plt.savefig("Rel-Abs-lr=adaptive-eps=adaptive-gamma=0.9-line.png")
    plt.close

    plt.figure()
    plt.hist(hist)
    plt.xlabel("Score")
    plt.ylabel("Counts")
    plt.savefig("Rel-Abs-lr=adaptive-eps=adaptive-gamma=0.9-hist.png")

    plt.figure()
    plt.plot(range(len(hist)), list(accumulate(hist)))
    plt.xlabel("Iterations")
    plt.ylabel("Running Score Sum")
    plt.savefig("Rel-Abs-lr=adaptive-eps=adaptive-gamma=0.9-sum.png")

    print(np.max(hist))
    print(np.mean(hist))
    pg.quit()
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 1000, 1)

	# Save history. 
	np.save('hist',np.array(hist))


