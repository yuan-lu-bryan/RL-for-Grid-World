from cgitb import small
from typing import Callable
from matplotlib.pyplot import grid
from tqdm import tqdm
import numpy as np

from model import Model, Actions

def sarsa(model: Model, maxit: int = 100, num_episode: int = 300, eps: float=0.1, alpha: float=0.3):
    V = np.zeros((model.num_states,))
    pi = np.zeros((model.num_states,))
    Q = np.zeros((model.num_states, len(Actions)))
    num_iter = np.zeros((num_episode,))
    
    def eps_greedily(s, epsilon):
        unif = np.random.rand()

        if unif < epsilon:
            idx = np.random.randint(0, len(Actions))
            return Actions(idx)
        
        else:
            return Actions(np.argmax(Q[s]))
    
    for i in tqdm(range(num_episode), disable=False):
        s = model.start_state
        a = eps_greedily(s, eps) 
        Q_old = np.copy(Q)
        for _ in range(maxit):
            num_iter[i]+=1
            r = model.reward(s, a)
            possible_dict= model._possible_next_states_from_state_action(s, a)
            possible_s = list(possible_dict.keys())
            prob = list(possible_dict.values())
            s_new = np.random.choice(possible_s, p = prob)
            a_new = eps_greedily(s_new,eps)
            if s_new == model.goal_state:
                Q[s, a] = Q[s, a] + alpha*(r - Q[s, a])
            else:
                Q[s, a] = Q[s, a] + alpha*(r + model.gamma*Q[s_new, a_new] - Q[s, a])
            s = s_new
            a = a_new
            if s == model.goal_state:
                break
        
        if np.all(abs(Q-Q_old)<=0.0001):
            print("breaking")
            break
        
    V = np.amax(Q, axis=1)
    pi = np.argmax(Q, axis=1)
    return V, pi, np.sum(num_iter)   
        
        