from cgitb import small
from typing import Callable
from matplotlib.pyplot import grid
from tqdm import tqdm
import numpy as np

from model import Model, Actions

def expected_sarsa(model: Model, maxit: int = 100, num_episode: int = 300, eps: float=0.1, alpha: float=0.3):
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
     
    def action_policy(s, epsilon):
        state_dist = [epsilon/len(Actions)]*len(Actions)
        max_Q = np.max(Q[s])
        count = np.sum(Q[s] == max_Q)
        
        if count > 1:
            indexes = [i for i in range(len(Actions)) if Q[s,i]== max_Q]
            action_idx = np.random.choice(indexes)
        else:
            action_idx = np.where(Q[s] == max_Q)[0].item()
        
        best_action = Actions(action_idx)
        state_dist[best_action]+= (1-epsilon)
        
        return state_dist
    
    for i in tqdm(range(num_episode), disable=False):
        s = model.start_state
        
        Q_old = np.copy(Q)
        for _ in range(maxit):
            num_iter[i]+=1
            a = eps_greedily(s, eps) 
            r = model.reward(s, a)
            possible_dict= model._possible_next_states_from_state_action(s, a)
            possible_s = list(possible_dict.keys())
            prob = list(possible_dict.values())
            s_new = np.random.choice(possible_s, p = prob)
            state_dist_next = action_policy(s_new, eps)
            E_a = sum([q * p_eps_greedy for q, p_eps_greedy in zip(Q[s_new], state_dist_next)])
            if s_new == model.goal_state:
                Q[s, a] = Q[s, a] + alpha*(r - Q[s, a])
            else:
                Q[s, a] = Q[s, a] + alpha*(r + model.gamma*E_a - Q[s, a])
            s = s_new
            if s == model.goal_state:
                break
        
        if np.all(abs(Q-Q_old)<=0.0001):
            print("breaking")
            break
        
    V = np.amax(Q, axis=1)
    pi = np.argmax(Q, axis=1)
    return V, pi, np.sum(num_iter)   