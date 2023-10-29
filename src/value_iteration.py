from cgitb import small
from typing import Callable
from matplotlib.pyplot import grid
from tqdm import tqdm
import numpy as np

from model import Model, Actions


def value_iteration(model: Model, maxit: int = 100, tol: float=0.01):

    V = np.zeros((model.num_states,))
    pi = np.zeros((model.num_states,))
    mae = []

    def compute_value(s, a, reward: Callable):
        return np.sum(
            [
                model.transition_probability(s, s_, a)
                * (reward(s, a) + model.gamma * V[s_])
                for s_ in model.states
            ]
        )
    
    def value_update():
        for s in model.states:
            action_val = np.max(
                [compute_value(s, a, model.reward) for a in Actions]
            )
            V[s] = action_val

#     def policy_evaluation():
#         for s in model.states:
#             R = model.reward(s, pi[s])
#             V[s] = compute_value(s, pi[s], lambda *_: R)

#     def policy_improvement():
#         for s in model.states:
#             action_index = np.argmax(
#                 [compute_value(s, a, model.reward) for a in Actions]
#             )
#             pi[s] = Actions(action_index)

    for i in tqdm(range(maxit)):
#         for _ in range(5):
#             policy_evaluation()
        V_old = np.copy(V)
        value_update()
        mae.append((abs(V - V_old)).mean())
        if  all(abs(V - V_old) <= tol):
            print("breaking")
            break
            
    for s in model.states:
        action_index = np.argmax(
            [compute_value(s, a, model.reward) for a in Actions]
            )
        pi[s] = Actions(action_index)

    return V, pi, mae