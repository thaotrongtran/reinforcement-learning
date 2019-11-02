# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 17:52:11 2019

@author: Admin
"""
import numpy as np
#from student import Student
from collections import deque
import random
import hashlib
import numpy as np
import os
import random as _random
from six import integer_types
import struct
import sys

from gym import error

def np_random(seed=None):
    if seed is not None and not (isinstance(seed, integer_types) and 0 <= seed):
        raise error.Error('Seed must be a non-negative integer or omitted, not {}'.format(seed))

    seed = create_seed(seed)

    rng = np.random.RandomState()
    rng.seed(_int_list_from_bigint(hash_seed(seed)))
    return rng, seed

def hash_seed(seed=None, max_bytes=8):
    """Any given evaluation is likely to have many PRNG's active at
    once. (Most commonly, because the environment is running in
    multiple processes.) There's literature indicating that having
    linear correlations between seeds of multiple PRNG's can correlate
    the outputs:
    http://blogs.unity3d.com/2015/01/07/a-primer-on-repeatable-random-numbers/
    http://stackoverflow.com/questions/1554958/how-different-do-random-seeds-need-to-be
    http://dl.acm.org/citation.cfm?id=1276928
    Thus, for sanity we hash the seeds before using them. (This scheme
    is likely not crypto-strength, but it should be good enough to get
    rid of simple correlations.)
    Args:
        seed (Optional[int]): None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the hashed seed.
    """
    if seed is None:
        seed = create_seed(max_bytes=max_bytes)
    hash = hashlib.sha512(str(seed).encode('utf8')).digest()
    return _bigint_from_bytes(hash[:max_bytes])

def create_seed(a=None, max_bytes=8):
    """Create a strong random seed. Otherwise, Python 2 would seed using
    the system time, which might be non-robust especially in the
    presence of concurrency.
    Args:
        a (Optional[int, str]): None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the seed.
    """
    # Adapted from https://svn.python.org/projects/python/tags/r32/Lib/random.py
    if a is None:
        a = _bigint_from_bytes(os.urandom(max_bytes))
    elif isinstance(a, str):
        a = a.encode('utf8')
        a += hashlib.sha512(a).digest()
        a = _bigint_from_bytes(a[:max_bytes])
    elif isinstance(a, integer_types):
        a = a % 2**(8 * max_bytes)
    else:
        raise error.Error('Invalid type for seed: {} ({})'.format(type(a), a))

    return a

# TODO: don't hardcode sizeof_int here
def _bigint_from_bytes(bytes):
    sizeof_int = 4
    padding = sizeof_int - len(bytes) % sizeof_int
    bytes += b'\0' * padding
    int_count = int(len(bytes) / sizeof_int)
    unpacked = struct.unpack("{}I".format(int_count), bytes)
    accum = 0
    for i, val in enumerate(unpacked):
        accum += 2 ** (sizeof_int * 8 * i) * val
    return accum

def _int_list_from_bigint(bigint):
    # Special case 0
    if bigint < 0:
        raise error.Error('Seed must be non-negative, not {}'.format(bigint))
    elif bigint == 0:
        return [0]

    ints = []
    while bigint > 0:
        bigint, mod = divmod(bigint, 2 ** 32)
        ints.append(mod)
    return ints# -*- coding: utf-8 -*-

################################################################

def categorical_sample(prob_n, np_random):
    """
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

        
class CampusEnv():
    #Create students
    #Create road capacity
    #Create parking decks
    def _calculate_transition_prob(self, current, assigned_lot):
        current_state_as_list = current.split(",")
    
        is_done = int(current_state_as_list[0]) <= 0 and int(current_state_as_list[1]) <= 0 and int(current_state_as_list[2]) <= 0 and int(current_state_as_list[3]) <= 0 and int(current_state_as_list[4]) <= 0 and int(current_state_as_list[5]) <= 0
        #Case 1: not assignable
        temp = []
        if is_done:
            temp.append((1.0, current, 0, is_done))
        elif int(current_state_as_list[assigned_lot]) == 0:
            temp.append((1.0, current, -100, is_done))
        else:
            current_state_as_list[assigned_lot] = str(int(current_state_as_list[assigned_lot])  - 1)
            for i in range(3):
                for k in range(3):
                        if i !=k:
                            reward = 0
                            
                            #Students preferences reward
                            if assigned_lot == int(current_state_as_list[6]):
                                reward = reward + 200
                            elif assigned_lot == int(current_state_as_list[7]):
                                reward = reward + 100
                            else:
                                reward = reward + 20
                                
                            #Work on capacity reward
                            if int(current_state_as_list[assigned_lot]) < (self.lots[assigned_lot] * 0.1):
                                reward = reward - 10
                            
                            #if int(current_state_as_list[0]) <= 1:
                                #reward = reward - 2
                            
                            #if int(current_state_as_list[1]) <= 1:
                               # reward = reward - 2
                            
                            #if int(current_state_as_list[2]) <= 1:
                               # reward = reward - 2
                                
                            #if int(current_state_as_list[3]) <= 1:
                              #  reward = reward - 2
                                
                            #if int(current_state_as_list[4]) <= 1:
                              #  reward = reward - 2
                                
                           # if int(current_state_as_list[5]) <= 1:
                               # reward = reward - 2
                                
                            current_state_as_list[6] = str(i)
                            current_state_as_list[7] = str(k)
                            
                            str_2 = ","

                            is_done = int(current_state_as_list[0]) <= 0 and int(current_state_as_list[1]) <= 0 and int(current_state_as_list[2]) <= 0 and int(current_state_as_list[3]) <= 0 and int(current_state_as_list[4]) <= 0 and int(current_state_as_list[5]) <= 0
                            #If out of cars
                            temp.append((1/30,str_2.join(current_state_as_list),  reward, is_done))

        return temp
    
    def seed(self, seed=None):
        self.np_random, seed = np_random(seed)
        return [seed]
    
    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        #i = random.randint(0,len(transitions)-1)
        p, s, r, d= transitions[i]
        self.s = s
        self.lastaction = a
        return (s, r, d, {"prob" : p})

    def reset(self):
        self.s = self.states_list[categorical_sample(self.isd, self.np_random)]
        self.lastaction = None
        return self.s

    def __init__(self):
        self.nS = 4*5*6*6*5*4*6*5
        self.lots = [4,5,6,6,5,4]
        self.nA = 6
        self.states_list = []
        #create a list of all states
        for a in range(6):
            for b in range(6):
                    if a != b:
                        for d in range(4):
                            for e in range(5):
                                for f in range(6):
                                    for g in range(6):
                                        for h in range(5):
                                            for i in range(4):
                                                temp = str(d)+","+str(e)+","+str(f)+","+str(g)+","+str(h)+","+str(i)+","+str(a)+","+str(b)
                                                self.states_list.append(temp)
                                            
                                            #print(len(self.states_list))
                                    #print(temp)
        print("getting states list")
        P = {}
        i = 0
        for s in self.states_list:
            current = s
            P[s] = { a : [] for a in range(self.nA) }
            P[s][0] = self._calculate_transition_prob(current, 0)
            P[s][1] = self._calculate_transition_prob(current, 1)
            P[s][2] = self._calculate_transition_prob(current, 2)
            P[s][3] = self._calculate_transition_prob(current, 3)
            P[s][4] = self._calculate_transition_prob(current, 4)
            P[s][5] = self._calculate_transition_prob(current, 5)
           
        
        
        self.P = P
        isd = np.full((1, self.nS), 1/self.nS).ravel()
        #isd = np.zeros(self.nS)
        #isd[] = 1.0  #Selecting the state at last and the first state
        self.isd = isd
        self.lastaction = None # for rendering
        self.seed()
        self.s = self.states_list[categorical_sample(self.isd, self.np_random)]
        print("finish init")
     
        
       
if __name__ == "__main__":
        campus = CampusEnv()