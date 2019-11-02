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
import copy
#from student import Student

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
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

        
class CampusEnv():
    #Just the transition not the actual stepping on step
    def _calculate_transition_prob(self, a):
        current = copy.deepcopy(self.s) #Get a copy of the current state
        return_transitions = []
        is_done = False;
        if (current[a] > 0):
            reward = 0;
            current[a] -= 1  #Assign the car there
            if(a == current[-1]):
                reward += 100
            else:
                reward += 5
            #get new student in
            if len(self.students) > 1:
                next_student = self.students[1]  #Get the next in line student
                state = np.append(current[0:8], next_student.preference)
            else:
                is_done = True
                state = np.append(current[0:8], [-1])  #Append dummy value to the next state with no student
            return_transitions.append((1.0, state, reward, is_done)) 
        else: #If the chosen lot is full, other lot can take it
            #indexes = np.where(current[0:8] > 0)[0]    
           # if len(indexes) > 0:
             #   probability = 1/len(indexes)
            #    for i in indexes:
               #     reward = 0;
                #    temp = copy.deepcopy(current)
               #     temp[i] -= 1 #Decrease capacity by one
                #    if(i == current[-1]): #Check reward
                #        reward += 10
                #    else:
                #        reward += 1
                #get new student in
                #    if len(self.students) > 1:
                 #       next_student = self.students[1]  #Get the next in line student
                 #       state = np.append(temp[0:8], next_student.preference)
                 #   else:
                 #       state = np.append(temp[0:8], [-1]) #Append dummy value to the next state with no student
                 #       is_done = True;
                 #   return_transitions.append((probability, state, reward, is_done)) 
            #else:
               # return_transitions.append((1.0, current, 0, True)) #If nothing else is available
            return_transitions.append((1.0, current, 0, is_done)) 
        return return_transitions
    
    def seed(self, seed=None):
        self.np_random, seed = np_random(seed)
        return [seed]
    
    def step(self, a):
        transitions = self._calculate_transition_prob(a)
        i = categorical_sample([t[0] for t in transitions], self.np_random) 
        p, s, r, d= transitions[i]
        #print(self.s, "old state")
        
        if( not np.array_equal(self.s,s )):
            self.s = s  #Actually making the transition
            self.lots = s[0:8] #Change the lots
            if(len(self.students) > 0):
                self.students.pop()
        
        #print(self.s, "new state")
        #self.lastaction = a
        return (s, r, d, {"prob" : p})

    def reset(self):
        self.lots = [3,4,5,6,5,4,3,2]
        self.students = copy.deepcopy(self.original_students)  #Reseting to original students
        #print(len(self.original_students))
        self.current_student = self.students.pop() #Pop a new student
        self.s = np.append(self.lots, self.current_student.preference)
        return self.s

    def __init__(self):
        #Initialize lots capacity
        self.lots = [3,4,5,6,5,4,3,2]
        
        students = []
        for i in range(28):
            students.append(Student())
        self.students = copy.deepcopy(students)
        self.original_students = copy.deepcopy(students)
        
        product = 1
        for i in self.lots:
            product *= i
        
        #Calculate number of states
        self.nS = product*len(self.students)
        self.nA = 8
        self.states_list = []
        #create a list of all states
        
        #for a in range(8):
           # for b in range(8):
                #    if a != b:
                   #     for d in range(3):
                     #       for e in range(3):
                      #          for f in range(3):
                      #              for g in range(3):
                            #            for h in range(3):
                                   #         for i in range(3):
                                      #          for k in range(3):
                                    #                for l in range(3):
                                           #             temp = str(d)+str(e)+str(f)+str(g)+str(h)+str(i)+str(k)+str(l)+str(a)+str(b)
                                           #             self.states_list.append(temp)
                                                        #print(temp)
       # P = {}
        #i = 0
        #for s in self.states_list:
          #  current = s
         #  P[s] = { a : [] for a in range(self.nA) }
            #P[s][0] = self._calculate_transition_prob(current, 0)
           # P[s][1] = self._calculate_transition_prob(current, 1)
            #P[s][2] = self._calculate_transition_prob(current, 2)
            #P[s][3] = self._calculate_transition_prob(current, 3)
           # P[s][4] = self._calculate_transition_prob(current, 4)
           # P[s][5] = self._calculate_transition_prob(current, 5)
           # P[s][6] = self._calculate_transition_prob(current, 6)
           # P[s][7] = self._calculate_transition_prob(current, 7)
#            print(i)
           # i+= 1
        
       # self.P = P
        #isd = np.zeros(self.nS)
        #isd[-1] = 1.0  #Selecting the state at last and the first state
        #self.isd = isd
        self.lastaction = None # for rendering
        self.seed()
        self.current_student = self.students.pop()
        self.s = np.append(self.lots, self.current_student.preference)
     
if __name__ == "__main__":
        campus = CampusEnv()
        print(campus.s)
        
class Student():
    def __init__(self):
        #self.arrival_time =  random.randint(25200,68400)  #Getting random arrival time for student
        #self.departure_time =  random.randint(43200,86400)  #Getting random arrival time for student
        three_ints = random.sample(range(8), 1)
        self.preference = three_ints