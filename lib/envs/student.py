# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 17:59:29 2019

@author: Admin
"""

import random

class Student():
    def __init__(self):
        #self.arrival_time =  random.randint(25200,68400)  #Getting random arrival time for student
        #self.departure_time =  random.randint(43200,86400)  #Getting random arrival time for student
        three_ints = random.sample(range(8), 1)
        self.preference = three_ints
        
#if __name__ == "__main__":
  #  students = []
  #  for i in range(100):
  #      students.append(Student())
  #  print(students[2].preference)