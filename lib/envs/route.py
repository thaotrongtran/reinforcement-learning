# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 17:59:29 2019

@author: Admin
"""

import random
import numpy as np
from graph import Graph

class Route():
    def __init__(self):
        g = Graph()
        
        for i in range(22):  
            g.add_vertex(i)

        g.add_edge(1, 2, 0.1)
        g.add_edge(3, 2, 0.2)
        g.add_edge(3, 4, 0.4)
        g.add_edge(4, 5, 0.066)
        g.add_edge(6, 5, 0.2)
        g.add_edge(6, 7, 0.3)
        g.add_edge(3, 7, 0.2)
        g.add_edge(15, 7, 0.2)

        self.graph = g

        
        

if __name__ == '__main__':
    route = Route()
    for v in route.graph:
        for w in v.get_connections():
            vid = v.get_id()
            wid = w.get_id()
            print ( '%s , %s, %.1f'  % ( vid, wid, v.get_weight(w)))

    for v in route.graph:
        print ('g.vert_dict[%s]=%s' %(v.get_id(), route.graph.vert_dict[v.get_id()]))