"""
Michael Altschuler
Created: 3/9/2019
Updated: 3/9/2019

data_structures.py
contains data structures that help the program run
Graph
Table


Pre-compilation:
pip install numpy
"""

import numpy as np

class Graph:
    """Graph data structure
    """
    def __init__(self):
        """constructor
        root -- root of the graph
        nodes -- list of nodes in the graph
        child_edges -- dictionary where key is parent node and value is list of children
        """
        self.root = None
        self.nodes = []
        self.child_edges = {}
    
    def set_root(self, root):
        """set the root of the graph"""
        self.root = root
    
    def add_node(self, node):
        """add a node to the graph"""
        self.nodes.append(node)
        self.child_edges[node] = []
    
    def remove_node(self, node):
        """remove a node from the graph"""
        self.nodes.remove(node)
        del self.child_edges[node]
    
    def add_child_edges(self, node, child):
        """add an edge from node (parent) to child (child)"""
        self.child_edges[node].append(child)
    
    def remove_child_edges(self, node):
        """remove all edges from a node"""
        self.child_edges[node] = []
    
    def print(self):
        """BFS search to print out all nodes"""
        queue = []
        queue.append(self.root)
        while len(queue) != 0:
            node = queue.pop(0)
            print(node + " with children " + str(self.child_edges[node]))
            for c in self.child_edges[node]:
                queue.append(c)

    def dfs_print(self):
        """DFS search to print out all nodes"""
        queue = []
        queue.append(self.root)
        while len(queue) != 0:
            node = queue.pop()
            print(node + " with children " + str(self.child_edges[node]))
            for c in self.child_edges[node]:
                queue.append(c)
    
    def to_html_structure(self, node):
        """print out content of graph into nested list html structure"""
        result = ""
        result = result + self.write_node(node)
        result = result + "<ul>"
        for child in self.child_edges[node]:
            result = result + self.to_html_structure(child)
        result = result + "</ul>"
        
        return result
    
    def write_node(self, node):
        """return node as a list item"""
        return "<li>" + node + "</li>"
        
        

class Table:
    """Hold a list of tuples containing sentence and corresponding vector"""
    
    def __init__(self, sentences, vectors):
        self.table = []
        for s, v in zip(sentences, vectors):
            self.table.append((s, v))
    
    def add_pair(self, sentence, vector):
        """add a sentence-vector pair to the table"""
        self.table.append((sentence, vector))
    
    def retrieve_vector(self, sentence):
        """given a vector, return a sentence"""
        for element in self.table:
            if element[0] == sentence:
                return element[1]
        return None
    
    def retrieve_sentence(self, vector):
        """given a sentence, return a vector"""
        for element in self.table:
            if np.array_equal(vector, element[1]):
                return element[0]
        return None
    