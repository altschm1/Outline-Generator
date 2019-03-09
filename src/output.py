"""
Michael Altschuler
Created: 3/9/2019
Updated: 3/9/2019

output.py
given a graph, create html file
"""

from data_structures import Graph

def create_output(file_name, graph):
    """create html file to represent graph
    
    Preconditions:
    file_name -- string -- name of html file to create
    graph -- Graph -- graph structure of sentences
    Postondition:
    returns Nothing
    creates html file in same directory 
    """
    f = open(file_name + ".html", "w+")
    f.write("<!DOCTYPE html>")
    f.write("<html><body><ul>")
    f.write(graph.to_html_structure(graph.root))
    f.write("</ul></body></html>")    
    f.close()

# TEST
if __name__ == "__main__":
    graph = Graph()
    graph.add_node("Root**")
    graph.add_node("Suptopic1*")
    graph.add_node("Suptopic2*")
    graph.add_node("sentence1")
    graph.add_node("sentence2")
    graph.add_node("sentence3")
    graph.add_node("sentence4")
    graph.add_node("sentence5")
    graph.add_node("sentence6")
    graph.add_node("sentence7")
    
    graph.set_root("Root**")
    graph.add_child_edges("Root**", "Suptopic1*")
    graph.add_child_edges("Root**", "Suptopic2*")
    graph.add_child_edges("Suptopic1*", "sentence1")
    graph.add_child_edges("Suptopic1*", "sentence2")
    graph.add_child_edges("Suptopic1*", "sentence3")
    graph.add_child_edges("Suptopic2*", "sentence4")
    graph.add_child_edges("Suptopic2*", "sentence5")
    graph.add_child_edges("Suptopic2*", "sentence6")
    graph.add_child_edges("Suptopic2*", "sentence7")
    
    create_output("example", graph)
    graph.print()
    
    # print(str(WebElement("p", "Hello World")))
    # print(str(WebElement("p", WebElement("p", "Hello World"))))    
    