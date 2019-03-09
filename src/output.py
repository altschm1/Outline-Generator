import graph

def create_output(file_name, graph):
    f = open(file_name + ".html", "w+")
    f.write("<!DOCTYPE html>")
    f.write("<html><body><ul>")
    f.write(graph.to_html_structure(graph.root))
    f.write("</ul></body></html>")    
    f.close()

if __name__ == "__main__":
    graph = graph.Graph()
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
    