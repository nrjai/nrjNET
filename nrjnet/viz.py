"""
viz.py - draw a Value's computation graph with graphviz.

Optional: needs `pip install graphviz` plus the Graphviz app (https://graphviz.org/download/).
The rest of nrjNET works without it.
"""


def trace(root):
    """Collect every node and edge in the graph that ends at `root`."""
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def draw_dot(root, rankdir='LR'):
    """Return a graphviz Digraph of the computation graph (inputs on the left, output on the right)."""
    from graphviz import Digraph  # imported here so nrjnet works without graphviz installed

    dot = Digraph(format='svg', graph_attr={'rankdir': rankdir})
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # A rectangle for every Value: label | data | grad
        dot.node(name=uid, label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            # A small node for the operation that produced this Value
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot
