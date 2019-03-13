from dipy.core.graph import Graph

from numpy.testing import assert_equal


def test_graph():

    g = Graph()

    g.add_node('a', 5)
    g.add_node('b', 6)
    g.add_node('c', 10)
    g.add_node('d', 11)

    g.add_edge('a', 'b')
    g.add_edge('b', 'c')
    g.add_edge('c', 'd')
    g.add_edge('b', 'd')

    print('Nodes')
    print(g.node)
    print('Successors')
    print(g.succ)
    print('Predecessors')
    print(g.pred)
    print('Paths above d')
    print(g.up('d'))
    print('Paths below a')
    print(g.down('a'))
    print('Shortest path above d')
    print(g.up_short('d'))
    print('Shortest path below a')
    print(g.down_short('a'))
    print('Deleting node b')
    # g.del_node_and_edges('b')
    g.del_node('b')
    print('Nodes')
    print(g.node)
    print('Successors')
    print(g.succ)
    print('Predecessors')
    print(g.pred)

    assert_equal(len(g.node), 3)
    assert_equal(len(g.succ), 3)
    assert_equal(len(g.pred), 3)
