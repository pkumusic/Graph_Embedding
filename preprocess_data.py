__author__="zhiting"

import numpy as np
import cPickle
import argparse


def preprocess_data(node_type_file, edge_file):
    # read node_type_file
    n_types = 0
    node_types = []
    typed_nodes = {}
    with open(node_type_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith('type'):
                n_types += 1
                cur_type = int(line.split()[1])
                typed_nodes[cur_type] = []
            else:
                u = int(line)
                node_types.append(cur_type)
                typed_nodes[cur_type].append(u)
    n_nodes = len(node_types)

    print "#node-types: " + str(n_types)
    print "#nodes: " + str(n_nodes)

    # read edge_file
    typed_dict_in = [{} for i in xrange(n_types)]
    typed_dict_out = [{} for i in xrange(n_types)]
    neighbors = [set() for i in xrange(n_nodes)]
    rev_neighbors = [set() for i in xrange(n_nodes)]
    typed_edges = [[] for i in xrange(n_types*n_types)]
    n_edges = 0
    with open(edge_file, "r") as f:
        for line in f:
            line = line.strip()
            parts = line.split()
            if line.startswith('type'):
                cur_type_0 = int(parts[1])
                cur_type_1 = int(parts[2])
                tp = cur_type_0*n_types+cur_type_1
                if cur_type_0 not in typed_dict_in[cur_type_1]:
                    next_index = len(typed_dict_in[cur_type_1])
                    typed_dict_in[cur_type_1][cur_type_0] = next_index
                if cur_type_1 not in typed_dict_out[cur_type_0]:
                    next_index = len(typed_dict_out[cur_type_0])
                    typed_dict_out[cur_type_0][cur_type_1] = next_index
            else:
                u = int(parts[0])
                v = int(parts[1])
                neighbors[u].add(v)
                rev_neighbors[v].add(u)
                typed_edges[tp].append([u,v])
                n_edges += 1
                
    # build data statistics
    type_pair_freqs = [0 for i in xrange(n_types*n_types)]
    for tp in xrange(n_types*n_types):
        type_pair_freqs[tp] = len(typed_edges[tp])
        typed_edges[tp] = np.array(typed_edges[tp], dtype='int32')
    typed_node_distr = [[] for i in xrange(n_types)]
    for t in xrange(n_types):
        typed_node_distr[t] = np.zeros([len(typed_nodes[t]),1+len(typed_dict_in[t])], dtype='int32')
        typed_node_distr[t][:,0] = typed_nodes[t]
        for i,u in enumerate(typed_nodes[t]):
            for v in rev_neighbors[u]: # edge (v,u)
                typed_node_distr[t][i,1+typed_dict_in[t][node_types[v]]] += 1

    print "neighbors"
    print neighbors
    print "typed_nodes"
    print typed_nodes
    print "typed_edges"
    print typed_edges
    print "type_pair_freqs"
    print type_pair_freqs
    print "typed_node_distr"
    print typed_node_distr
    print "typed_dict_in"
    print typed_dict_in
    print "typed_dict_out"
    print typed_dict_out
    print "n_nodes"
    print n_nodes
    print "n_types"
    print n_types
    print "n_edges"
    print n_edges

    assert len(neighbors) == n_nodes
    assert len(typed_edges) == n_types*n_types
    assert np.sum(type_pair_freqs) == n_edges

    data = [neighbors, typed_nodes, typed_edges, type_pair_freqs, typed_node_distr, typed_dict_in, typed_dict_out, n_nodes, n_types, n_edges]
    return data


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Pre-process node and edge data")
    parser.add_argument('-d', '--dataset', help='Give the dataset name under data folder', required="True")
    args = parser.parse_args()

    dataset = args.dataset
    #dataset = 'syn1'
    path = "./data/" + dataset
    node_type_file = '%s/node_types.txt' % path
    edge_file = '%s/edges.txt' % path
    data = preprocess_data(node_type_file, edge_file)
    cPickle.dump(data, open("%s/%s.binary.p" %(path,dataset), "wb"))

