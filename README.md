# Data preprocess
![Design Document](../syn_venue.png =400x200)

Targets point to similar context are similar.
Here:

1. Venues which belong to similar/same category are similar
2. Users who go to similar venues are similar
3. Users follow similar users are similar

3 and 0 follow same users. 4 and 5 belongs to same category. Since 0 goes to 4 and 5, 3 goes to 5, 3 are more likely to go to 4 than go to 6.

!? How to define target and context pairs? one-to-many, many-to-one?

####neighbors
neighbors[i] is a set of context nodes for node i.

e.g.,neighbors[0] = set([1, 2, 4, 5])
####typed_nodes
{0: [0, 1, 2, 3], 1: [4, 5, 6], 2: [7, 8]}
####typed_edges
tp = cur_type_0*n_types+cur_type_1

typed_edges[tp] contains edges from type_0 to type_1
####type_pair_freqs
typed_pair_freqs[i] = number of edges for this edge type
####typed_dict_in
types points to one type

typed_dict_in[context_type] = {target_type:index_start_from_0}
####typed_dict_out
types pointed by one type

typed_dict_in[target_type] = {context_type:index_start_from_0}
####typed_node_distr
how many nodes point to a context node.
####n_nodes
number of nodes
####n_types
number of types
####n_edges
number of edges

# Training
Learning distance(similarity) metric by DNN.
The objective function is the same as in Skip-gram model. Try to maximize the average log probability of seeing a context node given a target node. Loss function is reformalized by noise-construction (negative sampling). 
For the DNN, now the input is the concatenation of vectors of (target_node, context_node, target_type, context_type), output is the simlarity score.

Parameters:

* batch_size: number of batches.
