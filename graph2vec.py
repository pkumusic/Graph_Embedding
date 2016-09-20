__author__ = 'zhiting'

import time
import sys
import argparse
import numpy as np
import theano.tensor as T
import theano
import lasagne
import lasagne.nonlinearities as nonlinearities
import cPickle
from scipy import stats
from collections import OrderedDict

np.random.seed(2016)
floatX = theano.config.floatX

#theano.config.compute_test_value = 'warn'

def train_graph2vec(
        data,
        node_vecs_values=None, node_vecs_c_values=None, type_vecs_values=None, type_vecs_c_values=None,
        node_vecs_dim=100, type_vecs_dim=200,
        dist_nn_sizes=[500,100], dropout=[0,0],
        n_negs=5, n_epochs=10, batch_size=50,
        update_alg='momentum', momentum=0.9, learning_rate=0.1,
        epoch_step_size=20, anneal_factor=0.2,
        display_iter=100, snapshot_epoch=10, snapshot_prefix=''):
    rng = np.random.RandomState(201671)

    snapshot_prefix = '%s_n%i_t%i_dnn%s_drp%s_b%i_lr%f-%i-%0.1f' % (snapshot_prefix, node_vecs_dim, type_vecs_dim, 
                                                     '-'.join(str(e) for e in dist_nn_sizes), ''.join(str(e) for e in dropout),
                                                     batch_size, learning_rate, epoch_step_size, anneal_factor)
    neighbors, typed_nodes, typed_edges, type_pair_freqs, typed_node_distr, typed_dict_in, typed_dict_out, n_nodes, n_types, n_edges = data
    print("#nodes %i\n#types %i\n#edges %i" % (n_nodes, n_types, n_edges))

    # initialize embeddings
    print('Initializing ...')
    if node_vecs_values is None:            
        node_vecs_values = np.asarray(rng.normal(loc=0., scale=.01,
                                      size=(n_nodes, node_vecs_dim)), dtype=floatX)

    else:
        assert node_vecs_values.shape[1] == node_vecs_dim
    print "node_vecs_dim:", node_vecs_values.shape
    # exit()
    node_vecs = theano.shared(value=node_vecs_values, name='node_vecs')
    node_vecs_c = theano.shared(value=node_vecs_values, name='node_vecs_c')
    if type_vecs_values is None:
        type_vecs_values = np.asarray(rng.normal(loc=0., scale=.01,
                                      size=(n_types, type_vecs_dim)), dtype=floatX)
    else:
        assert type_vecs_values.shape[1] == type_vecs_dim
    type_vecs = theano.shared(value=type_vecs_values, name='type_vecs')
    type_vecs_c = theano.shared(value=type_vecs_values, name='type_vecs_c')
    print "type_vecs_dim:", type_vecs_values.shape

    # build the distance neural network
    x = T.imatrix('x') # batch_size x 2 : starting_node_index, end_node_index
    x_neg = T.imatrix('x_neg') # negative samples
    x_type_0 = T.iscalar('x_type_0')
    x_type_1 = T.iscalar('x_type_1')
    x_comb = T.concatenate([x,x_neg],axis=0)
    node_vec_input = T.concatenate([node_vecs[x_comb[:,0]],node_vecs_c[x_comb[:,1]]],axis=1)
    type_vec_input = T.concatenate([type_vecs[x_type_0], type_vecs_c[x_type_1]])  # a row. shape: (n,)
    type_vec_input = type_vec_input.dimshuffle('x', 0)
    type_vec_input = type_vec_input.repeat(x_comb.shape[0], axis=0)  # dimshuffle('x', 0)-> make (n,) to (1,n)
    if type_vecs_dim == 0:
        print "---- No Type Vector ----"
        vecs_input = node_vec_input
    else:
        vecs_input = T.concatenate([node_vec_input, type_vec_input], axis=1)
    vecs_input_dim = (node_vecs_dim+type_vecs_dim)*2
    input_layer = lasagne.layers.InputLayer(shape=(None,vecs_input_dim), input_var=vecs_input, name='input_layer')
    prev_layer = input_layer
    for lid,size in enumerate(dist_nn_sizes):
        if dropout[lid] == 1:
            prev_layer = lasagne.layers.DropoutLayer(prev_layer, p=0.5)
        layer = lasagne.layers.DenseLayer(prev_layer, num_units=size,
                                          nonlinearity=nonlinearities.rectify, name='dense_layer_%d'%lid)
        prev_layer = layer
    output_layer = lasagne.layers.DenseLayer(prev_layer, num_units=1, nonlinearity=nonlinearities.identity)
    nn_output = lasagne.layers.get_output(output_layer, deterministic=False)
    # output is similarity between two nodes

    # training loss
    print('Building theano functions ...')
    ones = T.ones([x.shape[0]])
    ones_neg = -1*T.ones([x_neg.shape[0]])
    ones_comb = T.concatenate([ones,ones_neg],axis=0)
    #loss_train = -1*T.mean(ones_comb.dimshuffle(0,'x')*T.log(nn_output))
    loss_train = -1*T.mean(T.log(T.nnet.sigmoid(ones_comb.dimshuffle(0,'x')*nn_output)))
    # training routines
    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    if type_vecs_dim == 0:
        params += [node_vecs, node_vecs_c]
    else:
        params += [node_vecs, node_vecs_c, type_vecs, type_vecs_c]
    updates = create_updates(loss_train, params, update_alg, learning_rate=learning_rate, momentum=momentum)
    train_fn = theano.function([x, x_neg, x_type_0, x_type_1], loss_train,
                               updates=updates, on_unused_input='ignore')
    normalize_updates = create_normalization(node_vecs,node_vecs_c,type_vecs,type_vecs_c)
    normalize_fn = theano.function(inputs=[], updates=normalize_updates)

    # data
    print('Preparing data ...')
    # (padded) number of batchs per epoch of each type pair
    type_pair_freqs = np.array(type_pair_freqs, dtype='int32')
    #print 'type_pair_freqs', type_pair_freqs
    type_pair_n_batchs = np.ceil(type_pair_freqs*1./batch_size).astype(int)
    #print 'type_pair_n_batchs', type_pair_n_batchs
    # pad and shuffle edges
    for tp in xrange(len(typed_edges)):
        extra_data_num = type_pair_n_batchs[tp]*batch_size-type_pair_freqs[tp]
        if extra_data_num > 0:
            rand_indices = np.random.permutation(np.arange(type_pair_freqs[tp]))
            temp_shuffled_data = typed_edges[tp][rand_indices] 
            #extra_data = typed_edges[tp][:extra_data_num]
            extra_data = temp_shuffled_data[:extra_data_num] # debugged by music
            typed_edges[tp] = np.append(typed_edges[tp],extra_data,axis=0)
        # shuffle
        rand_indices = np.random.permutation(np.arange(len(typed_edges[tp])))
        typed_edges[tp] = typed_edges[tp][rand_indices]
    # construct negative sampling distribution
    typed_node_distr = build_neg_distr(typed_node_distr)

    # train
    print('Training ...')
    print 'lr %f' % learning_rate
    biter = 0
    for epoch in xrange(n_epochs):
        loss = 0
        tp_batch_index = [0]*len(typed_edges)
        for tp in type_pair_iterator(type_pair_n_batchs):
            type_0 = int(tp / n_types)
            type_1 = tp % n_types
            index = tp_batch_index[tp]
            tp_batch_index[tp] += 1
            edge_batch = typed_edges[tp][index*batch_size:(index+1)*batch_size]
            # negative sampling
            node_distr = [typed_node_distr[type_1][:,0],
                          typed_node_distr[type_1][:,1+typed_dict_in[type_1][type_0]]]
            neg_samples = sample_negative(edge_batch, neighbors, node_distr, n_negs)
            # update
            loss = train_fn(edge_batch, neg_samples, type_0, type_1)
            normalize_fn()
            ##
            # display

            biter += 1
            if biter % display_iter == 0:
                print('epoch: %i, iter: %i, types %i-%i, loss %f' % (epoch, biter, type_0, type_1, loss))
        # snapshot
        if epoch>0 and epoch % snapshot_epoch == 0:
            snapshot(snapshot_prefix, 
                     [node_vecs.get_value(), node_vecs_c.get_value(), type_vecs.get_value(), type_vecs_c.get_value()],
                     output_layer, dist_nn_sizes, dropout, epoch)
        # update learning rate
        learning_rate, lr_updated = get_learning_rate(learning_rate, epoch, epoch_step_size, anneal_factor)
        if lr_updated:
            updates = create_updates(loss_train, params, update_alg, learning_rate=learning_rate, momentum=momentum)
            train_fn = theano.function([x, x_neg, x_type_0, x_type_1], loss_train,
                                       updates=updates, on_unused_input='ignore')
        print('epoch: %i, loss %f' % (epoch, loss))


#def sample_negative(edge_batch, neighbors, node_distr, n_negs):
#    neg_samples = []
#    #n_cand_nodes = node_distr[0].shape[0]
#    #neg_proposal = stats.rv_discrete(name='neg_proposal', 
#    #                                 values=(node_distr[0], node_distr[1]))
#    for i in xrange(edge_batch.shape[0]):
#        u = edge_batch[i,0]
#        u_neighbors = neighbors[u]
#        n_u_negs = 0
#        while (n_u_negs < n_negs):
#            #s_v = neg_proposal.rvs(size=1)[0]
#            s_v = np.random.choice(node_distr[0], 1, p=node_distr[1])[0]
#            if (s_v not in u_neighbors) and s_v != u:
#                neg_samples.append([u,s_v])
#                n_u_negs += 1
#    return np.array(neg_samples, dtype='int32')


def sample_negative(edge_batch, neighbors, node_distr, n_negs):
    neg_samples = []
    neg_sample_cand_size = edge_batch.shape[0] * n_negs * 10
    neg_sample_cands = np.random.choice(node_distr[0], neg_sample_cand_size, p=node_distr[1])
    neg_sample_cand_index = 0
    for i in xrange(edge_batch.shape[0]):
        u = edge_batch[i,0]
        u_neighbors = neighbors[u]
        n_u_negs = 0
        while (n_u_negs < n_negs):
            if neg_sample_cand_index >= neg_sample_cand_size:
                neg_sample_cands = np.random.choice(node_distr[0], neg_sample_cand_size, p=node_distr[1])
                neg_sample_cand_index = 0
            s_v = neg_sample_cands[neg_sample_cand_index]
            neg_sample_cand_index += 1 
            if (s_v not in u_neighbors) and s_v != u:
                neg_samples.append([u,s_v])
                n_u_negs += 1
    return np.array(neg_samples, dtype='int32')


def build_neg_distr(typed_node_distr):
    #print typed_node_distr
    for t in xrange(len(typed_node_distr)):
        node_distr = typed_node_distr[t]
        nodes = node_distr[:,0]
        node_distr = (node_distr+0.1)**0.75 # smoothing #TODO: smoothing parameter
        sums = node_distr.sum(axis=0)
        node_distr /= sums[np.newaxis, :]
        node_distr[:,0] = nodes
        typed_node_distr[t] = node_distr
    #print typed_node_distr
    return typed_node_distr

def type_pair_iterator(type_pair_n_batchs):
    type_pair_queue = []
    for tp,n_batchs in enumerate(type_pair_n_batchs):
        type_pair_queue += [tp]*n_batchs
    type_pair_queue = np.array(type_pair_queue) 
    n_batchs = type_pair_queue.shape[0]
    # shuffle
    shuffled_type_pair_queue = type_pair_queue[np.random.permutation(np.arange(n_batchs))] 
    for i,t in enumerate(shuffled_type_pair_queue):
        yield t

def create_updates(loss, params, update_alg, learning_rate, momentum=None):
    """
    create updates for training
    :param loss: loss for gradient
    :param params: parameters for update
    :param update_alg: update algorithm
    :param learning_rate: learning rate
    :param momentum: momentum
    :return: updates
    """

    if update_alg == 'sgd':
        return lasagne.updates.sgd(loss, params=params, learning_rate=learning_rate)
    elif update_alg == 'momentum':
        return lasagne.updates.momentum(loss, params=params, learning_rate=learning_rate, momentum=momentum)
    elif update_alg == 'nesterov':
        return lasagne.updates.nesterov_momentum(loss, params=params, learning_rate=learning_rate, momentum=momentum)
    elif update_alg == 'adadelta':
        return lasagne.updates.adadelta(loss, params=params)
    else:
        raise ValueError('Unkown update algorithm: %s' % update_alg)


def create_normalization(node_vecs,node_vecs_c,type_vecs,type_vecs_c):
    """
    projects embeddings to the L2 unit ball
"""
    updates = OrderedDict()
    updates[node_vecs] = node_vecs / T.sqrt(T.sum(node_vecs**2,axis=1,keepdims=True))
    updates[node_vecs_c] = node_vecs_c / T.sqrt(T.sum(node_vecs_c**2,axis=1,keepdims=True))
    updates[type_vecs] = type_vecs / T.sqrt(T.sum(type_vecs**2,axis=1,keepdims=True))
    updates[type_vecs_c] = type_vecs_c / T.sqrt(T.sum(type_vecs_c**2,axis=1,keepdims=True))
    return updates 

def snapshot(snapshot_prefix, embeddings, dist_nn, dist_nn_sizes, dropout, epoch):
    snapshot_filename = '%s_e%i.model' % (snapshot_prefix, epoch)
    all_params = lasagne.layers.get_all_params(dist_nn)
    all_param_values = [p.get_value() for p in all_params]
    print 'snapshot to %s' % snapshot_filename
    cPickle.dump([embeddings, all_param_values, dist_nn_sizes, dropout, epoch],
                 open(snapshot_filename, "wb"))

def get_learning_rate(learning_rate, epoch, epoch_step_size, anneal_factor):
    updated = False
    if epoch > 0 and epoch%epoch_step_size==0:
        learning_rate = learning_rate * anneal_factor
        print 'lr %f' % learning_rate
        updated = True
    return learning_rate, updated

def inspect_inputs(i, node, fn):
    print i, node, "input(s) value(s):", [input[0] for input in fn.inputs]

def inspect_outputs(i, node, fn):
    print " output(s) value(s):", [output[0] for output in fn.outputs]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Graph embedding configuration')
    parser.add_argument('-d', '--dataset', help='Give the dataset name under data folder', required="True")
    #parser.add_argument('--fine_tune', action='store_true', help='Fine tune the word embeddings')
    #parser.add_argument('--embedding', choices=['word2vec', 'glove', 'senna'], help='Embedding for words',
    #                    required=True)
    #parser.add_argument('--embedding_dict', default='data/word2vec/GoogleNews-vectors-negative300.bin',
    #                    help='path for embedding dict')
    #parser.add_argument('--batch_size', type=int, default=10, help='Number of sentences in each batch')

    args = parser.parse_args()
    #logger = utils.get_logger("graph-embed")
    #dataset = 'foursquare'

    dataset = args.dataset
    data = cPickle.load(open("./data/%s/%s.binary.p"%(dataset,dataset),"rb"))
    # train_graph2vec(
    #     data=data,
    #     node_vecs_dim=30, type_vecs_dim=30,
    #     dist_nn_sizes=[200,50], dropout=[1,0],
    #     n_negs=5, n_epochs=501, batch_size=1000,
    #     update_alg='momentum', momentum=0.9, learning_rate=0.001,
    #     epoch_step_size=100, anneal_factor=0.2,
    #     display_iter=200, snapshot_epoch=50, snapshot_prefix='output/foursquare')
    train_graph2vec(
        data=data,
        node_vecs_values=None, node_vecs_c_values=None, type_vecs_values=None, type_vecs_c_values=None,
        node_vecs_dim=10, type_vecs_dim=0,
        dist_nn_sizes=[30, 15], dropout=[1, 0],
        n_negs=2, n_epochs=400, batch_size=1,
        update_alg='momentum', momentum=0.9, learning_rate=0.01,
        epoch_step_size=20, anneal_factor=0.2,
        display_iter=100, snapshot_epoch=50, snapshot_prefix='output/%s'%dataset)

