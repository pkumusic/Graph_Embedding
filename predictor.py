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

np.random.seed(2016)
floatX = theano.config.floatX

def predict(snapshot_filename, targets, typed_nodes, venue_info,
            batch_size=500, output_filename='output/predict', output_topk=5000):
    """
    targets: nx3 with each row = [int:source_index, int:source_node_type, int:target_node_type]
"""
    embeddings, all_param_values, dist_nn_sizes, dropout, _ = restore(snapshot_filename)

    # initialize embeddings
    print('Restoring embeddings ...')
    node_vecs = theano.shared(value=embeddings[0], name='node_vecs')
    node_vecs_c = theano.shared(value=embeddings[1], name='node_vecs_c')
    type_vecs = theano.shared(value=embeddings[2], name='type_vecs')
    type_vecs_c = theano.shared(value=embeddings[3], name='type_vecs_c')
    node_vecs_dim = embeddings[0].shape[1]
    type_vecs_dim = embeddings[2].shape[1]
    print("#nodes %i\t dims %i" % (embeddings[0].shape[0], node_vecs_dim))
    print("#types %i\t dims %i" % (embeddings[2].shape[0], type_vecs_dim))

    # build the distance neural network
    print('Restoring distance NN ...')
    x = T.imatrix('x')
    x_type_0 = T.iscalar('x_type_0')
    x_type_1 = T.iscalar('x_type_1')
    node_vec_input = T.concatenate(
                         [node_vecs[x[:,0]].reshape((x.shape[0], node_vecs.shape[1])),
                          node_vecs_c[x[:,1]].reshape((x.shape[0], node_vecs_c.shape[1]))],
                         axis=1)
    type_vec_input = T.concatenate([type_vecs[T.cast(x_type_0,dtype='int32')],
                                    type_vecs_c[T.cast(x_type_1,dtype='int32')]], axis=0)
    type_vec_input = T.extra_ops.repeat(type_vec_input.dimshuffle('x', 0), x.shape[0], axis=0)
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
    output_layer = lasagne.layers.DenseLayer(prev_layer, num_units=1, nonlinearity=nonlinearities.sigmoid)
    nn_output = lasagne.layers.get_output(output_layer, deterministic=True)
    # restore dist_nn weights
    all_params = lasagne.layers.get_all_params(output_layer)
    for p,v in zip(all_params, all_param_values):
        p.set_value(v)

    # compile theano functions
    predict_fn = theano.function([x, x_type_0, x_type_1], nn_output,
                                 allow_input_downcast=True)

    # transform data
    for k,v in typed_nodes.iteritems():
        typed_nodes[k] = np.expand_dims(np.array(v,dtype='int32'), axis=1)

    # create ouput file
    f = open(output_filename, 'w')

    # predict
    for i,target in enumerate(targets): 
        src_node, src_type, dst_type = target[0],target[1],target[2]
        print('%i\t%i\t%i' % (src_node, src_type, dst_type))
        results = np.array([])
        x_0 = np.ones([batch_size,1])*src_node
        dst_nodes = typed_nodes[dst_type]
        nbatches = int(dst_nodes.shape[0]/batch_size)
        for index in xrange(nbatches):
            pred = predict_fn(np.concatenate([x_0,dst_nodes[index*batch_size:(index+1)*batch_size]],axis=1),
                              src_type, dst_type)
            results = np.append(results, pred) 
        # the last batch
        if nbatches*batch_size < dst_nodes.shape[0]:
            size = dst_nodes.shape[0] - nbatches*batch_size
            x_0 = np.ones([size,1])*src_node
            pred = predict_fn(np.concatenate([x_0,dst_nodes[-size:]],axis=1),
                              src_type, dst_type)
            results = np.append(results, pred)
        # rank
        ranked_nodes = np.argsort(results)[::-1] # descending
        f.write('%i,%i,%i\n' % (src_node, src_type, dst_type))
        #for j in xrange(len(ranked_nodes)):

        # # For foursquare data
        # for j in xrange(output_topk):
        #     venue_index = dst_nodes[ranked_nodes[j]][0]
        #     f.write('%i\t%f\t%s\n' % (venue_index, results[ranked_nodes[j]], venue_info[venue_index]))

        # For general data
        f.write(','.join(str(dst_nodes[e])+':'+str(results[e]) for e in ranked_nodes[:output_topk]))
        f.write('\n')
    f.flush()
    f.close()
            

def restore(snapshot_filename):
    print 'restore from %s' % snapshot_filename
    x = cPickle.load(open(snapshot_filename,"rb"))
    embeddings, all_param_values, dist_nn_sizes, dropout, epoch = x[0],x[1],x[2],x[3],x[4]
    return embeddings, all_param_values, dist_nn_sizes, dropout, epoch


def read_venue_info(filename):
    venueIndex_info = {}
    with open(filename, "r") as f:
        for line in f:
            parts = line.split(',')
            venueIndex_info[int(parts[-1])] = ','.join(parts[1:-1])
    return venueIndex_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Graph embedding prediction configuration')
    args = parser.parse_args()

    # read data
    # dataset = 'foursquare'
    dataset = 'syn_event'
    data = cPickle.load(open("./data/%s/%s.binary.p" % (dataset, dataset), "rb"))
    typed_nodes = data[1]
    venueIndex_info = read_venue_info("./data/foursquare/venue_info.txt");

    # read targets to predict
    #targets=[[0,0,1],[1,0,1],[100,0,1],[1000,0,1],[3000,0,1]]
    targets=[[3,0,1],[2,0,1],[1,0,1],[0,0,1],[3,0,0],[4,1,2],[5,1,2],[6,1,2]]
    
    #snapshot_filename = 'output/foursquare_n30_t30_dnn50-10_b1000_lr0.000100_e20_it16653.model'
    #snapshot_filename = 'output/syn_event_n20_t10_dnn30-15_drp10_b2_lr0.010000-20-0.2_e175.model'
    snapshot_filename = 'output/syn_event_n9_t1_dnn30-15_drp10_b1_lr0.010000-20-0.2_e175.model'

    # predict
    predict(snapshot_filename=snapshot_filename,
            targets=targets,
            typed_nodes=typed_nodes,
            venue_info = venueIndex_info,
            output_filename=snapshot_filename+'.predict',
            output_topk=20)


