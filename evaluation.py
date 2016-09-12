from __future__ import division
__author__ = 'Music'

# Given the model, evaluate the Accuracy@N defined in the paper

from collections import defaultdict
import cPickle
import bisect
import theano
import theano.tensor as T
import numpy as np
import lasagne
import lasagne.nonlinearities as nonlinearities

def train_test_split(checkin_file, training_percentage = 0.8):
    """ In the paper, we choose data before 80% time for each user as training data
        , remaining 20% as test data
    :return:
    """
    # All data: {user:[venue,venue,venue...]}
    all_data = defaultdict(list)
    count = 0
    f = open(checkin_file, 'r')
    for line in f:
        if count == 0:
            count += 1
            continue
        count += 1
        userID, time, venueID, venue_name, venueLocation, venueCategory = line.strip().split('\t')
        all_data[userID].append(venueID)
    for userID in all_data:
        all_data[userID].reverse() # Now it's in forward time order

    test_count = 0
    training_data = {}
    test_data = {}
    for userID in all_data:
        cutoff = int(len(all_data[userID]) * training_percentage)
        training_data[userID] = all_data[userID][:cutoff]
        test_data[userID] = all_data[userID][cutoff:]
        test_count += len(test_data[userID])

    print '#test %i' % test_count

    return training_data, test_data


def read_mappings(data_path):
    venueID_index = {}
    with open(data_path+"/venueID_index.txt", "r") as f:
        for line in f:
            parts = line.strip().split(',')
            venueID_index[parts[0]] = int(parts[1])
    userID_index = {}
    with open(data_path+"/userID_index.txt", "r") as f:
        for line in f:
            parts = line.strip().split(',')
            userID_index[parts[0]] = int(parts[1])
    return venueID_index, userID_index


def build_predictor(snapshot_filename):
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

    return predict_fn


def accuracy_at_N(test_data, location_circle, Ns=[1],
                  snapshot_filename="", data_path="data/", batch_size=1000):
    predictor = build_predictor(snapshot_filename)
    venueID_index, userID_index = read_mappings(data_path)
    src_type = 0 #user
    dst_type = 1 #venue 

    # predict
    total_num = 0
    total_correct = np.array([0]*len(Ns),dtype='float32')
    for userID, venueIDs in test_data.iteritems():
        for venueID in venueIDs:
            total_num += 1
            possible_venues = location_circle[venueID]
            # transform data
            venue_indexes = [venueID_index[venueID]] # the 0-th element is the target
            venue_indexes += [venueID_index[e] for e in possible_venues]
            venue_indexes = np.expand_dims(np.array(venue_indexes,dtype='int32'), axis=1)
            x_0 = np.ones([batch_size,1])*userID_index[userID]
            # predict by minibatches
            results = np.array([])
            nbatches = int(venue_indexes.shape[0]/batch_size)
            for index in xrange(nbatches):
                pred = predictor(np.concatenate([x_0,venue_indexes[index*batch_size:(index+1)*batch_size]],axis=1),
                                  src_type, dst_type)
                #pred = np.random.randn(batch_size,1)
                results = np.append(results, pred) 
            # the last batch
            if nbatches*batch_size < venue_indexes.shape[0]:
                size = venue_indexes.shape[0] - nbatches*batch_size
                x_0 = np.ones([size,1])*userID_index[userID]
                pred = predictor(np.concatenate([x_0,venue_indexes[-size:]],axis=1),
                                  src_type, dst_type)
                #pred = np.random.randn(size,1)
                results = np.append(results, pred)
            # rank
            ranklist = np.argsort(results)[::-1] # descending
            rank = np.where(ranklist==0)[0][0]
            for i,N in enumerate(Ns):
                if rank <= N:
                    total_correct[i] += 1
            if total_num % 100 == 0:
                print "%i\t" % total_num, 
                print total_correct/total_num

    return total_correct/total_num


def restore(snapshot_filename):
    print 'restore from %s' % snapshot_filename
    x = cPickle.load(open(snapshot_filename,"rb"))
    embeddings, all_param_values, dist_nn_sizes, dropout, epoch = x[0],x[1],x[2],x[3],x[4]
    return embeddings, all_param_values, dist_nn_sizes, dropout, epoch


if __name__ == "__main__":
    training_data, test_data = train_test_split(checkin_file='data/foursquare/checkin_CA_venues.txt')

    location_circle = cPickle.load(open("data/foursquare/location_circle.p", "rb"))
    snapshot_filename="output/foursquare_n30_t30_dnn200-50_drp10_b1000_lr0.010000-100-0.2_e200.model" 

    #TODO: rule out unvisited venue in training set
    accuracy = accuracy_at_N(test_data, location_circle, Ns=[1,5,10,15,20],
                             snapshot_filename=snapshot_filename,
                             data_path="data/foursquare/",
                             batch_size=5000)
    print accuracy

