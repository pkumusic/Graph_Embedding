import time
import sys
import argparse
import numpy as np
import theano.tensor as T
import theano
import lasagne
import lasagne.nonlinearities as nonlinearities
from collections import OrderedDict


x_values = np.asarray([[3,4],[1,2]],dtype='float32')
x = theano.shared(value=x_values, name='x')
index = T.iscalar('index')
iden_fn = theano.function([index], x[index])

updates = OrderedDict()
updates[x] = x / T.sqrt(T.sum(x**2,axis=1,keepdims=True))
normalize_fn = theano.function(inputs=[], updates=updates)

print iden_fn(0)
print iden_fn(1)
normalize_fn()
print iden_fn(0)
print iden_fn(1)

# -------------

"""
a = np.ones([5,1]) * 2
b = np.asarray([1,2,3,4,5,6,7])
b = np.expand_dims(b, axis=1)
c = b[0:5,:]
print a
print c
d = np.concatenate([a,c],axis=1)
print d

"""

# -------------

"""
a = set([10,20,30])
for e in a:
    print e

"""

# --------------

"""
a = np.array([[2,2,4],[2,4,2]])
a = a**0.75
print a
a_sums = a.sum(axis=0)
a /= a_sums[np.newaxis, :]
print a

"""

# --------------

"""
def foo(a):
    b = a
    b[0] = 0
    return b

a = [np.array([2]),np.array([3]),np.array([4])]
b = foo(a)
print a
print b

"""

# --------------

"""
from scipy import stats
xk = np.arange(3)
pk = [0.1, 0.1, 0.8]
custm = stats.rv_discrete(name='custm', values=(xk, pk))
print custm.rvs(size=1)[0]

"""

# --------------

"""
freq = np.array([10, 5, 9])
print np.ceil(freq*1./5).astype(int)

"""

# ---------------

"""
vec_values = np.asarray([[1,2,3],[4,5,6],[7,8,9]],dtype='int32')
vec = theano.shared(value=vec_values, name='vec')
x = np.asarray([[0,1],[1,2]])
vec_col_0 = vec[T.cast(x[:,0],dtype="int32")].reshape((x.shape[0], vec.shape[1]))
vec_col_1 = vec[T.cast(x[:,1],dtype="int32")].reshape((x.shape[0], vec.shape[1]))
vec_cols = T.concatenate([vec_col_0, vec_col_1], axis=1)
print vec_cols.eval()

"""

# ---------------

"""
ones = T.ones([1])
neg_ones = -1*T.ones([1])
ones_comb = T.concatenate([ones, neg_ones], axis=0)
print ones_comb.eval()

a = theano.shared(value=np.array([[2],[3]]))
a = a*1
print a.eval()
#b = ones_comb.reshape([ones_comb.shape[0], 1])*a
b = ones_comb.dimshuffle(0,'x')*a
print b.eval()
c = T.sum(b)
print c.eval()

#twos = 2*T.ones([5])
#mul = neg_ones * twos
#print mul.eval()
#sum_mul = T.sum(mul)
#print sum_mul.eval()
"""

# ---------------

"""
type_vec_values = np.asarray([[1,2,3],[4,5,6]],dtype='int32')
type_vecs = theano.shared(value=type_vec_values, name='node_vecs')
output = T.concatenate([type_vecs[T.cast(0,dtype="int32")], type_vecs[T.cast(1,dtype="int32")]], axis=0)
print output.eval()

output_repeat = T.extra_ops.repeat(output.dimshuffle('x', 0), 2, axis=0)
print output_repeat.eval()

"""
