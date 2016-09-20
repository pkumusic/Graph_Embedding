export PATH=/home/ubuntu/venv/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda:/usr/local/cuda/lib64:/opt/OpenBLAS-no-openmp/lib
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python graph2vec.py



