from scipy.sparse import coo_matrix
from scipy.sparse import *
import numpy as np
import pickle
import random
from tqdm import tqdm



vocab_cut = "vocab_cut.txt"
vocab_pkl = "vocab.pkl"
coco_pkl = "cooc.pkl"
DATA_PATH = "twitter-datasets"
embd = "embeddings"


def pickle_vocab(vocab_cut, vocab_pkl):
    vocab = dict()
    with open(vocab_cut) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open(vocab_pkl, "wb") as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

def cooc_(vocab_pkl, DATA_PATH, cooc_pkl, data = 0):
    with open(vocab_pkl, "rb") as f:
        vocab = pickle.load(f)

    data, row, col = [], [], []
    counter = 1
    pos = "/train_pos_clean.txt" if data ==0 else "/train_pos_full.txt"
    neg = "/train_neg_clean.txt" if data ==0 else "/train_neg_full.txt"
    for fn in [DATA_PATH+pos, DATA_PATH+neg]:
        with open(fn) as f:
            for line in f:
                tokens = [vocab.get(t, -1) for t in line.strip().split()]
                tokens = [t for t in tokens if t >= 0]
                for t in tokens:
                    for t2 in tokens:
                        data.append(1)
                        row.append(t)
                        col.append(t2)

                if counter % 10000 == 0:
                    print(counter)
                counter += 1
    cooc = coo_matrix((data, (row, col)))
    print("summing duplicates (this can take a while)")
    cooc.sum_duplicates()
    with open(cooc_pkl, "wb") as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)



def glove(cooc_pkl, embd, n_emd =20):
    print("loading cooccurrence matrix")
    with open(cooc_pkl, "rb") as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    embedding_dim = n_emd
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001
    alpha = 3 / 4

    epochs = 10

    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in tqdm(zip(cooc.row, cooc.col, cooc.data)):
            logn = np.log(n)
            fn = min(1.0, (n / nmax) ** alpha)
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * eta * fn * (logn - np.dot(x, y))
            xs[ix, :] += scale * y
            ys[jy, :] += scale * x
    np.save(embd, xs)