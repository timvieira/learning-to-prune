import numpy as np
from ldp.math.sparse import SparseVectorHash, SparseVector, SparseBinaryVector

def test_sparse_vector():
    ks = np.array([1,3,5])
    vs = np.array([10.0, 30.0, 50.0])

    x = SparseVector(ks, vs)
    w = np.random.uniform(-1,1,size=10)

    # test dot product

    assert abs(w[ks].dot(vs) - x.dot(w)) < 1e-10

    assert repr(x) == 'SparseVector({1: 10.0, 3: 30.0, 5: 50.0})'

    print '[test_sparse_vector] pass'


def test_sparse_binary_vector():
    ks = np.array([1,3,5])

    d = 10

    x = SparseBinaryVector(ks)
    w = np.random.uniform(-1,1,size=d)

    # test dot product
    assert abs(w[ks].sum() - x.dot(w)) < 1e-10

    # test repr
    assert repr(x) == 'SparseBinaryVector([1, 3, 5])'

    # test pluseq
    w0 = w.copy()
    w0[ks] += 100
    x.pluseq(w, coeff=100)
    assert np.abs(w - w0).max() < 1e-8

    print '[test_sparse_binary_vector] pass'


def test_sparse_vector_hash():

    ks = np.array([1,3,5])
    vs = np.array([10.0, 30.0, 50.0])

    x = SparseVectorHash(ks, vs)
    w = np.random.uniform(-1,1,size=10)

    # test __repr__
    assert repr(x) == 'SparseVectorHash({1: 10.0, 3: 30.0, 5: 50.0})'

    a,b = x.to_arrays()
    assert (a == ks).all()
    assert (b == vs).all()

    # test dot product
    assert abs(w[ks].dot(vs) - x.dot(w)) < 1e-10

    # test += (this test modifies an existing element and adds a new key.
    y = SparseVectorHash(np.array([5, 6]),
                         np.array([1, 60.]))
    x += y
    assert x.to_dict() == {1: 10.0, 3: 30.0, 5: 51.0, 6: 60.0}
    print y

    # test argmin
    assert x.argmin() == (1, 10.0)

    # test __iter__
    assert list(x) == [(1, 10.0), (3, 30.0), (5, 51.0), (6, 60.0)]

    print '[test_sparse_vector_hash] pass'


def test_sparse_vector_hash_lazyops():
    d = {1: 10.0, 3: 30.0, 5: 50.0}

    x = SparseVectorHash(np.asarray(d.keys()),
                         np.asarray(d.values()))

    z = 5 * x**2

    for i in xrange(10):
        assert abs(z[i] - 5*d.get(i,0)**2) < 1e-20

    print '[test_sparse_vector_hash_lazyops] pass'


if __name__ == '__main__':
    test_sparse_vector_hash()
    test_sparse_vector()
    test_sparse_vector_hash_lazyops()
    test_sparse_binary_vector()
