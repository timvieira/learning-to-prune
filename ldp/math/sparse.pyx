#!python
#cython: initializedcheck=False
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: infertypes=True
#cython: cdivision=True
#distutils: language = c++
#distutils: libraries = ['stdc++']
#distutils: extra_compile_args = -Wno-unused-function -Wno-unneeded-internal-declaration

import numpy as np
from numpy import array, empty

from libc.stdlib cimport malloc, free


Dt = np.int
Vt = np.double


cdef class SparseVector(object):

    def __cinit__(self, keys, vals):
        self.keys = array(keys, dtype=Dt)
        self.vals = array(vals, dtype=Vt)
        self.length = self.keys.shape[0]

    cpdef double dot(self, V_t[:] w):
        return self._dot(w)

    cdef double _dot(self, V_t[:] w) nogil:
        cdef:
            int i
            double x
        x = 0.0
        for i in range(self.length):
            x += w[self.keys[i]] * self.vals[i]
        return x

    cpdef pluseq(self, V_t[:] w, double coeff):
        self._pluseq(w, coeff)

    cdef void _pluseq(self, V_t[:] w, double coeff) nogil:
        """
        Compute += update, dense vector ``w`` is updated inplace.

           w += coeff*this

        More specifically the sparse update:

          w[self.keys] += coeff*self.vals

        """
        cdef:
            int i
        for i in range(self.length):
            w[self.keys[i]] += coeff * self.vals[i]

    def __reduce__(self):
        # can't pickle memoryview slice, but can pickle ndarray
        keys = array(self.keys, dtype=Dt)
        vals = array(self.vals, dtype=Vt)
        return (SparseVector, (keys, vals), {})

    def __setstate__(self, _):
        pass

    def to_dict(self):
        return dict(zip(self.keys, self.vals))

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.to_dict())


cdef class SparseBinaryVector(object):

    def __cinit__(self, keys):
        self.length = keys.shape[0]
        self.keys = <D_t*>malloc(sizeof(D_t)*self.length) #array(keys, dtype=Dt)
        for i in range(self.length):
            self.keys[i] = keys[i]

    def __dealloc__(self):
        free(self.keys)

    cpdef double dot(self, V_t[:] w):
        return self._dot(w)

    cdef double _dot(self, V_t[:] w) nogil:
        cdef:
            int i
            double x
        x = 0
        for i in range(self.length):
            x += w[self.keys[i]]
        return x

    cpdef pluseq(self, V_t[:] w, V_t coeff):
        """
        Compute += update, dense vector ``w`` is updated inplace.

           w += coeff*this

        More specifically the sparse update:

          w[self.keys] += coeff

        """
        self._pluseq(w, coeff)

    cdef void _pluseq(self, V_t[:] w, V_t coeff) nogil:
        cdef:
            int i
        for i in range(self.length):
            w[self.keys[i]] += coeff

    def __repr__(self):
        keys = self.get_keys()
        #keys.sort()
        return 'SparseBinaryVector(%s)' % keys

    def get_keys(self):
        return [self.keys[i] for i in range(self.length)]

#    def __reduce__(self):
#        # can't pickle memoryview slice, but can pickle ndarray
#        keys = array(self.keys, dtype=Dt)
#        return (SparseBinaryVector, (keys,), {})

#    def __setstate__(self, _):
#        pass


import numpy as np

# Numpy must be initialized. When using numpy from C or Cython you must _always_
# do that, or you will have segfaults
#np.import_array()

from libcpp.utility cimport pair

# Import the C-level symbols of numpy
cimport numpy as np
#cimport cython

from cython.operator cimport dereference as deref, preincrement as inc, \
    predecrement as dec


# Lookup is faster than dict (up to 10 times), and so is full traversal
# (up to 50 times), and assignment (up to 6 times), but creation is
# slower (up to 3 times). Also, a large benefit is that memory
# consumption is reduced a lot compared to a Python dict
cdef class SparseVectorHash:

    """
    Uses C++ map containers for fast dict-like behavior with keys being
    integers, and values float.

    Original Author: Gael Varoquaux (with small modifications by Tim Vieira)
    License: BSD
    """

    def __init__(self, np.ndarray[D_t, ndim=1] keys = None,
                       np.ndarray[V_t, ndim=1] values = None):
        if keys is None:
            assert values is None
            return
        cdef int i
        cdef int size = values.size
        # Should check that sizes for keys and values are equal, and
        # after should boundcheck(False)
        for i in range(size):
            self._map[keys[i]] = values[i]

    @property
    def length(self):
        return len(self)

    def __len__(self):
        return self._map.size()

    def __getitem__(self, D_t key):
        cdef Map_it it = self._map.find(key)
        if it == self._map.end():
            #raise KeyError('%s' % key)
            return 0.0
        return deref(it).second

    def __setitem__(self, D_t key, float value):
        self._map[key] = value

    def __iter__(self):
        cdef D_t size = self._map.size()
        cdef D_t[:] keys = np.empty(size, dtype=Dt)
        cdef V_t[:] values = np.empty(size, dtype=Vt)
        self._to_arrays(keys, values)
        cdef D_t idx
        cdef D_t key
        cdef V_t value
        for idx in range(size):
            key = keys[idx]
            value = values[idx]
            yield key, value

    def to_arrays(self):
        """Return the key, value representation of the IntFloatDict object.

        Returns
        =======
        keys : ndarray, shape (n_items, ), dtype=int
             The indices of the data points
        values : ndarray, shape (n_items, ), dtype=float
             The values of the data points

        """
        cdef int size = self._map.size()
        cdef np.ndarray[D_t, ndim=1] keys = np.empty(size, dtype=Dt)
        cdef np.ndarray[V_t, ndim=1] values = np.empty(size, dtype=np.float64)
        self._to_arrays(keys, values)
        return keys, values

    def to_dict(self):
        return dict(zip(*self.to_arrays()))

    cdef _to_arrays(self, D_t[:] keys, V_t[:] values):
        # Internal version of to_arrays that takes already-initialized arrays
        cdef Map_it it = self._map.begin()
        cdef Map_it end = self._map.end()
        cdef int index = 0
        while it != end:
            keys[index] = deref(it).first
            values[index] = deref(it).second
            inc(it)
            index += 1

    def update(self, SparseVectorHash other):
        cdef Map_it it = other._map.begin()
        cdef Map_it end = other._map.end()
        while it != end:
            self._map[deref(it).first] = deref(it).second
            inc(it)

    def __iadd__(self, SparseVectorHash other):
        cdef Map_it it = other._map.begin()
        cdef Map_it end = other._map.end()
        while it != end:
            self._map[deref(it).first] += deref(it).second
            inc(it)
        return self

    def plusequals_binary(self, SparseBinaryVector f, V_t coeff):
        k = f.keys
        for i in range(f.length):
            self._map[k[i]] += coeff

    def plusequals(self, SparseVector f, V_t coeff):
        k = f.keys
        v = f.vals
        for i in range(f.length):
            self._map[k[i]] += coeff * v[i]

    def copy(self):
        cdef SparseVectorHash out_obj = SparseVectorHash.__new__(SparseVectorHash)
        # The '=' operator is a copy operator for C++ maps
        out_obj._map = self._map
        return out_obj

    def append(self, D_t key, V_t value):
        cdef Map_it end = self._map.end()
        # Decrement the iterator
        dec(end)
        # Construct our arguments
        cdef pair[D_t, V_t] args
        args.first = key
        args.second = value
        self._map.insert(end, args)

    def argmin(self):
        "Return key-value with smallest value."
        cdef Map_it it = self._map.begin()
        cdef Map_it end = self._map.end()
        cdef D_t min_key = -1
        cdef V_t min_value = np.inf
        while it != end:
            if deref(it).second < min_value:
                min_value = deref(it).second
                min_key = deref(it).first
            inc(it)
        return min_key, min_value

    cpdef double dot(self, V_t[:] w):
        return self._dot(w)

    cdef double _dot(self, V_t[:] w) nogil:
        cdef Map_it it = self._map.begin()
        cdef Map_it end = self._map.end()
        cdef V_t val = 0.0
        while it != end:
            val += w[deref(it).first] * deref(it).second
            inc(it)
        return val

    cpdef pluseq(self, V_t[:] w, V_t coeff):
        """
        Compute += update, dense vector ``w`` is updated inplace.
           w += coeff*this
        """
        self._pluseq(w, coeff)

    cdef void _pluseq(self, V_t[:] w, V_t coeff) nogil:
        cdef Map_it it = self._map.begin()
        cdef Map_it end = self._map.end()
        while it != end:
            w[deref(it).first] += coeff*deref(it).second
            inc(it)

    def __reduce__(self):
        return (SparseBinaryVector, self.to_arrays(), {})

    def __setstate__(self, _):
        pass

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.to_dict())

    #___________
    # lazy ops

    def __mul__(self, V_t c):
        return LazyVector(lambda x: x*c, self)

    def __rmul__(self, V_t c):
        return LazyVector(lambda x: x*c, self)

    def __pow__(self, y, _):
        return LazyVector(lambda x: x**y, self)


class LazyVector(object):

    def __init__(self, f, x):
        self.x = x  # underlying vector
        self.f = f  # lazy function

    def __getitem__(self, k):
        return self.f(self.x[k])

    def __mul__(self, V_t c):
        return LazyVector(lambda x: x*c, self)

    def __rmul__(self, V_t c):
        return LazyVector(lambda x: x*c, self)

    def __pow__(self, y, _):
        return LazyVector(lambda x: x**y, self)
