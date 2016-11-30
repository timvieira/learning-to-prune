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

from ldp.parse.common cimport tri
from cython.operator cimport dereference as deref


cdef class Agenda:
    """Specialized agenda which does approximately CKY ordering.

    It is designed for efficiency and should be used with caution!

    If an element is pushed which has a smaller width than the current width bad
    things will probably happen.

    """

    def __cinit__(self, int N, int NT):
        cdef intset emptyset
        for _  in range(N*(N+1)/2):
            self.q.push_back(emptyset)
        self.N = N
        self.NT = NT
        self.begin = 0
        self.end = 1
        self.done = 0

    cdef inline void set(self, int I, int K, int X) nogil:
        self.q[tri(I,K)].insert(X)
        self.done = 0

    cdef inline int wait(self, int I, int K, int X) nogil:
        return self.q[tri(I,K)].count(X)

#    cdef int totally_empty(self):
#        cdef int i,k,N
#        N = self.N
#        error = 0
#        for i in range(N):
#            for k in range(i+1, N+1):
#                if self.q[tri(i,k)].size():
#                    print '[error] agenda is not empty. span[%s,%s] has %s elements (sentence length = %s).' \
#                        % (i, k, self.q[tri(i,k)].size(), self.N)
#                    error += 1
#        return error == 0

    cdef inline void pop(self) nogil:
        cdef:
            int w
        while self.q[tri(self.begin, self.end)].empty():
            self.begin += 1
            self.end += 1
            if self.end > self.N:
                w = (self.end-self.begin)
                self.begin = 0
                self.end = w+1
                if w+1 > self.N:
                    break
        if (self.end-self.begin) > self.N:
            self.begin = 0
            self.end = 1
            self.done = True
            return
        self.done = False
        it = self.q[tri(self.begin, self.end)].begin()
        self.sym = deref(it)
        self.q[tri(self.begin, self.end)].erase_return_void(it)


cdef class Agenda2:
    """Specialized agenda which does approximately CKY ordering.

    It is designed for efficiency and should be used with caution!

    If an element is pushed which has a smaller width than the current width bad
    things will probably happen.

    """

    def __cinit__(self, int N, int NT):
        cdef IntSet x
        for _  in range(N*(N+1)/2):
            x._contains = <int*>malloc(sizeof(int)*NT)
            for i in range(NT):
                x._contains[i] = 0
            x._elements = new vector[int]()
            self.q.push_back(x)

        self.N = N
        self.NT = NT
        self.begin = 0
        self.end = 1
        self.done = 0

    def __dealloc__(self):
        for x in self.q:
            free(x._contains)
            del x._elements

    cdef inline void set(self, int I, int K, int X) nogil:
        intset_add(self.q[tri(I,K)], X)
        self.done = 0

    cdef inline int wait(self, int I, int K, int X) nogil:
        return intset_contains(self.q[tri(I,K)], X)

    cdef inline void pop(self) nogil:
        cdef:
            int w
        while intset_empty(self.q[tri(self.begin, self.end)]):
            self.begin += 1
            self.end += 1
            if self.end > self.N:
                w = (self.end-self.begin)
                self.begin = 0
                self.end = w+1
                if w+1 > self.N:
                    break
        if (self.end-self.begin) > self.N:
            self.begin = 0
            self.end = 1
            self.done = True
            return
        self.done = False
        self.sym = intset_pop(self.q[tri(self.begin, self.end)])
