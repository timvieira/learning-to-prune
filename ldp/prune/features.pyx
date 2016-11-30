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


"""Feature extraction for pruning spans.

TODO:

 - hooks for plugging-in different feature sets

 - The longest-frequent-suffix implementation doesn't handle unseen words.

 - Use a pipeline to compute integerized longest-frequent-suffix
   features. Similarly for Berkeley unseen word representation.

 - POS tags features

"""
import cPickle
from collections import defaultdict
import numpy as np
from numpy cimport ndarray
from ldp.math.sparse cimport SparseBinaryVector
from ldp.parsing.unk import signature
from ldp.parse.grammar cimport Grammar
from ldp.prune.example cimport Example

cimport numpy as np
from numpy cimport uint64_t, uint32_t, int32_t
from murmurhash.mrmr cimport hash32

#from cython.operator cimport dereference
#from libc.stdio cimport printf
#cdef uint32_t hash32(void* key, int length, uint32_t seed) nogil:
#    printf("ERROR using fake murmurhash.\n")
#    return seed


cdef uint32_t hashit(uint64_t key, unsigned int seed):
    """Compute the 32bit murmurhash3 of a int key at seed."""
    #cdef uint32_t out
    #MurmurHash3_x86_32(&key, sizeof(uint64_t), seed, &out)
    #return out
    return hash32(&key, sizeof(uint64_t), seed)


cpdef uint32_t hash_bytes(bytes key, unsigned int seed):
    """Compute the 32bit murmurhash3 of a bytes key at seed."""
    return hash32(<char*> key, len(key), seed)


from arsenal.cache import memoize

def one_or_more(a):
    plus = a + '+'
    return lambda m: (plus if len(m.group(1)) > 1 else a)

import re
DIGIT = re.compile('[0-9]')
LOWER = re.compile('([a-z]+)')
UPPER = re.compile('([A-Z]+)')

@memoize
def letter_pattern(w):
    if w.startswith('-') and w.endswith('-'):
        # handles ptb's encoding of parentheses -LRB-
        return w
    w = UPPER.sub(one_or_more('A'), w)
    w = LOWER.sub(one_or_more('a'), w)
    w = DIGIT.sub('8', w)
    return w


cdef class Features(object):

    def __init__(self, grammar, nfeatures,
                 longest_frequent_suffix=False):

        self.LFS = None
        if longest_frequent_suffix:
            print '[suffix-map] loading...'
            self.LFS = LongestFrequentSuffixTable.load(longest_frequent_suffix)

        self.nfeatures = nfeatures
        self.grammar = grammar

        # get unique "terminal" ids for begin and end sentence
        self.BOS = len(self.grammar.lex)
        self.EOS = len(self.grammar.lex)+1

        self.seed = 0

    def __call__(self, Example e):
        cdef:
            int[4] b
            int[4] sh
            long[:] shape
            long[:] w
            object[:,:] F
            int I, K, S, width, width_bucket, N
            ndarray[int,ndim=1] M

        S = 16

        assert self.nfeatures == 2**22

        # Using array instead of dict save lots of memory.
        F = np.zeros((e.N,e.N+1),dtype=object)

        N = e.N
        if self.LFS is not None:
            # represent words as their longest frequent suffix.
            w = np.array([self.LFS[x] for x in e.tokens])       # XXX: careful with unseen word tools!
        else:
            w = e.tokens #e.sentence.split()   # uses integers

        # words before mapping to ints and OOV-signatures
        words = e.sentence.split()

        # word shape features
        shape = np.array([hash_bytes(letter_pattern(t), self.seed) for t in words])

        # temp array
        #m = np.empty(N, dtype=np.uint64)

        for (I,K) in e.nodes:

            #str(len(K-I)),  # boolean feature for width
            width = K-I
            width_bucket = 0
            if width == 1:
                width_bucket = 1
            elif width == 2:
                width_bucket = 2
            elif width == 3:
                width_bucket = 3
            elif width == 4:
                width_bucket = 4
            elif width == 5:
                width_bucket = 5
            elif width <= 10:
                width_bucket = 10
            elif width <= 20:
                width_bucket = 20
            else: # width >= 21
                width_bucket = 21

            b[0] = w[I-1] if I-1 >= 0 else self.BOS
            b[1] = w[I]
            b[2] = w[K-1]
            b[3] = w[K] if K < N else self.EOS

            # words shapes of boundary
            sh[0] = shape[I-1] if I-1 >= 0 else self.BOS
            sh[1] = shape[I]
            sh[2] = shape[K-1]
            sh[3] = shape[K] if K < N else self.EOS

            # span_shape
            span_shape = hash_bytes(bytes(shape[I:K]), self.seed)

            # TODO: SparseBinaryVector reallocates memory, make an alternative
            # constructor for the case where we already have a pointer and the
            # length. Actually, creating a vector at all is wasteful if all you
            # want is the dot-product.
            M = np.empty(S, dtype=np.int32)
            M[ 0] = self.hashpack(  0,      0,            0)
            M[ 1] = self.hashpack(  1,      0,         b[0])
            M[ 2] = self.hashpack(  2,      0,         b[1])
            M[ 3] = self.hashpack(  3,      0,         b[2])
            M[ 4] = self.hashpack(  4,      0,         b[3])
            M[ 5] = self.hashpack(  5,   b[0],         b[1])
            M[ 6] = self.hashpack(  6,   b[2],         b[3])
            M[ 7] = self.hashpack(  7,   b[0],         b[3])
            M[ 8] = self.hashpack(  8,   b[1],         b[2])
            M[ 9] = self.hashpack(  9,  sh[0],        sh[1])
            M[10] = self.hashpack( 10,  sh[2],        sh[3])
            M[11] = self.hashpack( 11,  sh[0],        sh[3])
            M[12] = self.hashpack( 12,  sh[1],        sh[2])
            M[13] = self.hashpack( 13,      0,   span_shape)
            M[14] = self.hashpack( 14,      0, width_bucket)
            M[15] = self.hashpack( 15,      0,            N if N <= 40 else 40)

            F[I,K] = SparseBinaryVector(M)

        return F

    cdef inline int32_t hashpack(self, int a, int b, int c):
        cdef uint64_t packed = pack(a,b,c)
        cdef int32_t hashed = <int32_t>hashit(packed, self.seed)
        cdef int32_t smashd = (hashed if hashed >= 0 else -hashed) & 0b1111111111111111111111
        return smashd

    def mask(self, Example e, double[:] W):
        cdef:
            int[4] b
            int[4] sh
            long[:] shape
            long[:] w
            object[:,:] F
            int I, K, S, width, width_bucket, N
            ndarray[int,ndim=1] M

        S = 16

        m = e.mask

        # Using array instead of dict save lots of memory.
        F = np.zeros((e.N,e.N+1),dtype=object)

        N = e.N
        if self.LFS is not None:
            # represent words as their longest frequent suffix.
            w = np.array([self.LFS[x] for x in e.tokens])       # XXX: careful with unseen word tools!
        else:
            w = e.tokens #e.sentence.split()   # uses integers

        # words before mapping to ints and OOV-signatures
        words = e.sentence.split()

        # word shape features
        shape = np.array([hash_bytes(letter_pattern(t), self.seed) for t in words])

        # temp array
        #m = np.empty(N, dtype=np.uint64)

        for (I,K) in e.nodes:  # TODO: replace e.nodes with a for-loop

            #str(len(K-I)),  # boolean feature for width
            width = K-I
            width_bucket = 0
            if width == 1:
                width_bucket = 1
            elif width == 2:
                width_bucket = 2
            elif width == 3:
                width_bucket = 3
            elif width == 4:
                width_bucket = 4
            elif width == 5:
                width_bucket = 5
            elif width <= 10:
                width_bucket = 10
            elif width <= 20:
                width_bucket = 20
            else: # width >= 21
                width_bucket = 21

            b[0] = w[I-1] if I-1 >= 0 else self.BOS
            b[1] = w[I]
            b[2] = w[K-1]
            b[3] = w[K] if K < N else self.EOS

            # words shapes of boundary
            sh[0] = shape[I-1] if I-1 >= 0 else self.BOS
            sh[1] = shape[I]
            sh[2] = shape[K-1]
            sh[3] = shape[K] if K < N else self.EOS

            # span_shape
            span_shape = hash_bytes(bytes(shape[I:K]), self.seed)

            m[I,K] = (W[self.hashpack(  0,      0,            0)]
                      + W[self.hashpack(  1,      0,         b[0])]
                      + W[self.hashpack(  2,      0,         b[1])]
                      + W[self.hashpack(  3,      0,         b[2])]
                      + W[self.hashpack(  4,      0,         b[3])]
                      + W[self.hashpack(  5,   b[0],         b[1])]
                      + W[self.hashpack(  6,   b[2],         b[3])]
                      + W[self.hashpack(  7,   b[0],         b[3])]
                      + W[self.hashpack(  8,   b[1],         b[2])]
                      + W[self.hashpack(  9,  sh[0],        sh[1])]
                      + W[self.hashpack( 10,  sh[2],        sh[3])]
                      + W[self.hashpack( 11,  sh[0],        sh[3])]
                      + W[self.hashpack( 12,  sh[1],        sh[2])]
                      + W[self.hashpack( 13,      0,   span_shape)]
                      + W[self.hashpack( 14,      0, width_bucket)]
                      + W[self.hashpack( 15,      0,            N if N <= 40 else 40)]
                      ) >= 0
        return m


cdef inline uint64_t pack(uint64_t template, uint64_t a, uint64_t b):
    # bits
    # 5    |          32 | template id
    # 29   | 536,870,912 | conjunction A
    # 29   | 536,870,912 | conjunction B
    # 64
    cdef uint64_t key = 0
    key |= template
    key |= a << (5)
    key |= b << (5+29)
    return key


class LongestFrequentSuffixTable(object):

    def __init__(self):
        self.C = None
        self.M = None
        self.threshold = None

    def fit(self, sentences, threshold):
        n = 0
        t = 0
        C = defaultdict(int)
        for t, s in enumerate(sentences):
            for w in s.split():
                n += 1
                for i in range(len(w)):
                    C[w[i:]] += 1

            if t % 1000 == 0:
                print '[suffix count] processed %s sentences; tokens: %s, suffixes: %s' % (t, n, len(C))

        # final
        print '[suffix count] FINAL processed %s sentences; tokens: %s, suffixes: %s' % (t, n, len(C))

        M = {}
        for t, x in enumerate(sentences):
            for w in x.split():
                if C[w] >= threshold:   # entire word is frequent enough
                    M[w] = w
                else:
                    for i in range(len(w)):
                        s = w[i:]
                        if C[s] >= threshold:
                            M[w] = s
                            break

        self.C = C
        self.M = M
        self.threshold = threshold

    def __getitem__(self, e):
        assert isinstance(e, int)
        M = self.M
        return [M[x] for x in e.tokens]

    def __call__(self, w):
        assert isinstance(w, basestring)
        for i in range(len(w)):
            s = w[i:]
            if self.C[s] >= self.threshold:
                return s

    def save(self, filename):
        assert self.C is not None and self.M is not None
        with file(filename, 'wb') as f:
            cPickle.dump([self.C, self.M, self.threshold], f)

    @classmethod
    def load(cls, filename):
        x = cls()
        with file(filename) as f:
            [x.C, x.M, x.threshold] = cPickle.load(f)
        return x
