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


from cython.operator cimport dereference as deref

from libcpp.utility cimport pair
from libcpp.vector cimport vector
from numpy cimport int_t, double_t, int16_t

ctypedef int16_t    D_t
ctypedef double_t   V_t
ctypedef int_t      K_t

cdef V_t semizero
cdef object Cell_dt
cdef object Vt
cdef object Dt
cdef object Kt


# left-child indexed binary rule
cdef packed struct BR:
    D_t right
    D_t parent
    V_t weight

# right-child indexed binary rule
cdef packed struct RCBR:
    D_t left
    D_t parent
    V_t weight

# left-child indexed binary rule
cdef packed struct UR:
    D_t parent
    V_t weight

# lhs-indexed unary rule
cdef packed struct LHSUR:
    D_t child
    V_t weight

# rhs-index binary rule
cdef packed struct RHSBR:
    V_t weight
    D_t parent

# lhs-indexed binary rule
cdef packed struct LHSBR:
    D_t left
    D_t right
    V_t weight


# new school
ctypedef vector[RHSBR] RHSBRvv
ctypedef vector[LHSBR] LHSBRvv
ctypedef vector[LHSUR] LHSURvv
ctypedef vector[RCBR]  RCBRvv
ctypedef vector[BR]    BRvv
ctypedef vector[UR]    URvv

ctypedef RHSBRvv *RHSBRv
ctypedef LHSBRvv *LHSBRv
ctypedef LHSURvv *LHSURv
ctypedef RCBRvv  *RCBRv
ctypedef BRvv    *BRv
ctypedef URvv    *URv


from ldp.parse.containers cimport unordered_map
ctypedef V_t* V_ptr


cdef class Grammar:

    cdef readonly str name
    cdef readonly object lex
    cdef readonly object sym
    cdef readonly int root, nsymbols
    cdef readonly dict _coarse_label
    cdef readonly int n_lrules, n_rules
    cdef readonly coarse2fine
    cdef readonly object[:] fine2coarse
    cdef readonly D_t[:] fine2coarse_int
    cdef readonly object coarse_alphabet, value_domain, as_prob
    cdef readonly object rules, lrules

    cdef unordered_map[pair[int,pair[int,int]], vector[V_ptr]] weight_refs
    cdef unordered_map[pair[int,int], vector[V_ptr]] weight_refs_lex

    # new school grammar indexes
    cdef vector[LHSURv]     r_x_y    # lhs_unary
    cdef vector[URv]        r_y_x    # lc_unary
    cdef vector[URv]        preterm

    cdef vector[LHSBRv]     r_x_yz   # lhs_binary
    cdef vector[BRv]        r_y_xz   # lc_binary
    cdef vector[RCBRv]      r_z_xy   # rc_binary
    cdef vector[RHSBRv]     r_yz_x   # rhs_binary

    cpdef int token(self, str w, int i)
    cpdef long[:] encode_sentence(self, list sentence)
