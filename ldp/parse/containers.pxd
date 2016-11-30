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

from libcpp.pair cimport pair


cdef extern from "<boost/unordered_set.hpp>" namespace "boost" nogil:
    cdef cppclass unordered_set[T]:
        cppclass iterator:
            T& operator*()
            iterator operator++()
            iterator operator--()
            bint operator==(iterator)
            bint operator!=(iterator)
        cppclass reverse_iterator:
            T& operator*()
            iterator operator++()
            iterator operator--()
            bint operator==(reverse_iterator)
            bint operator!=(reverse_iterator)
        #cppclass const_iterator(iterator):
        #    pass
        #cppclass const_reverse_iterator(reverse_iterator):
        #    pass
        unordered_set() except +
        unordered_set(int) except +
        unordered_set(unordered_set&) except +
        #unordered_set(key_compare&)
        #unordered_set& operator=(unordered_set&)
        bint operator==(unordered_set&, unordered_set&)
        bint operator!=(unordered_set&, unordered_set&)
        bint operator<(unordered_set&, unordered_set&)
        bint operator>(unordered_set&, unordered_set&)
        bint operator<=(unordered_set&, unordered_set&)
        bint operator>=(unordered_set&, unordered_set&)
        iterator begin()
        #const_iterator begin()
        void clear()
        size_t count(T&)
        bint empty()
        iterator end()
        #const_iterator end()
        pair[iterator, iterator] equal_range(T&)
        #pair[const_iterator, const_iterator] equal_range(T&)
        void erase(iterator)
        void quick_erase(iterator)
        void erase(iterator, iterator)
        size_t erase(T&)
        void erase_return_void(iterator)
        iterator find(T&)
        #const_iterator find(T&)
        pair[iterator, bint] insert(T&)
        iterator insert(iterator, T&)
        #void insert(input_iterator, input_iterator)
        #key_compare key_comp()
        iterator lower_bound(T&)
        #const_iterator lower_bound(T&)
        size_t max_size()
        reverse_iterator rbegin()
        #const_reverse_iterator rbegin()
        reverse_iterator rend()
        #const_reverse_iterator rend()
        size_t size()
        void swap(unordered_set&)
        iterator upper_bound(T&)
        #const_iterator upper_bound(T&)
        #value_compare value_comp()


cdef extern from "<boost/unordered_map.hpp>" namespace "boost" nogil:
#    cdef cppclass unordered_map[K, T]: # K: key_type, T: mapped_type
#        cppclass iterator:
#            pair[K,T]& operator*()
#            bint operator==(iterator)
#            bint operator!=(iterator)
#            iterator operator++()
#            iterator operator--()
#        unordered_map()
#        bint empty()
#        size_t size()
#        iterator begin()
#        iterator end()
#        pair[K,T] emplace(K, T)
#        iterator find(K)
#        void clear()
#        size_t count(K)
#        T& operator[](K)

    cdef cppclass unordered_map[T, U]:
        cppclass iterator:
            pair[T, U]& operator*()
            iterator operator++()
            iterator operator--()
            bint operator==(iterator)
            bint operator!=(iterator)
        cppclass reverse_iterator:
            pair[T, U]& operator*()
            iterator operator++()
            iterator operator--()
            bint operator==(reverse_iterator)
            bint operator!=(reverse_iterator)
        #cppclass const_iterator(iterator):
        #    pass
        #cppclass const_reverse_iterator(reverse_iterator):
        #    pass
        unordered_map() except +
        unordered_map(unordered_map&) except +
        #unordered_map(key_compare&)
        U& operator[](T&)
        #unordered_map& operator=(unordered_map&)
        bint operator==(unordered_map&, unordered_map&)
        bint operator!=(unordered_map&, unordered_map&)
        bint operator<(unordered_map&, unordered_map&)
        bint operator>(unordered_map&, unordered_map&)
        bint operator<=(unordered_map&, unordered_map&)
        bint operator>=(unordered_map&, unordered_map&)
        U& at(T&)
        iterator begin()
        #const_iterator begin()
        void clear()
        size_t count(T&)
        size_t size()

        bint empty()
        iterator end()
        #const_iterator end()
        pair[iterator, iterator] equal_range(T&)
        #pair[const_iterator, const_iterator] equal_range(key_type&)
        void erase(iterator)
        void erase(iterator, iterator)
        size_t erase(T&)
        iterator find(T&)
        #const_iterator find(key_type&)
        pair[iterator, bint] insert(pair[T, U]) # XXX pair[T,U]&
        iterator insert(iterator, pair[T, U]) # XXX pair[T,U]&
        #void insert(input_iterator, input_iterator)
        #key_compare key_comp()
        iterator lower_bound(T&)
        #const_iterator lower_bound(key_type&)
        size_t max_size()
        reverse_iterator rbegin()
        #const_reverse_iterator rbegin()
        reverse_iterator rend()
        #const_reverse_iterator rend()
        void swap(unordered_map&)
        iterator upper_bound(T&)
        #const_iterator upper_bound(key_type&)
        #value_compare value_comp()
