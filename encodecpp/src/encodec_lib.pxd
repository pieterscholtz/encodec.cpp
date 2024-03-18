cimport cython

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "encodec.h":
    ctypedef struct encodec_context:
        vector[int] out_codes
        pass

    encodec_context * encodec_load_model(string model_path, int n_gpu_layers, float bandwidth)

    bint encodec_compress_audio(encodec_context * ectx, vector[float] raw_audio, int n_threads)

    void encodec_free(encodec_context * ectx)

cdef extern from "<mutex>" namespace "std" nogil:
    cdef cppclass mutex:
        pass
    cdef cppclass lock_guard[T]:
        lock_guard(mutex mm)
