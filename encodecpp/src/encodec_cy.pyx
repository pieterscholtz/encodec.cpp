from encodec_lib cimport *
from cython.operator cimport dereference as deref
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.string cimport memcpy
from libcpp.memory cimport shared_ptr, make_shared
import numpy as np
cimport numpy as np
from multiprocessing import cpu_count

cdef mutex compress_mutex

cdef class EncodecModel:
    cdef encodec_context *ectx
    cdef vector[float] audio_buffer

    def __cinit__(self, str model_path, float bandwidth = 6.0):
        cdef bytes input_bytes = model_path.encode('utf-8')  # Convert Python str to bytes
        cdef char* input_char_ptr = input_bytes  # Get a pointer to the underlying char array
        self.ectx = encodec_load_model(input_char_ptr, 0, bandwidth)
        self.audio_buffer = vector[float](60 * 24000, 0.0)

    # don't need a Python constructor at this stage
    # def __init__(self):
    #     pass

    def __dealloc__(self):
        encodec_free(self.ectx)
        self.ectx = NULL

    def compress_audio(self, raw_audio, n_threads = cpu_count() // 2):
        # TODO: mutually exclusive access
        cdef shared_ptr[lock_guard[mutex]] lock = make_shared[lock_guard[mutex]](compress_mutex)
        self.audio_buffer.resize(len(raw_audio))
        memcpy(self.audio_buffer.data(), np.PyArray_DATA(raw_audio), len(raw_audio) * 4);
        cdef int result = encodec_compress_audio(self.ectx, self.audio_buffer, n_threads)
        if result == 0:
            raise ValueError("Compression failed")
        code_len = deref(self.ectx).out_codes.size()
        cdef np.ndarray[np.int32_t, ndim=1] arr = np.zeros(code_len, dtype=np.int32)
        memcpy(&arr[0], &deref(self.ectx).out_codes[0], code_len * sizeof(np.int32_t))
        return arr
