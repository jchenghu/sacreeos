# TO-DO: test FastLib also for Windows / Mac or not Linux based Systems
# OpenMP and virtual memory mechanism may not work as intended...


import os
import ctypes
import numpy as np
from typing import List
from collections import defaultdict

from sacreeos.metric import MetricClass
from sacreeos.metric.c_core import _SEP_, C_CIDER_BASE_ID, C_CIDER_D_ID, C_CIDER_R_ID


LIB_PATH = '/c_scorer/c_core_lib.so'


def invoke_bleu_computer(c_lib, preds, gts, num_images, ref_lens):
    n = 4
    data_preds = (ctypes.c_char_p * num_images)(*preds)
    data_ref_lens = (ctypes.c_uint * num_images)(*ref_lens)
    tot_ref_len = sum(ref_lens)
    data_gts = (ctypes.c_char_p * tot_ref_len)()
    from_idx = 0
    for i in range(num_images):
        to_idx = from_idx + ref_lens[i]
        data_gts[from_idx:to_idx] = gts[i]
        from_idx = to_idx
    ret_array = (ctypes.c_float * ((num_images + 1) * n))()
    c_lib.compute_bleu_score.argtypes = [(ctypes.c_char_p * num_images),
                                           (ctypes.c_char_p * tot_ref_len),
                                           ctypes.c_uint, (ctypes.c_uint * num_images),
                                           (ctypes.c_float * ((num_images + 1) * n))]
    c_lib.compute_bleu_score(data_preds, data_gts,
                               num_images, data_ref_lens,
                               ret_array)

    ret_array = list(ret_array)
    return (ret_array[:n], [ret_array[n:][from_idx:to_idx]
                            for from_idx, to_idx in zip(range(0, num_images * n, n), range(n, num_images * n + n, n))])


def invoke_cider_computer(c_lib, preds, gts, num_images, ref_lens,
                          precomp_corpus_df, precomp_corpus_len, precomp_corpus_ptr,

                          CIDER_CLASS, cider_args):

    data_preds = (ctypes.c_char_p * num_images)(*preds)
    data_ref_lens = (ctypes.c_uint * num_images)(*ref_lens)
    tot_ref_len = sum(ref_lens)
    data_gts = (ctypes.c_char_p * tot_ref_len)()
    from_idx = 0
    for i in range(num_images):
        to_idx = from_idx + ref_lens[i]
        data_gts[from_idx:to_idx] = gts[i]
        from_idx = to_idx

    ret_array = (ctypes.c_float * (num_images + 1))()
    if (precomp_corpus_df is not None) and (precomp_corpus_len is not None):
        use_precomp_corpus = True

        if precomp_corpus_ptr is not None:
            # TRICK: Ugly code but significantly improves efficiency
            # Avoid massiv data conversion and keep only the important parameters

            num_entries = 2  # Placeholder number!
            c_precomp_corpus_key = (ctypes.c_char_p * num_entries)()
            c_precomp_corpus_ns = (ctypes.c_uint * num_entries)()
            c_precomp_corpus_doc_freq = (ctypes.c_uint * num_entries)()
            populate_idx = 0
            for key, value in {'foo': 0, 'bar': 0}.items():
                c_precomp_corpus_key[populate_idx] = bytes(_SEP_.join(key), 'utf-8')
                c_precomp_corpus_ns[populate_idx] = len(key)
                c_precomp_corpus_doc_freq[populate_idx] = value
                populate_idx += 1
            c_precomp_corpus_ptr = (ctypes.c_void_p)(precomp_corpus_ptr)
        else:
            num_entries = len(precomp_corpus_df.items())
            c_precomp_corpus_key = (ctypes.c_char_p * num_entries)()
            c_precomp_corpus_ns = (ctypes.c_uint * num_entries)()
            c_precomp_corpus_doc_freq = (ctypes.c_uint * num_entries)()
            # populate data
            populate_idx = 0
            for key, value in precomp_corpus_df.items():
                c_precomp_corpus_key[populate_idx] = bytes(_SEP_.join(key), 'utf-8')
                c_precomp_corpus_ns[populate_idx] = len(key)
                c_precomp_corpus_doc_freq[populate_idx] = value
                populate_idx += 1
            c_precomp_corpus_ptr = None

        c_lib.compute_cider_score.argtypes = [(ctypes.c_char_p * num_images),
                                              (ctypes.c_char_p * tot_ref_len),
                                              ctypes.c_uint, (ctypes.c_uint * num_images),
                                              ctypes.c_int,

                                              # cider args
                                              ctypes.c_float, ctypes.c_float,
                                              ctypes.c_float, ctypes.c_float,

                                              # precomp corpus data
                                              ctypes.c_bool, ctypes.c_uint,
                                              (ctypes.c_char_p * num_entries), (ctypes.c_uint * num_entries),
                                              (ctypes.c_uint * num_entries), ctypes.c_uint,
                                              ctypes.c_void_p,

                                              ctypes.c_int,
                                              (ctypes.c_float * (num_images + 1))]
        # pass precomputed corpus df and len
        new_precomp_corpus_ptr = c_lib.compute_cider_score(data_preds, data_gts,
                                                           num_images, data_ref_lens, cider_args['n'],

                                                           cider_args['sigma'], cider_args['repeat_coeff'],
                                                           cider_args['length_coeff'], cider_args['alpha'],

                                                           use_precomp_corpus, num_entries,
                                                           c_precomp_corpus_key, c_precomp_corpus_ns,
                                                           c_precomp_corpus_doc_freq, precomp_corpus_len,
                                                           c_precomp_corpus_ptr, # if None it means ptr == NULL in C,
                                                                                 # which will replace the pointer with
                                                                                 # an actual reference

                                                           CIDER_CLASS, ret_array)
    else:
        use_precomp_corpus = False

        plchold_num_entries = 2
        plchold_corpus_key = (ctypes.c_char_p * plchold_num_entries)()
        plchold_corpus_ns = (ctypes.c_uint * plchold_num_entries)()
        plchold_corpus_doc_freq = (ctypes.c_uint * plchold_num_entries)()
        plchold_corpus_len = 2
        # populate with place holder data
        plchold_corpus_key[:] = [bytes("foo", 'utf-8'), bytes("bar", 'utf-8')]
        plchold_corpus_ns[:] = [1, 2]
        plchold_corpus_doc_freq[:] = [1, 2]
        c_lib.compute_cider_score.argtypes = [(ctypes.c_char_p * num_images),
                                                (ctypes.c_char_p * tot_ref_len),
                                                ctypes.c_uint, (ctypes.c_uint * num_images),
                                                ctypes.c_int,

                                                # cider args
                                                ctypes.c_float, ctypes.c_float,
                                                ctypes.c_float, ctypes.c_float,

                                                # precomp corpus data
                                                ctypes.c_bool, ctypes.c_uint,
                                                (ctypes.c_char_p * plchold_num_entries),
                                                (ctypes.c_uint * plchold_num_entries),
                                                (ctypes.c_uint * plchold_num_entries), ctypes.c_uint,
                                                ctypes.c_void_p,

                                                ctypes.c_int,
                                                (ctypes.c_float * (num_images + 1))]
        c_lib.compute_cider_score(data_preds, data_gts,
                                  num_images, data_ref_lens, cider_args['n'],

                                  cider_args['sigma'], cider_args['repeat_coeff'],
                                  cider_args['length_coeff'], cider_args['alpha'],

                                  use_precomp_corpus, plchold_num_entries,
                                  plchold_corpus_key, plchold_corpus_ns,
                                  plchold_corpus_doc_freq, plchold_corpus_len,
                                  None,

                                  CIDER_CLASS, ret_array)
        new_precomp_corpus_ptr = None   # no precomp corpus pointer is returned in this case from the C function

    ret_array = list(ret_array)

    return ret_array[0], ret_array[1:], new_precomp_corpus_ptr


class Invoker(object):

    @staticmethod
    def c_compute_score(metric_obj, tests: List[str], refss: List[List[str]],
                        precomp_corpus_df: defaultdict, precomp_corpus_len: int,
                        precomp_corpus_ptr: str):
        """ Compute the Cider score invoking the C library.

        Calculates the Cider score accordingly to the specified cider class,
        invoking the functions from the C library.

        Warning: the precomp_corpus_df and precomp_corpus_len are exploited
                only once

        :param metric_obj: metric object instance
        :param tests: list of sentences tested by the metric
        :param refss: list of list of sentences used as ground truths by the metric
        :param precomp_corpus_df: defaultdict : pre-computed corpus document frequency
        :param precomp_corpus_len: total number of documents in the corpus
        :param precomp_corpus_ptr: location pointer in which the precomputed data is stored in Heap in order to
                                   cut down the conversion cost between Python and C of the pre-computed dictionary
        :return: the corpus metric score and an array of sentence-level scores
        """
        if os.path.isfile(os.path.dirname(__file__) + LIB_PATH):
            c_lib = ctypes.CDLL(os.path.dirname(__file__) + LIB_PATH)
        else:
            os.system('cd ' + os.path.dirname(__file__) + '/c_scorer/; make')
            if not os.path.isfile(os.path.dirname(__file__) + LIB_PATH):
                raise Exception("Making of C engine library failed. In case you moved the files into a new platform. "
                                "Try a make clean and compile again")
            else:
                c_lib = ctypes.CDLL(os.path.dirname(__file__) + LIB_PATH)

        pred_frompy = []
        gts_frompy = []
        ref_lens = []
        num_images = len(tests)
        for i in range(num_images):
            pred_frompy.append(bytes(tests[i], 'utf-8'))
            ref_lens.append(len(refss[i]))
        for i in range(num_images):
            gts_list = []
            for j in range(ref_lens[i]):
                gts_list.append(bytes(refss[i][j], 'utf-8'))
            gts_frompy.append(gts_list)

        # since the C function require all arguments, all metrics argument are collected here first,
        # using the respective metrics default values

        if metric_obj.get_metric_class() == MetricClass.BLEU:
            bleu_score, bleu_score_array = invoke_bleu_computer(c_lib, pred_frompy, gts_frompy, num_images, ref_lens)
            return bleu_score, bleu_score_array

        if metric_obj.get_metric_class() == MetricClass.CIDEr_Base:
            cider_args = {'n': metric_obj.get_n(), 'sigma': -1.0,
                          'repeat_coeff': -1.0, 'length_coeff': -1.0, 'alpha': -1.0}
            cider_class = C_CIDER_BASE_ID
        elif metric_obj.get_metric_class() == MetricClass.CIDEr_D:
            cider_args = {'n': metric_obj.get_n(),
                          'sigma': metric_obj.get_sigma(),
                          'repeat_coeff': -1.0, 'length_coeff': -1.0, 'alpha': -1.0}
            cider_class = C_CIDER_D_ID
        elif metric_obj.get_metric_class() == MetricClass.CIDEr_R:
            cider_args = {'n': metric_obj.get_n(), 'sigma': -1.0,
                          'repeat_coeff': metric_obj.get_repeat_coeff(),
                          'length_coeff': metric_obj.get_length_coeff(),
                          'alpha': metric_obj.get_alpha()}
            cider_class = C_CIDER_R_ID
        else:
            raise ValueError("Illegal MetricClass for the cider_class argument.")

        cider_score, cider_score_array, new_precomp_corpus_ptr \
            = invoke_cider_computer(c_lib, pred_frompy, gts_frompy, num_images, ref_lens,
                                    precomp_corpus_df, precomp_corpus_len,
                                    precomp_corpus_ptr,
                                    cider_class, cider_args)

        return cider_score, np.array(cider_score_array), new_precomp_corpus_ptr

    @staticmethod
    def free_cider_precomp_df(precomp_corpus_ptr: str):
        """ Free memory allocated in C.
        :param precomp_corpus_ptr: precomputed doc frequency pointer previously allocated by C, to be freed
        :param n: precomp_df n-grams sizes
        :return: 
        """
        if os.path.isfile(os.path.dirname(__file__) + LIB_PATH):
            c_lib = ctypes.CDLL(os.path.dirname(__file__) + LIB_PATH)
        else:
            os.system('cd ' + os.path.dirname(__file__) + '/c_scorer/; make')
            if not os.path.isfile(os.path.dirname(__file__) + LIB_PATH):
                raise Exception("Making of C engine library failed, please open an issue on the official repository\n"
                                + "https://github.com/jchenghu/Scst providing details of the system and make file"
                                + " output.")
            else:
                c_lib = ctypes.CDLL(os.path.dirname(__file__) + LIB_PATH)
        c_lib.free_cider_precomp_df.argtypes = [ctypes.c_void_p]
        c_lib.free_cider_precomp_df(precomp_corpus_ptr)
