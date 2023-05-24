# Credits of this package are reserved to:
#   - Tsung-Yi Lin <tl483@cornell.edu> Ramakrishna Vedantam <vrama91@vt.edu>
#     for the core functions and structure
#     original source https://github.com/tylin/coco-caption/
#
#   - Gabriel Oliveira Dos Santos
#     for the CIDEr-R related parts
#     original source: https://github.com/gabrielsantosrv/coco-caption/tree/master/pycocoevalcap/ciderR

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import copy
from collections import defaultdict, Counter
from nltk.tokenize import RegexpTokenizer
from scipy.stats.mstats import gmean

from sacreeos.err.commonerr import CommonErrors
from sacreeos.metric.py_core.ngram_utils import cook_refss, cook_tests
from sacreeos.metric import MetricClass


def compute_penalty_by_length(candidate_len, reference_len, alpha=1.0):
    """
    :param candidate_len: candidate length
    :param reference_len: reference length
    :param alpha: parameter to adjust the penalty
    :return: penalty score by difference in length
    """
    delta = abs(reference_len - candidate_len)
    return np.e ** (-(delta ** 2) / (float(alpha) * float(reference_len) ** 2))


def compute_penalty_by_repetition(candidate_sent, reference_sent, penalty_func=lambda x: np.exp(-x)):
    """
    :param candidate_sent: candidate sentence
    :param reference_sent: reference sentence
    :param penalty_func: penalty function
    :return: penalty score by repetition
    """
    # tokenize only words
    tokenizer = RegexpTokenizer(r'\w+')
    tokens_candidate = tokenizer.tokenize(candidate_sent)
    tokens_reference = tokenizer.tokenize(reference_sent)

    word_freq_candidate = Counter(tokens_candidate)
    word_freq_reference = Counter(tokens_reference)

    scores = []

    for word, freq in word_freq_candidate.items():
        # words in the reference and in the hypothesis
        if word_freq_reference.get(word, None) is not None:
            diff = abs(word_freq_reference[word] - freq)
            scores.append(penalty_func(diff))
        else:
            # words in the hypothesis but not in the reference
            scores.append(penalty_func(freq - 1))

    return gmean(np.array(scores))


def compute_doc_freq(count_refss):
    '''
    Compute term frequency for reference data.
    This will be used to compute idf (inverse document frequency later)
    The term frequency is stored in the object
    :return: corpus document frequency
    '''
    df = defaultdict(int)
    for refs in count_refss:
        # refs, k ref captions of one image
        for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
            df[ngram] += 1
    return df


class PyCiderScorer(object):
    """PyCIDEr scorer.
    """

    @staticmethod
    def compute_cider(cider_class, corpus_df, corpus_len, count_tests, count_refss, sent_test, sent_refs, n, penalty_args):
        def counts2vec(cnts, ref_len):
            """
            Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.
            :param cnts:
            :return: vec (array of dict), norm (array of float), length (int)
            """
            vec = [defaultdict(float) for _ in range(n)]
            length = 0
            norm = [0.0 for _ in range(n)]
            for (ngram, term_freq) in cnts.items():
                df = np.log(max(1.0, corpus_df[ngram]))
                # ngram index
                k = len(ngram)-1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[k][ngram] = float(term_freq)*(ref_len - df)
                # compute norm for the vector.  the norm will be used for computing similarity
                norm[k] += pow(vec[k][ngram], 2)

                if k == 1:
                    length += term_freq
            norm = [np.sqrt(val) for val in norm]
            return vec, norm, length

        def sim(vec_test, sent_test, vec_ref, sent_ref, norm_test, norm_ref, length_test, length_ref):
            '''
            Compute the cosine similarity of two vectors.
            :param vec_test: array of dictionary for vector corresponding to hypothesis
            :param sent_test: hypothesis sentence
            :param vec_ref: array of dictionary for vector corresponding to reference
            :param sent_ref: reference sentence
            :param norm_test: array of float for vector corresponding to hypothesis
            :param norm_ref: array of float for vector corresponding to reference
            :param length_test: int containing length of hypothesis
            :param length_ref: int containing length of reference
            :return: array of score for each n-grams cosine similarity
            '''
            delta = float(length_test - length_ref)
            # measure consine similarity
            val = np.array([0.0 for _ in range(n)])
            for k in range(n):
                for (ngram, count) in vec_test[k].items():
                    if cider_class == MetricClass.CIDEr_D or cider_class == MetricClass.CIDEr_R:
                        # vrama91 : added clipping
                        val[k] += min(vec_test[k][ngram], vec_ref[k][ngram]) * vec_ref[k][ngram]
                    elif cider_class == MetricClass.CIDEr_Base:
                        val[k] += vec_test[k][ngram] * vec_ref[k][ngram]

                if (norm_test[k] != 0) and (norm_ref[k] != 0):
                    val[k] /= (norm_test[k]*norm_ref[k])

                assert(not math.isnan(val[k]))

                if cider_class == MetricClass.CIDEr_D:
                    # vrama91: added a length based gaussian penalty
                    val[k] *= np.e**(-(delta**2)/(2*penalty_args['sigma']**2))
                elif cider_class == MetricClass.CIDEr_R:
                    rep_penalty = compute_penalty_by_repetition(sent_test, sent_ref, penalty_func=lambda x: 1 / (1 + x))
                    len_penalty = compute_penalty_by_length(length_test + 1, length_ref + 1, penalty_args['alpha'])
                    val[k] *= (rep_penalty ** penalty_args['repeat_coeff']) * \
                              (len_penalty ** penalty_args['length_coeff'])

            return val

        # compute log reference length
        corpus_len = np.log(float(corpus_len))

        scores = []
        for c_test, c_refs, s_test, s_refs in zip(count_tests, count_refss, sent_test, sent_refs):
            # compute vector for test captions
            v_test, norm_test, length = counts2vec(c_test, corpus_len)
            # compute vector for ref captions
            score = np.array([0.0 for _ in range(n)])
            for c_ref, s_ref in zip(c_refs, s_refs):
                v_ref, norm_ref, length_ref = counts2vec(c_ref, corpus_len)
                score += sim(v_test, s_test, v_ref, s_ref, norm_test, norm_ref, length, length_ref)
            # change by vrama91 - mean of ngram scores, instead of sum
            score_avg = np.mean(score)
            # divide by number of references
            score_avg /= len(c_refs)
            # multiply score by 10
            score_avg *= 10.0
            # append score of an image to the score list
            scores.append(score_avg)
        return scores

    @staticmethod
    def py_compute_score(cider_class, tests, refss, n, corpus_df, corpus_len, penalty_args):
        """ Compute the Cider score.

        Calculates the Cider score accordingly to the specified cider class.

        :param cider_class: int : cider class identifier to choose the Cider formulation
        :param tests: list of sequences tested by the metric
        :param refss: list of list of sequences used as ground truths by the metric
        :param n: int : maximum size of ngrams calculated from the inputs
        :param corpus_df: defaultdict : ngram document-wise counts of the training corpus
        :param corpus_len: int : number of elements in the corpu
        :param penalty_args: dict : arguments related to the penalties
        :return:
        """
        if not (cider_class == MetricClass.CIDEr_D or cider_class == MetricClass.CIDEr_Base
                or cider_class == MetricClass.CIDEr_R):
            raise ValueError(CommonErrors.invalid_metric_class())
        if (corpus_df is None and not corpus_len is None) or (not corpus_df is None and corpus_len is None):
            raise ValueError("\'corpus_df\' and \'corpus_len\' must be both either defined or undefined")

        # compute idf
        count_test = [cook_tests(test, n) for test in tests]
        count_refss = [cook_refss(refs, n) for refs in refss]

        if corpus_df is None:
            corpus_df = compute_doc_freq(count_refss)
            corpus_len = len(refss)
        else:
            pass
            # this method modifies the argument corpus_df by adding zeros entries
            # to avoid unexpected behaviours it may be appropriate to use copy.copy(corpus_df)
            # but since it doesn't impact the cider results, skip the operation for efficiency
            # corpus_df = copy.copy(corpus_df)

        # compute cider score
        scores = PyCiderScorer.compute_cider(cider_class, corpus_df, corpus_len, count_test, count_refss,
                                             tests, refss, n, penalty_args)
        return np.mean(np.array(scores)), np.array(scores)
