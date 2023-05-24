# Code inspiration and consistency check:
# https://github.com/tylin/coco-caption/tree/master/pycocoevalcap/bleu
# Original credits to David Chiang <chiang@isi.edu>, Hao Fang <hfang@uw.edu>, Tsung-Yi Lin <tl483@cornell.edu>

import numpy as np
from typing import List
from collections import Counter

from sacreeos.metric.py_core.ngram_utils import cook_tests, cook_refss


class PyBleuScorer(object):
    """ BLEU score calculator. """

    @staticmethod
    def py_compute_score(tests: List[str], refss: List[List[str]]):
        """ Compute the BLEU score.

        :param tests: list of sentences tested by the metric
        :param refss: list of list of sentences used as ground truths by the metric
        :return: List[float], List[List[float]] : the corpus bleu scores and an array of sentence-level bleu scores
        """
        assert (len(tests) == len(refss)), "tests and refss must have same length."
        n = 4
        small = 1e-9
        tiny = 1e-15

        count_tests = [cook_tests(test) for test in tests]
        count_refss = [cook_refss(refs) for refs in refss]
        max_count_refs = []
        for count_refs in count_refss:
            max_count = Counter()
            for count_ref in count_refs:
                max_count = max_count | Counter(count_ref)
            max_count_refs.append(max_count)

        corpus_test_len = 0
        corpus_ref_len = 0
        corpus_num_test_ngrams = np.zeros(n)
        corpus_matchings = [0] * n
        all_sent_level_bleu_scores = []

        # computing sentence level BLEUs
        test_lens = [len(test.split()) for test in tests]

        complete_ref_len = 0
        for count_test, test_len, count_refs, refs in zip(count_tests, test_lens, max_count_refs, refss):
            sent_level_match = [0] * n
            # 'closest' approach
            # (x / test_len) solves ambiguities of same gaps
            ref_len = sorted([len(ref.split()) for ref in refs], key=lambda x: abs(x - test_len) + (x / test_len))[0]

            complete_ref_len += sum([len(ref.split()) for ref in refs])

            num_test_ngrams = np.array([max(0, test_len-k+1) for k in range(1, n+1)])
            corpus_num_test_ngrams += num_test_ngrams
            corpus_test_len += test_len
            corpus_ref_len += ref_len

            for ngram, count in count_test.items():
                if ngram in count_refs.keys():
                    matching = min(count_refs[ngram], count)
                    corpus_matchings[len(ngram)-1] += matching
                    sent_level_match[len(ngram)-1] += matching

            sent_len_ratio = (test_len + tiny) / (ref_len + small)
            sent_bleu = 1.0
            sent_bleu_score = []
            for k in range(n):
                sent_bleu *= (sent_level_match[k] + tiny) / (num_test_ngrams[k] + small)
                bleu_refined = (sent_bleu ** (1.0 / (k + 1)))
                if sent_len_ratio < 1:
                    bleu_refined *= np.exp(1 - 1 / sent_len_ratio)
                sent_bleu_score.append(bleu_refined)
            all_sent_level_bleu_scores.append(sent_bleu_score)

        # computing corpus BLEU
        corpus_len_ratio = (corpus_test_len + tiny) / (corpus_ref_len + small)
        corpus_bleu = 1.0
        corpus_bleu_score = []
        for k in range(n):
            corpus_bleu *= (corpus_matchings[k] + tiny) / (corpus_num_test_ngrams[k] + small)
            bleu_refined = (corpus_bleu ** (1.0 / (k + 1)))
            if corpus_len_ratio < 1:
                bleu_refined *= np.exp(1 - 1 / corpus_len_ratio)
            corpus_bleu_score.append(bleu_refined)

        return corpus_bleu_score, all_sent_level_bleu_scores
