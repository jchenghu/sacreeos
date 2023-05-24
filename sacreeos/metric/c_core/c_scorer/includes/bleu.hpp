#ifndef BLEU_HPP
#define BLEU_HPP

#include <string>

/** Compute the BLEU score.

 tests: array of N sentences to be tested by the metric
 refss: array of N sets of references, paired with ref_lens to represent
        the python list of list of sentences
 num_pairs: number of input test and ground truths pairs
 ref_lens: array of N num references in each input pair

 ret_array: array in which results are placed, the first four elements
    represents the corpus bleus, the remaining 4 * N represent the
    sentence-level bleu scores in the following format
    bleu1-1, bleu2-1, bleu3-1, bleu4-1,
    bleu1-2, bleu2-2, ...
    bleu1-N, bleu2-N, bleu3-N, bleu4-N
*/
void compute_bleu_score(std::string* tests, std::string** refss,
                        uint num_pairs, uint* ref_lens,
                        float* ret_array);

#endif
