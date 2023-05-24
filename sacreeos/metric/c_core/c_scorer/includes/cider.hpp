#ifndef CIDER_HPP
#define CIDER_HPP

#include <string>

#define CIDER_CLASS_BASE 1
#define CIDER_CLASS_D 2
#define CIDER_CLASS_R 3

/* Compute CIDEr score.

    tests_frompy: array of N sentences to be tested by the metric
    refss_frompy: array of N sets of references, paired with ref_lens to represent
        the python list of list of sentences
    num_pairs: number of input test and ground truths pairs
    ref_lens: array of N num references in each input pair
    n: maximum of size of n-grams compute from the inputs

    sigma: gaussian length penalty deviance in Cider-D
    repeat_coeff: repeatition penalty weight for Cider-R
    length_coeff: length penalty weight for Cider-R
    alpha: length penalty deviance in Cider-R

    use_precomp_corpus: only if set to True, the function make use of the 5 following arguments
    num_precomp_entries: number of ngram stored in the precomputed corpus document frequencies
    precomp_corpus_keys: array of n-grams in form of concatenated strings (hence why the next argument is needed)
    precomp_corpus_ns: array of n-gram sizes
    precomp_corpus_doc_freq: array of document frequencies
    precomp_corpus_len: number of documents in the corpus which document frequencies was originally calculated from
    precomp_unique_name: part of virtual address in which the precomputed data is stored onto RAM
                       in order to cut down the conversion cost between Python and C of the pre-computed dictionary

    cider_class: selection of Cider class
    ret_array:


    Return value void* It actually returns the pointer referencing to the precomputed df data
    in the first iteration, the pointer 'precomp_corpus_df_ptr' is NULL, so C performs data conversion
    from python defaultdict to internal data structure, but then it returns at the end the pointer
    of the just allocated data
    Starting from the second training iteration, the 'precomp_corpus_df_ptr' should not be NULL anymore!
    so this function can assign its value to the internal structure pointer, saving the conversion cost.

    Finally, if 'use_precomp_corpus' is True, this function in this mode,
    never free the memory of the precomputed corpus document frequency
    thus it is up to the user,  the responsibility of freeing the data referenced by the
    returned pointer, when its job is done using 'free_cider_precomp_df'.
*/
void* compute_cider_score(const std::string* tests_frompy, const std::string** refss_frompy,
                          uint num_pairs, uint* ref_lens,
                          uint n,

                          // cider args
                          float sigma, // cider-d
                          float repeat_coeff, float length_coeff, float alpha, // cider-r

                          // pre computed corpus tf-idf
                          bool use_precomp_corpus, uint num_precomp_entries,
                          char** precomp_corpus_keys, uint* precomp_corpus_ns,
                          uint* precomp_corpus_doc_freq, uint precomp_corpus_len,
                          void* precomp_corpus_ptr,

                          int cider_class,
                          float* ret_array);


#endif
