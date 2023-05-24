#include <math.h>
#include <omp.h>
#include <regex>

#include "bleu.hpp"
#include "ngram_utils.hpp"

#define SMALL 1e-9
#define TINY 1e-15


extern "C" {

void compute_bleu_score(const char** tests_frompy, const char** refss_frompy,
                         uint num_pairs, uint* ref_lens,
                         float* ret_array) {
    int n = 4;

    // convert tests and refss chars into string objects...
    std::string *tests;
    std::string **refss;
    tests = new std::string[num_pairs];
    refss = new std::string*[num_pairs];
    int index_accum = 0;
    for (int i = 0 ; i < num_pairs; i++) {
        tests[i] = std::string(tests_frompy[i]);
        refss[i] = new std::string[ref_lens[i]];
        for (int j = 0; j < ref_lens[i]; j++) {
            refss[i][j] = std::string(refss_frompy[index_accum]);
            index_accum += 1;
        }
    }

    // allocate data
    std::unordered_map<std::string, float>** tests_ngrams;
    std::unordered_map<std::string, float>** refss_ngrams;
    tests_ngrams = new std::unordered_map<std::string, float>*[n];
    refss_ngrams = new std::unordered_map<std::string, float>*[n];
    for (int k = 0; k < n; k++) {
        tests_ngrams[k] = new std::unordered_map<std::string, float>[num_pairs];
        refss_ngrams[k] = new std::unordered_map<std::string, float>[num_pairs];
    }

    // 1. compute ngrams
    for (int k = 0; k < n; k++) {
        #pragma omp parallel for default(none) \
              shared(tests_ngrams, refss_ngrams, ref_lens, num_pairs, refss, tests, k)
        for (int i = 0; i < num_pairs; i++) {
            tests_ngrams[k][i] = get_ngram_count(tests[i], k+1, false);
            std::vector<std::string> refs_list;
            for (int j = 0; j < ref_lens[i]; j++) {
                refs_list.push_back(refss[i][j]);
            }
            refss_ngrams[k][i] = get_collective_ngram_count(refs_list, k+1, false, MAX_OP_ID);
        }
    }

    // 2. compute matchings
    float* corpus_matchings = new float[n];
    // init
    for (int k = 0; k < n; k++) {
        corpus_matchings[k] = 0.0;
        for (int i = 0; i < num_pairs; i++) {
            ret_array[n + i*n + k] = 0.0;
        }
    }
    for (int k = 0; k < n; k++) {
        #pragma omp parallel for default(none) shared(tests_ngrams,  \
            refss_ngrams, num_pairs, corpus_matchings, k, n, ret_array)
        for (int i = 0; i < num_pairs; i++) {
            std::unordered_map<std::string, float>::iterator test_itr;
            for (test_itr = tests_ngrams[k][i].begin(); test_itr != tests_ngrams[k][i].end(); test_itr++) {
                auto refs_ptr = refss_ngrams[k][i].find(test_itr->first);
                bool is_in_refs = refs_ptr != refss_ngrams[k][i].end();
                if (is_in_refs) {
                    float matching = std::min(test_itr->second, refs_ptr->second);
                    #pragma omp atomic
                    corpus_matchings[k] += matching;
                    ret_array[n + i*n + k] += matching;
                }
            }
        }
    }

    // 3. compute corpus matchings and sentence bleu
    int corpus_test_len = 0;
    int corpus_ref_len = 0;
    int* corpus_num_test_ngram = new int[n];
    // init 
    for (int k = 0; k < n; k++)
        corpus_num_test_ngram[k] = 0.0;

    #pragma omp parallel for default(none) shared(num_pairs, refss, tests, corpus_num_test_ngram, \
        ref_lens, corpus_test_len, corpus_ref_len, n, ret_array)
    for (int i = 0; i < num_pairs; i++) {
        int test_len = split_line(tests[i], ' ').size();

        // closest ref len approach
        int ref_len = split_line(refss[i][0], ' ').size();
        int closest_ref_len = ref_len;
        float closest_observed_diff = (float)std::abs(ref_len - test_len) + ((float)ref_len / (float)   test_len);
        for (int j = 1; j < ref_lens[i]; j++) {
            ref_len = split_line(refss[i][j], ' ').size();
            float diff = (float)std::abs(ref_len - test_len) + ((float)ref_len / (float)test_len);
            if (diff < closest_observed_diff) {
                closest_observed_diff = diff;
                closest_ref_len = ref_len;
            }
        }
        #pragma omp atomic
        corpus_test_len += test_len;
        #pragma omp atomic
        corpus_ref_len += closest_ref_len;

        // compute sentence bleu
        float sent_len_ratio = ((float)test_len + TINY) / ((float)closest_ref_len + SMALL);
        float sent_len_penalty = std::exp(1 - 1 / sent_len_ratio);
        float sent_bleu = 1.0;
        for (int k = 1; k <= n; k++) {
            int num_test_ngrams = std::max(0, test_len-k+1);
            # pragma omp atomic
            corpus_num_test_ngram[k-1] += num_test_ngrams;
            sent_bleu *= ((float)ret_array[n + i*n + (k-1)] + TINY) / ((float)num_test_ngrams + SMALL);
            float refined_bleu = std::pow(sent_bleu, 1.0 / ((float)k));
            if (sent_len_ratio < 1) {
                refined_bleu *= sent_len_penalty;
            }
            ret_array[n + i*n + (k-1)] = refined_bleu;
        }

    }
    float corpus_len_ratio = ((float)corpus_test_len + TINY) / ((float)corpus_ref_len + SMALL);
    float corpus_len_penalty = std::exp(1 - 1 / corpus_len_ratio);

    // 4. final bleu computation
    float corpus_bleu = 1.0;
    for (int k = 0; k < n; k++) {
        corpus_bleu *= (corpus_matchings[k] + TINY) / (corpus_num_test_ngram[k] + SMALL);
        float bleu_refined = std::pow(corpus_bleu, 1.0 / ((float)k + 1.0));
        if (corpus_len_ratio < 1)
            bleu_refined *= corpus_len_penalty;
        ret_array[k] = bleu_refined;
    }

    // free memory
    for (int k = 0; k < n; k++) {
        delete [] tests_ngrams[k];
        delete [] refss_ngrams[k];
        for (int i = 0; i < num_pairs; i++) {
            if (k == 0) { // a hacky way to do the operation just once
                delete [] refss[i];
            }
        }
    }
    delete [] tests_ngrams;
    delete [] refss_ngrams;
    delete [] corpus_matchings;
    delete [] refss;
    delete [] tests;
    delete [] corpus_num_test_ngram;
}


}
