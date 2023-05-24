#include <iostream>
#include <set>
#include <math.h>
#include <omp.h>

#include "cider.hpp"
#include "ngram_utils.hpp"

#define EPS 1e-9


void compute_corpus_df(std::unordered_map<std::string, uint>* corpus_df,
                       std::string** refss,
                       uint num_pairs, uint* ref_lens,
                       uint n
                       ) {

    // generating corpus_df
    for (int i = 0; i < num_pairs; i++) {
        std::set<std::string> add_ngrams;
        for (int j = 0; j < ref_lens[i]; j++) {
            std::vector<std::string> tokens = split_line(refss[i][j], ' ');
            for (int q = 0; q < tokens.size()-n+1; q++) {
                std::string n_gram(tokens[q]);
                for (int t = 1; t < n; t++) {
                    n_gram.append(_SEP_);
                    n_gram.append(tokens[q+t]);
                }
                bool is_not_in_set = add_ngrams.find(n_gram) == add_ngrams.end();
                if (is_not_in_set) {
                    add_ngrams.insert(n_gram);
                }
            }
        }

        std::set<std::string>::iterator itr;
        for (itr = add_ngrams.begin(); itr != add_ngrams.end(); itr++) {
            auto entry_ptr = corpus_df->find(*itr);
            bool is_in_dict = entry_ptr != corpus_df->end();
            if (is_in_dict) {
                entry_ptr->second = entry_ptr->second + 1;
            } else {
                corpus_df->insert(std::make_pair(*itr, 1));
            }
        }

    }
}


void compute_tfidf(const std::unordered_map<std::string, uint>* corpus_df,
                   uint corpus_len,
                   std::unordered_map<std::string, float>* ngram_counter) {

    std::unordered_map<std::string, float>::iterator c_entry;
    for (c_entry = ngram_counter->begin(); c_entry != ngram_counter->end(); c_entry++) {
        auto corpus_ptr = corpus_df->find(c_entry->first);
        float df_weight = log(float(corpus_len));
        bool is_in_corpus = corpus_ptr != corpus_df->end();
        // the condition (corpus_ptr->second != 0) may look redundant, but it's required because the corpus_df in python
        // is implemented with defaultdicts, which means it might contain zero valued entries
        if (is_in_corpus && corpus_ptr->second != 0) {
            df_weight -= log(corpus_ptr->second);
        }
        c_entry->second = c_entry->second * df_weight;
    }
}


float compute_norm(std::unordered_map<std::string, float>* dict_values) {
    std::unordered_map<std::string, float>::iterator itr;
    float norm = 0.0;
    for (itr = dict_values->begin(); itr != dict_values->end(); itr++) {
        norm += pow(itr->second, 2.0);
    }
    return sqrt(norm);
}


float compute_sum(std::unordered_map<std::string, float>* dict_values) {
    std::unordered_map<std::string, float>::iterator itr;
    float sum = 0.0;
    for (itr = dict_values->begin(); itr != dict_values->end(); itr++) {
        sum += itr->second;
    }
    return sum;
}


float compute_similarity_cider_d(std::unordered_map<std::string, float>* pred_weights,
                         std::unordered_map<std::string, float>* ref_weights) {
    std::unordered_map<std::string, float>::iterator itr;
    float sim = 0;
    for (itr = pred_weights->begin(); itr != pred_weights->end(); itr++) {
        auto ref_ptr = ref_weights->find(itr->first);
        bool is_in_ref = ref_ptr != ref_weights->end();
        if (is_in_ref) {
            sim += std::min(itr->second, ref_ptr->second)*(ref_ptr->second);
        }
    }
    return sim;
}


float compute_similarity_cider_base(std::unordered_map<std::string, float>* pred_weights,
                         std::unordered_map<std::string, float>* ref_weights) {
    std::unordered_map<std::string, float>::iterator itr;
    float sim = 0;
    for (itr = pred_weights->begin(); itr != pred_weights->end(); itr++) {
        auto ref_ptr = ref_weights->find(itr->first);
        bool is_in_ref = ref_ptr != ref_weights->end();
        if (is_in_ref) {
            sim += itr->second * ref_ptr->second;
        }
    }
    return sim;
}


extern "C" {


void* compute_cider_score(const char** tests_frompy, const char** refss_frompy,
                          uint num_pairs, uint* ref_lens, uint n,

                          float sigma,
                          float repeat_coeff, float length_coeff, float alpha,

                          bool use_precomp_corpus, uint num_precomp_entries,
                          char** precomp_corpus_keys, uint* precomp_corpus_ns,
                          uint* precomp_corpus_doc_freq, uint precomp_corpus_len,
                          void* precomp_corpus_df_ptr,

                          int cider_class,
                          float* ret_array) {
    // n must be at least greater than 2... since pseudo length is based on it

    // convert tests and refss chars into string objects
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
    std::unordered_map<std::string, float>*** refss_ngrams;
    tests_ngrams = new std::unordered_map<std::string, float>*[n];
    refss_ngrams = new std::unordered_map<std::string, float>**[n];
    for (int k = 0; k < n; k++) {
        tests_ngrams[k] = new std::unordered_map<std::string, float>[num_pairs];
        refss_ngrams[k] = new std::unordered_map<std::string, float>*[num_pairs];
        for (int i = 0; i < num_pairs; i++) {
            refss_ngrams[k][i] = new std::unordered_map<std::string, float>[ref_lens[i]];
        }
    }


    // 1. Compute Corpus DF
    std::unordered_map<std::string, uint>* corpus_df;
    int corpus_len;
    if (!use_precomp_corpus) {
        corpus_df = new std::unordered_map<std::string, uint>[n];
        #pragma omp parallel for num_threads(n) default(none) \
              shared(corpus_df, ref_lens, num_pairs, refss, n)
        for (int k = 0; k < n; k++) {
            compute_corpus_df(&corpus_df[k], refss, num_pairs, ref_lens, k+1);
        }
        corpus_len = num_pairs;

    // This bit of code might look odd. When 'use_precomp_corpus' is activated,
    // in the first iteration, the pointer 'precomp_corpus_df_ptr' is NULL, so C performs data conversion
    // from python defaultdict to unordered_map, but then it returns at the end the pointer 'corpus_df'
    // which points to the just allocated data.
    // Starting from the second iteration, the 'precomp_corpus_df_ptr' should not be NULL anymore!
    // so it can be directly assigned to 'corpus_df' saving the conversion cost.
    // Finally, this function in this mode, never free the memory pointed by either of the two pointers,
    // which leaves the responsibility to the user, to free it afterwards using 'free_cider_precomp_df'.
    } else {
        if (precomp_corpus_df_ptr == NULL) {
            // if the pointer is NULL, allocate memory once, and keep it until the python object
            // is destroyed, using the function 'free_cider_precomp_df' (below)
            corpus_df = new std::unordered_map<std::string, uint>[n];
            // convert python default_dict into C dictionary
            for (int i = 0 ; i < num_precomp_entries; i++) {
                uint k = precomp_corpus_ns[i];
                std::string key_string(precomp_corpus_keys[i]);
                uint qty = precomp_corpus_doc_freq[i];
                corpus_df[k-1].insert(std::make_pair(key_string, qty));
            }
        } else {
            // otherwise, simply assign reference
            corpus_df = (std::unordered_map<std::string, uint>*)precomp_corpus_df_ptr;
        }
        corpus_len = precomp_corpus_len;
    }


    // 2. Compute count
    float* pseudo_pred_len = new float[num_pairs];
    float** pseudo_ref_len = new float*[num_pairs];
    for (int k = 0; k < n; k++) {
        #pragma omp parallel for default(none) \
              shared(tests_ngrams, refss_ngrams, \
              pseudo_pred_len, pseudo_ref_len, ref_lens, num_pairs, refss, tests, k)
        for (int i = 0; i < num_pairs; i++) {
            tests_ngrams[k][i] = get_ngram_count(tests[i], k+1, false);
            if (k == 1) {
                pseudo_pred_len[i] = compute_sum(&tests_ngrams[k][i]);
                pseudo_ref_len[i] = new float[ref_lens[i]];
            }
            for (int j = 0; j < ref_lens[i]; j++) {
                refss_ngrams[k][i][j] = get_ngram_count(refss[i][j], k+1, false);
                if (k == 1) {
                    pseudo_ref_len[i][j] = compute_sum(&refss_ngrams[k][i][j]);
                }
            }
        }
    }
    float** repeat_pen;
    if (cider_class == CIDER_CLASS_R) {
        repeat_pen = new float*[num_pairs];
        #pragma omp parallel for default(none) \
		     shared(num_pairs, ref_lens, repeat_pen, tests, refss)
        for (int i = 0; i < num_pairs; i++) {
            repeat_pen[i] = new float[ref_lens[i]];
            std::unordered_map<std::string, float> pred_count = get_ngram_count(tests[i], 1, true);
            for (int j = 0; j < ref_lens[i]; j++) {
                std::unordered_map<std::string, float> ref_count = get_ngram_count(refss[i][j], 1, true);
                // compute cider-R repeatition penalty
                std::unordered_map<std::string, float>::iterator itr;
                float gmean = 1;
                for (itr = pred_count.begin(); itr != pred_count.end(); itr++) {
                    auto ref_res = ref_count.find(itr->first);
                    bool is_in_ref = ref_res != ref_count.end();
                    if (is_in_ref) {
                        gmean *= 1.0 / (1 + (abs(itr->second - ref_res->second)));
                    } else {
                        gmean *= 1.0 / itr->second;
                    }
                }
                gmean = pow(gmean, 1.0 / (float)pred_count.size());
                repeat_pen[i][j] = gmean;
            }
        }
    }


    for (int k = 0; k < n; k++) {
        #pragma omp parallel for default(none) \
              shared(corpus_df, tests_ngrams, refss_ngrams, ref_lens, num_pairs, corpus_len, k)
        for (int i = 0; i < num_pairs; i++) {
            compute_tfidf(&corpus_df[k], corpus_len, &tests_ngrams[k][i]);
            for (int j = 0; j < ref_lens[i]; j++) {
                compute_tfidf(&corpus_df[k], num_pairs, &refss_ngrams[k][i][j]);
            }
        }
    }


    float** tests_norm = new float*[n];
    float*** refss_norm = new float**[n];

    for (int k = 0; k < n; k++) {
        tests_norm[k] = new float[num_pairs];
        refss_norm[k] = new float*[num_pairs];
        #pragma omp parallel for default(none) \
              shared(tests_ngrams, tests_norm, refss_norm, refss_ngrams, \
                     ref_lens, num_pairs, k)
        for (int i = 0; i < num_pairs; i++) {
            tests_norm[k][i] = compute_norm(&tests_ngrams[k][i]);
            refss_norm[k][i] = new float[ref_lens[i]];
            for (int j = 0; j < ref_lens[i]; j++) {
                refss_norm[k][i][j] = compute_norm(&refss_ngrams[k][i][j]);
            }
        }
    }



    // 3. compute scores
    float** scores = new float*[n];
    for (int k = 0; k < n; k++) {
        scores[k] = new float[num_pairs];
    }
    for (int k = 0; k < n; k++) {
        #pragma omp parallel for default(none) \
		     shared(tests_ngrams, tests_norm, refss_norm, refss_ngrams, scores, repeat_pen, pseudo_pred_len, \
		        pseudo_ref_len, ref_lens, sigma, repeat_coeff, length_coeff, alpha, cider_class, num_pairs, k)
        for (int i = 0; i < num_pairs; i++) {
            scores[k][i] = 0;
            for (int j = 0; j < ref_lens[i]; j++) {
                float length_delta = pseudo_pred_len[i] - pseudo_ref_len[i][j];
                float sim;
                switch (cider_class) {
                    case CIDER_CLASS_D: {
                        sim = compute_similarity_cider_d(&tests_ngrams[k][i], &refss_ngrams[k][i][j]);
                        float penalty = exp(-pow(length_delta, 2.0) / (2*pow(sigma, 2.0)));
                        sim *= penalty;
                        break;
                    } case CIDER_CLASS_R: {
                        sim = compute_similarity_cider_d(&tests_ngrams[k][i], &refss_ngrams[k][i][j]);
                        float repeat_pen_weight = repeat_coeff;
                        float len_pen_weight = length_coeff;
                        float length_pen = exp(-pow(length_delta, 2.0) / (alpha * pow(pseudo_ref_len[i][j]+1, 2.0)));
                        float penalty = pow(repeat_pen[i][j], repeat_pen_weight) * (pow(length_pen, len_pen_weight));
                        sim *= penalty;
                        break;
                    } default: {
                        sim = compute_similarity_cider_base(&tests_ngrams[k][i], &refss_ngrams[k][i][j]);
                        ; // no penalty
                    }
                }
                float norm_div = tests_norm[k][i] * refss_norm[k][i][j];
                if (norm_div >= EPS) {
                    sim /= norm_div;
                }
                scores[k][i] += sim;
            }
            scores[k][i] = scores[k][i]*10.0 / ref_lens[i];
        }
    }


    // 4. Combine scores
    float* ngram_score = new float[n];
    #pragma omp parallel for default(none) \
         shared(ngram_score, num_pairs, scores, n)
    for (int k = 0; k < n; k++) {
        float sum_score = 0;
        for (int i = 0; i < num_pairs; i++) {
            sum_score += scores[k][i];
        }
        ngram_score[k] = sum_score / num_pairs;
    }

    float final_score = 0;
    for (int k = 0 ; k < n; k++) {
        final_score += ngram_score[k];
    }
    final_score /= n;

    // Populate ret_array
    ret_array[0] = final_score;
    #pragma omp parallel for default(none) \
         shared(ret_array, num_pairs, scores, n)
    for (int i = 0; i < num_pairs; i++) {
        ret_array[i+1] = 0;
        for (int k = 0; k < n; k++) {
            ret_array[i+1] += scores[k][i];
        }
        ret_array[i+1] /= n;
    }


    // free memory
    delete [] ngram_score;
    for (int k = 0; k < n; k++) {
        delete [] tests_ngrams[k];
        delete [] tests_norm[k];
        for (int i = 0; i < num_pairs; i++) {
            delete [] refss_ngrams[k][i];
            delete [] refss_norm[k][i];
            if (k == 0) { // a hacky way to do the operations just once
                delete [] pseudo_ref_len[i];
                delete [] refss[i];
                if (cider_class == CIDER_CLASS_R) {
                    delete [] repeat_pen[i];
                }
            }
        }
        delete [] refss_ngrams[k];
        delete [] refss_norm[k];
        delete [] scores[k];
    }

    if (cider_class == CIDER_CLASS_R) {
        delete [] repeat_pen;
    }

    delete [] pseudo_pred_len;
    delete [] pseudo_ref_len;
    delete [] tests_ngrams;
    delete [] refss_ngrams;
    delete [] tests_norm;
    delete [] refss_norm;
    delete [] refss;
    delete [] tests;
    delete [] scores;

    if (!use_precomp_corpus) {
        // delete only if precomp_corpus mode was deactivated
        delete [] corpus_df;
        return NULL;
    } else {
        return corpus_df;
    }
}

void free_cider_precomp_df(void* precomp_corpus_df_ptr) {
    delete [] (std::unordered_map<std::string, uint>*)precomp_corpus_df_ptr;
}


}
