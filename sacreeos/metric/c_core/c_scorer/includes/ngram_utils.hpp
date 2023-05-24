#ifndef NGRAM_UTILS_HPP
#define NGRAM_UTILS_HPP

#include <string>
#include <vector>
#include <unordered_map>


const char _SEP_[6] = "_!>._"; // this should be fairly safe
// instead of splitting and generating list of strings,
// n-grams are represented by strings + <separator>

#define SUM_OP_ID 1
#define MAX_OP_ID 2
#define MIN_OP_ID 3

std::vector<std::string> split_line(std::string &line, const char delim);

std::unordered_map<std::string, float> get_ngram_count(std::string line, uint n, bool tokenize);

std::unordered_map<std::string, float> get_collective_ngram_count(std::vector<std::string> lines,
                                                                  uint n, bool tokenize, int op_id);

#endif
