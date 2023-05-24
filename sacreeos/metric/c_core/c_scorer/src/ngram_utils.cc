#include <sstream>
#include <omp.h>
#include <regex>

#include "ngram_utils.hpp"


std::vector<std::string> split_line(std::string &line, const char delim) {
    std::stringstream ss(line);
    std::string s;
    std::vector<std::string> out;
    while (std::getline(ss, s, delim)) {
        out.push_back(s);
    }
    return out;
}


std::unordered_map<std::string, float> get_ngram_count(std::string line, uint n, bool tokenize) {
    std::string working_line("");
    if (tokenize) {
        std::regex e("\\w+");
        if (std::regex_search(line, e)) {
            bool not_first_word = false;
            for(auto itr = std::sregex_iterator(line.begin(), line.end(), e);
                itr != std::sregex_iterator(); ++itr) {
                std::smatch m = *itr;
                if (not_first_word) {
                    working_line.append(" ");
                } else {
                    not_first_word = true;
                }
                working_line.append(m.str());
            }
        }
    } else {
        working_line = line;
    }

    std::unordered_map<std::string, float> counter_dict;
    std::vector<std::string> tokens = split_line(working_line, ' ');
    for (int q = 0; q < tokens.size()-n+1; q++) {
        std::string n_gram(tokens[q]);
        for (int t = 1; t < n; t++) {
            n_gram.append(_SEP_);
            n_gram.append(tokens[q+t]);
        }
        auto collect_ptr = counter_dict.find(n_gram);
        bool is_not_in_set = collect_ptr == counter_dict.end();
        if (is_not_in_set) {
            counter_dict.insert(std::make_pair(n_gram, 1.0));
        } else {
            collect_ptr->second = collect_ptr->second + 1.0;
        }
    }
    return counter_dict;
}


std::unordered_map<std::string, float> get_collective_ngram_count(std::vector<std::string> lines,
                                                                  uint n, bool tokenize, int op_id) {
    std::unordered_map<std::string, float> collect_count;

    std::vector<std::string>::iterator lines_itr;
    for (lines_itr = lines.begin(); lines_itr != lines.end(); lines_itr++) {
        std::unordered_map<std::string, float> count = get_ngram_count(*lines_itr, n, tokenize);
        std::unordered_map<std::string, float>::iterator count_itr;
        for (count_itr = count.begin(); count_itr != count.end(); count_itr++) {
            auto collect_ptr = collect_count.find(count_itr->first);
            bool is_in_collect = collect_ptr != collect_count.end();
            if (is_in_collect) {
                switch (op_id) {
                    case SUM_OP_ID: {
                        collect_ptr->second = collect_ptr->second + count_itr->second;
                        break;
                    } case MAX_OP_ID: {
                        if (collect_ptr->second < count_itr->second)
                            collect_ptr->second = count_itr->second;
                        break;
                    } case MIN_OP_ID: {
                        if (collect_ptr->second > count_itr->second)
                            collect_ptr->second = count_itr->second;
                        break;
                    } default: {
                        // just sum
                        collect_ptr->second += count_itr->second;
                    }
                }
            } else {
                collect_count.insert(std::make_pair(count_itr->first, count_itr->second));
            }
        }
    }

    return collect_count;
}
