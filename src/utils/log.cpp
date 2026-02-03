#include "log.hpp"
void print_config(const std::string &label, int value) {
    const int label_width = 22;
    const int value_width = 10;

    std::cout << "   " << " "
              << std::left << std::setw(label_width) << (label + ":") // 自动加冒号
              << std::right << std::setw(value_width) << value << std::endl;
}