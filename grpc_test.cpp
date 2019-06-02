#include <iostream>
#include "nlohmann/json.hpp"

using json = nlohmann::json;

int main() {
    json hello;
    hello["Hello, "] = "world!";
    std::cout << hello << std::endl;
    return 0;
}
