#include <future>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <string>
#include <array>


constexpr std::array<char, 24> encrypt_flag() {
    const char plain[] = "dcCTF{m3m0ry_dump_fl4g}";
    std::array<char, 24> encrypted = {};
    constexpr char key = 0x42;
    
    for (int i = 0; i < 24; ++i) {
        encrypted[i] = plain[i] ^ key;
    }
    encrypted[23] = '\0';
    
    return encrypted;
}

// Doing this so that the flag is not visible via `strings` command
constexpr auto encrypted_flag = encrypt_flag();

std::string decrypt_flag() {
    std::string decrypted;
    constexpr char key = 0x42;
    
    for (int i = 0; i < 24; ++i) {
        decrypted += static_cast<char>(encrypted_flag[i] ^ key);
    }
    
    return decrypted;
}

int main() {
    std::cout << "Running forever...\n";
    auto s = decrypt_flag();
    std::promise<void>().get_future().wait();
    return 0;
}