#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <string>
#include <array>

const char* filename = "log.txt";
const off_t TOTAL_SIZE = 16L * 1024 * 1024 * 1024;
const off_t FLAG_OFFSET = TOTAL_SIZE / 8;
const off_t TEXT_INTERVAL = 500L * 1024 * 1024;

constexpr std::array<char, 15> encrypt_flag() {
    const char plain[] = "dcCTF{b1G_L0g}";
    std::array<char, 15> encrypted = {};
    constexpr char key = 0x42;
    
    for (int i = 0; i < 14; ++i) {
        encrypted[i] = plain[i] ^ key;
    }
    encrypted[14] = '\0';
    
    return encrypted;
}

// Doing this so that the flag is not visible via `strings` command
constexpr auto encrypted_flag = encrypt_flag();

std::string decrypt_flag() {
    std::string decrypted;
    constexpr char key = 0x42;
    
    for (int i = 0; i < 14; ++i) {
        decrypted += static_cast<char>(encrypted_flag[i] ^ key);
    }
    
    return decrypted;
}

void write_log_line(int fd, const std::string& line) {
    write(fd, line.c_str(), line.size());
}

int main() {
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("open");
        return 1;
    }
    bool flag_written = false;

    for (off_t offset = 0; offset < TOTAL_SIZE; offset += TEXT_INTERVAL) {
        if (lseek(fd, offset, SEEK_SET) < 0) {
            perror("lseek");
            close(fd);
            return 1;
        }

        if (offset >= FLAG_OFFSET && !flag_written) {
            std::string decrypted_flag = decrypt_flag();
            std::string flag_line = "!! ALERT: hidden flag --> " + decrypted_flag + "\n";
            write_log_line(fd, flag_line);
            flag_written = true;
        } else {
            std::string line = "Log checkpoint at offset " + std::to_string(offset / (1024 * 1024)) + " MB: All systems OK.\n";
            write_log_line(fd, line);
        }
    }

    if (lseek(fd, TOTAL_SIZE - 1, SEEK_SET) < 0) {
        perror("lseek to end");
        close(fd);
        return 1;
    }

    char end_byte = '\n';
    write(fd, &end_byte, 1);

    close(fd);
    std::cout << "Log file generated.\n";
    return 0;
}