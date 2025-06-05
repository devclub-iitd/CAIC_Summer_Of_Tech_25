import os

letters = "abcdefghijklmnopqrstuvwxyz"

flag = "dcCTF{scr1p71ng_1s_c00l}"

for letter in letters[:len(flag)]:
    for i in range(256):
        os.makedirs(letter + "/" + "/".join(list(f"{i:08b}")))
        with open(letter + "/" + "/".join(list(f"{i:08b}")) + "/" + "tof", "w") as file:
            file.write("1" if i == ord(flag[letters.index(letter)]) else "0")
            
