import os

letters = "abcdefghijklmnopqrstuvwxyz"

flag = []

for letter in letters[:24]:
    for i in range(256):
        with open(letter + "/" + "/".join(list(f"{i:08b}")) + "/" + "tof") as file:
            if file.read() == "1":
                flag.append(i)
                
print("".join([chr(i) for i in flag]))
