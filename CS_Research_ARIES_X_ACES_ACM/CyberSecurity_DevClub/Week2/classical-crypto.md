# Classical Cryptography & Encoding

## Understanding Ciphers vs Encoding

**Encoding** transforms data for compatibility (not security):
- Eg. Base64, URL encoding, ASCII/Unicode conversion
- Reversible without a key
- Used for data transmission and storage

**Encryption** transforms data for security:
- Requires a key to decrypt
- Eg. Caesar cipher, AES, RSA
- Designed to protect information

---

## Classical Ciphers

### Caesar Cipher
The simplest substitution cipher - shifts each letter by a fixed number.

**How it works:**
- Each letter is shifted by the same amount (e.g., A→D, B→E, C→F with shift of 3)
- The cipher "wraps around" (Z shifts to A, B, C...)
- Numbers and symbols typically remain unchanged

**Example:**
- Plaintext: `HELLO WORLD`
- Shift: 3
- Ciphertext: `KHOOR ZRUOG`

**Breaking Caesar Ciphers:**
Since there are only 25 possible shifts, you can try them all! Look for the shift that produces readable English text.

**Practice Examples:**
1. `WKLV LV D WHVW` (hint: shift 3)
2. `DQQD ORYH SLFWXUHV` (hint: shift 23)
3. `FDHVDU FLSKHU LV HDV\` (hint: shift 3)

**Try it yourself:**
- CyberChef Caesar Cipher: https://gchq.github.io/CyberChef/#recipe=Caesar_Cipher(13,true,true,false)
- dCode Caesar Decoder: https://www.dcode.fr/caesar-cipher

**Practical Example - Solving Caesar Cipher:**
```bash
# Manual process for "WKLV LV D WHVW"
Input: WKLV LV D WHVW
Try shift 1: VJKU UK C VGUV
Try shift 2: UIJT TJ B UFTU  
Try shift 3: THIS IS A TEST ← Readable English!
Answer: "THIS IS A TEST" with shift of 3
```

**CyberChef Recipe for Caesar Brute Force:**
```
Input: WKLV LV D WHVW
Recipe: ROT13_Brute_Force
Result: Shows all 25 possible shifts, find readable English
```

### Vigenère Cipher
Uses a keyword to create multiple Caesar ciphers - much stronger than basic Caesar!

**How it works:**
- Uses a repeating keyword to determine shift amounts
- Each letter of the key determines the shift for that position
- Example: Keyword "CAT" gives shifts of 2,0,19 (C=2, A=0, T=19)

**Example:**
- Plaintext: `HELLO`
- Key: `CRYPTO`
- Process: H+C=J, E+R=V, L+Y=J, L+P=A, O+T=H
- Ciphertext: `JVJAH`

**Practice Examples:**
1. `LXFOPVEFRNHR` with key `LEMON`
2. `ALQZ MW R XIPQ GSPEXTO` with key `CRYPTO`
3. `BPVVB CSYEB NKFDC` with key `SECRET`

**Try it yourself:**
- CyberChef Vigenère: https://gchq.github.io/CyberChef/#recipe=Vigen%C3%A8re_Decode('fire')
- dCode Vigenère: https://www.dcode.fr/vigenere-cipher

**Practical Example - Solving Vigenère:**
```bash
# Solving "LXFOPVEFRNHR" with key "LEMON"
Ciphertext: L X F O P V E F R N H R
Key:        L E M O N L E M O N L E
Process:    L-L=A, X-E=T, F-M=T, O-O=A, P-N=C, V-L=K...
Result:     ATTACK AT DAWN
```

**CyberChef Recipe for Vigenère:**
```
Input: LXFOPVEFRNHR
Recipe: Vigenère_Decode('LEMON')
Output: ATTACKATDAWN
```

### ROT13 and Variants
ROT13 is simply Caesar cipher with a shift of 13 - it's its own reverse!

**Examples:**
- `Hello World` → `Uryyb Jbeyq`
- `DevClub Rocks` → `QriPyho Ebpxf`
- `CTF Challenge` → `PGS Punyratyr`

**ROT47 Examples:**
- `Hello World!` → `w6==@ (@C=5P`
- `123 ABC xyz` → `'(# pqr JKL`

**Try it yourself:**
- CyberChef ROT13: https://gchq.github.io/CyberChef/#recipe=ROT13(true,true,false,13)
- Online ROT13: https://rot13.com/

### Atbash Cipher
Ancient Hebrew cipher that reverses the alphabet - A becomes Z, B becomes Y, etc.

**How it works:**
- First letter of alphabet maps to last letter
- Second letter maps to second-to-last, etc.
- A→Z, B→Y, C→X, D→W, E→V, F→U...

**Examples:**
- `HELLO` → `SVOOL`
- `ATBASH` → `ZGYZHS`
- `SECRET` → `HVXIVG`

**Try it yourself:**
- CyberChef Atbash: https://gchq.github.io/CyberChef/#recipe=Atbash_Cipher()
- Practice: Decode `XIBIVHB XOFMTVGH`

### Playfair Cipher
Uses a 5x5 grid of letters and encrypts pairs of letters (digraphs).

**How it works:**
1. Create 5x5 grid with keyword (I/J share same cell)
2. Break plaintext into pairs of letters
3. Apply encryption rules based on position in grid

**Example with keyword "MONARCHY":**
```
M O N A   R
C H Y B   D
E F G I/J K
L P Q S   T
U V W X   Z
```

**Encryption rules:**
- Same row: Move one position right
- Same column: Move one position down  
- Rectangle: Take letter from same row, opposite corner

**Example:**
- `HE` → `DK` (rectangle rule)
- `LL` → `QP` (same row, but first add filler X: `LX`)

**Try it yourself:**
- dCode Playfair: https://www.dcode.fr/playfair-cipher
- Practice keyword: `PLAYFAIR`

### Rail Fence Cipher (Zigzag)
Writes message in zigzag pattern across multiple "rails" then reads off linearly.

**How it works with 3 rails:**
```
H . . . O . . . R . . .
. E . L . W . L . ! . .
. . L . . . O . . . D .

Read off: HEORELWL!LD → "HELLO WORLD!"
```

**Examples:**
- `ATTACK AT DAWN` (3 rails) → `ACDTTAAATKAWN`
- `CRYPTO CIPHER` (4 rails) → `RTIYHPOCE PRC`

**Try it yourself:**
- CyberChef Rail Fence: https://gchq.github.io/CyberChef/#recipe=Rail_Fence_Cipher_Decode(3,0)
- dCode Rail Fence: https://www.dcode.fr/rail-fence-cipher

### Polybius Square (Prison Tap Code)
Maps letters to coordinates in a 5x5 grid.

**Standard Polybius Square:**
```
  1 2 3 4   5
1 A B C D   E
2 F G H I/J K
3 L M N O   P
4 Q R S T   U
5 V W X Y   Z
```

**Examples:**
- `A` → `11`, `B` → `12`, `C` → `13`
- `HELLO` → `23 15 31 31 34`
- `SECRET` → `43 15 13 42 15 44`

**Tap Code Variant (used by prisoners):**
- Tap row number, pause, tap column number
- `HELLO` = tap-tap-tap pause tap-tap-tap (H = 23)

**Try it yourself:**
- dCode Polybius: https://www.dcode.fr/polybius-cipher
- Practice: Decode `44 23 24 43 24 43 13 42 54 51 44 34`

### Beaufort Cipher
Similar to Vigenère but uses subtraction instead of addition.

**How it works:**
```
# Standard Vigenère: Plaintext + Key = Ciphertext
# Beaufort: Key - Plaintext = Ciphertext

Example with key "SECRET":
Plaintext: HELLO
Key:       SECRE
Process:   S-H=K, E-E=A, C-L=R, R-L=F, E-O=Q
Result:    KARFQ
```

**Variant - Beaufort Autokey:**
- Uses plaintext itself as part of the key
- More secure than standard Beaufort

**Try it yourself:**
- dCode Beaufort: https://www.dcode.fr/beaufort-cipher
- Practice key: `BEAUFORT`

### Trifid Cipher
3D extension of the Polybius square, more secure than traditional methods.

**How it works:**
1. Uses three 3x3x3 grids with letters distributed
2. Each letter maps to three coordinates (layer, row, column)
3. Coordinates are written in rows then read in columns
4. New coordinates are converted back to letters

**Example:**
```
A = (1,1,1), B = (1,1,2), C = (1,1,3)...
HELLO → coordinates → scrambled → new letters
```

**Try it yourself:**
- dCode Trifid: https://www.dcode.fr/trifid-cipher
- Note: Complex setup, good for advanced challenges

## Common Encoding Schemes

### Base64
Base64 converts binary data into text using 64 printable characters.

**How to recognize Base64:**
- Uses characters A-Z, a-z, 0-9, +, /
- Often ends with = or == padding
- Length is always a multiple of 4

**Examples:**
- `Hello` → `SGVsbG8=`
- `DevClub` → `RGV2Q2x1Yg==`
- `Flag: cyber{base64_master}` → `RmxhZzogY3liZXJ7YmFzZTY0X21hc3Rlcn0=`

**Multi-layer Example:**
```
Start: Hello World
Base64: SGVsbG8gV29ybGQ=
Base64 again: U0dWc2JHOGdWMjl5YkdRPQ==
```

**Try it yourself:**
- CyberChef Base64: https://gchq.github.io/CyberChef/#recipe=From_Base64('A-Za-z0-9%2B/%3D',true,false)
- Online Base64 Decoder: https://www.base64decode.org/

### Hexadecimal
Hexadecimal (base 16) represents binary data using 0-9 and A-F.

**Examples:**
- `Hello` → `48656c6c6f`
- `DevClub` → `446576436c7562`
- `Flag{hex_is_easy}` → `466c61677b6865785f69735f656173797d`

**Try it yourself:**
- CyberChef Hex: https://gchq.github.io/CyberChef/#recipe=From_Hex('Auto')
- Online Hex Decoder: https://www.rapidtables.com/convert/number/hex-to-ascii.html

### URL Encoding
URL encoding converts special characters into % followed by hex values.

**Examples:**
- `Hello World!` → `Hello%20World%21`
- `user@domain.com` → `user%40domain.com`
- `flag{url_encoding_fun}` → `flag%7Burl_encoding_fun%7D`

**Try it yourself:**
- CyberChef URL Decode: https://gchq.github.io/CyberChef/#recipe=URL_Decode()
- Online URL Decoder: https://www.urldecoder.org/

### Binary and ASCII
Binary represents text using 1s and 0s - the language computers actually understand.

**Examples:**
- `Hi` → `01001000 01101001`
- `ABC` → `01000001 01000010 01000011`
- `Flag` → `01000110 01101100 01100001 01100111`

**Try it yourself:**
- CyberChef Binary: https://gchq.github.io/CyberChef/#recipe=From_Binary('Space',8)
- Online Binary Decoder: https://www.rapidtables.com/convert/number/binary-to-ascii.html

---

### Morse Code
Represents letters as combinations of dots and dashes.

**Examples:**
- `HELLO` → `.... . .-.. .-.. ---`
- `SOS` → `... --- ...`

**Variants:**
- **Fractionated Morse**: Convert to ternary then substitute
- **Morse + Numbers**: Include digits 0-9
- **International Morse**: Standard version

**Try it yourself:**
- CyberChef Morse: https://gchq.github.io/CyberChef/#recipe=From_Morse_Code
- Practice: `- .... .. ... / .. ... / -- --- .-. ... . / -.-. --- -.. .`

### Base32 and Other Bases
Alternative base encodings with different character sets.

**Base32 Example:**
- Character set: A-Z, 2-7 (32 characters total)
- `HELLO` → `JBSWY3DP`
- Often used in TOTP authentication codes

**Base58 (Bitcoin):**
- Excludes confusing characters: 0, O, I, l
- Used in Bitcoin addresses and IPFS hashes
- `HELLO` → `9Ajdvzr`

**Base85 (ASCII85):**
- Uses 85 printable ASCII characters  
- More compact than Base64
- `HELLO` → `87cURD`

**Try it yourself:**
- CyberChef Base32: https://gchq.github.io/CyberChef/#recipe=From_Base32('A-Z2-7%3D',true)
- Practice all bases with same input to see differences

---

## Practice Challenges

### Challenge 1: Multi-Step Decoding
```
V1c5MUlFZHZkQ0JwZENFPQ==
```
Hint: This has been encoded twice with the same method.

### Challenge 2: Mixed Encoding
```
54 6d 6c 6a 5a 53 42 72 5a 57 56 77 49 47 64 76 61 57 35 6e
```
Hint: Two different encoding methods used here.

### Challenge 3: Caesar
```
WKLV PHVVDJH KDV EHHQ HQFRGHG ZLWK D VSHFLDO PHWKRG
```
Hint: Bruteforce
---

## CyberChef - The Swiss Army Knife

**Website:** https://gchq.github.io/CyberChef/