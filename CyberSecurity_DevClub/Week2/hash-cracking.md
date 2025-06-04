# Hash Functions & Password Cracking

## Understanding Hashes

Hash functions are mathematical one-way streets - easy to go forward, nearly impossible to reverse.

**Key Properties:**
- **Deterministic**: Same input always produces same output
- **Fixed size**: "hello" and a 1GB file both produce same length hash
- **Avalanche effect**: Changing one bit changes ~50% of the hash
- **One-way**: Computationally infeasible to reverse

**Real-world example:**
Instead of storing your password "password123", websites store something like `ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f`. Even if hackers steal the database, they can't directly see your password!

---

## Common Hash Types

### MD5 (Message Digest 5)
- **Length**: 128-bit (32 hexadecimal characters)
- **Status**: Cryptographically broken - don't use for security!
- **Still used for**: File integrity checks, non-security applications

**Examples:**
- `hello` → `5d41402abc4b2a76b9719d911017c592`
- `password` → `5f4dcc3b5aa765d61d8327deb882cf99`
- `admin` → `21232f297a57a5a743894a0e4a801fc3`
- `123456` → `e10adc3949ba59abbe56e057f20f883e`

### SHA-1 (Secure Hash Algorithm 1)
- **Length**: 160-bit (40 hexadecimal characters)
- **Status**: Cryptographically broken as of 2017
- **Still found in**: Legacy systems

**Examples:**
- `hello` → `aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d`
- `password` → `5baa61e4c9b93f3f0682250b6cf8331b7ee68fd8`
- `admin` → `d033e22ae348aeb5660fc2140aec35850c4da997`

### SHA-256 (Secure Hash Algorithm 256)
- **Length**: 256-bit (64 hexadecimal characters)
- **Status**: Currently secure and widely used
- **Used in**: Bitcoin, TLS certificates, modern applications

**Examples:**
- `hello` → `2cf24dba4f21d4288094f5c0b48e8151b142fc3d2dc85e0b9ac49a7295b1a14`
- `password` → `5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8`
- `flag` → `327a6c4304ad5938eaf0efb6cc3e53dc9d42a2e9a8d5f3bc84b991ce23e92da2`

### SHA-512 (Secure Hash Algorithm 512)
- **Length**: 512-bit (128 hexadecimal characters)
- **Status**: Very secure but slower
- **Used when**: Maximum security is needed

**Examples:**
- `securepassword` → `54c8e9ed836eb9622f6694876dabd83e44c6f7ce11cb97c9be368eaac9edc7cd3b8a78888129018ec4bdf2a2d4d83c6b7cae722c22615e9e1cf309c3e3e12ad2`
---

## Hash Identification

### Quick Identification by Length
- **32 characters**: MD5 or NTLM (Windows passwords)
- **40 characters**: SHA-1
- **64 characters**: SHA-256
- **128 characters**: SHA-512

### Practice Examples
Try identifying these hashes:
1. `5d41402abc4b2a76b9719d911017c592`
2. `aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d`
3. `2cf24dba4f21d4288094f5c0b48e8151b142fc3d2dc85e0b9ac49a7295b1a14`

**Online hash identifiers:**
- HashID Online: https://hashes.com/en/tools/hash_identifier
- Tunnels Up Hash Analyzer: https://www.tunnelsup.com/hash-analyzer/

---

## Password Cracking Concepts

### Why Hash Cracking Works
Since you can't reverse a hash, you have to guess the original password and check if it produces the same hash. This is like trying to figure out what word someone was thinking of by comparing hashes.

### Attack Methods

**1. Dictionary Attack:**
- Try common passwords from lists
- Fast but limited to known passwords

**Common passwords that are easily cracked:**
- `password` → `5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8`
- `123456` → `e10adc3949ba59abbe56e057f20f883e`
- `qwerty` → `d8578edf8458ce06fbc5bb76a58c5ca4`

**2. Brute Force:**
- Try every possible combination
- Guaranteed to work eventually
- Extremely slow for long passwords

**3. Mask Attack:**
- Brute force with patterns
- Example: "8 chars, starts with capital, ends with number"

**4. Rule-based Attack:**
- Take dictionary words and apply rules
- `password` → `Password123`, `p@ssw0rd`, `PASSWORD!`

---

## Online Hash Cracking

### Free Cracking Services

**CrackStation** - https://crackstation.net/
- Massive precomputed hash database
- Supports MD5, SHA1, SHA256, and more
- Free and fast for common passwords

**MD5 Decrypt** - https://md5decrypt.net/
- Specializes in MD5 hashes
- Good success rate for common passwords

**Hash Killer** - https://hashkiller.io/
- Multiple hash types supported
- Good for CTF challenges

### Practice Examples

Try cracking these hashes online:

**Easy (Common Passwords):**
1. `5d41402abc4b2a76b9719d911017c592` (MD5)
2. `356a192b7913b04c54574d18c28d46e6395428ab` (SHA-1)
3. `e10adc3949ba59abbe56e057f20f883e` (MD5)

**Medium (Dictionary Words):**
1. `8b1a9953c4611296a827abf8c47804d7` (MD5)
2. `5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5` (SHA-256)

**Hard (May require rules/mutations):**
1. `098f6bcd4621d373cade4e832627b4f6` (MD5)
2. `ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f` (SHA-256)

---

## Password Security Best Practices

### How Secure Systems Store Passwords

**Salting:**
- Add random data to passwords before hashing
- Prevents rainbow table attacks
- Example: `password + salt123 → hash`

**Multiple Iterations:**
- Hash the result multiple times
- Makes brute force attacks much slower
- Modern algorithms: bcrypt, scrypt, Argon2

**Example of Secure Password Storage:**
```
Original: mypassword123
Salt: randomsalt456
Iterations: 10,000
Final Hash: bcrypt(mypassword123 + randomsalt456, 10000)
```

### Real-World Hash Usage

**File Integrity:**
- SHA-256 checksums verify file downloads
- Git uses SHA-1 for version control
- Package managers use hashes to detect tampering

**Digital Signatures:**
- HTTPS certificates use hash functions
- Code signing for software verification
- Blockchain proof-of-work systems

**Password Authentication:**
- Passwords stored as hashes
- When you try to login, the service adds the salt and hashes it again, and if the hashes match, it means you entered the correct password

---

## Popular Cracking Tools (Local Installation)

### Hashcat - GPU Accelerated
- Extremely fast with modern graphics cards
- Supports hundreds of hash types
- Advanced attack modes and rules

### John the Ripper - CPU Based
- Classic password cracking tool
- Good for older systems
- Excellent rule-based attacks

### Online vs Local Tools

**Use Online Tools When:**
- Learning and practicing
- Common passwords and hashes
- Quick verification during CTFs

**Use Local Tools When:**
- Custom wordlists needed
- Advanced attack techniques
- Professional penetration testing

---

## CTF Hash Cracking Strategy

### Step-by-Step Approach

1. **Identify the hash type** using online tools
2. **Try online databases** first (fastest)
3. **Use common wordlists** (rockyou.txt, common passwords)
4. **Apply password rules** (capitalization, numbers, symbols)
5. **Try mask attacks** for short passwords
6. **Custom wordlists** based on challenge context

### Common CTF Hash Examples

**Beginner Level:**
- Common passwords: admin, password, 123456 etc
- Simple variations: Admin123, Password!

**Intermediate Level:**
- Dictionary words with mutations
- Names and dates related to challenge theme
- Base64 encoded passwords

**Advanced Level:**
- Custom algorithms or encoding
- Multiple layers of hashing
- Context-specific wordlists needed

---

## Practice Challenges

### Crack these hashes:

```
1. 4a4be40c96ac6314e91d93f38043a634
2. bb3508569b89cdbabb7e5bea39cf09162dfe9c91
3. 7580eb01a6f11edaee3698eebb1c808e07377917a40a309a683122d5dfaffeac
```