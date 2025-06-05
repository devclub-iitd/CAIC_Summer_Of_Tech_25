# Modern Cryptography

Modern cryptography uses mathematical algorithms that are computationally secure - meaning they would take thousands of years to break with current technology.

---

## Symmetric Encryption

### AES (Advanced Encryption Standard)
The gold standard for symmetric encryption, used everywhere from HTTPS to file encryption.

**Key Properties:**
- Block cipher (encrypts 16-byte blocks at a time)
- Key sizes: 128, 192, or 256 bits
- Extremely fast and secure

**AES Modes - ECB vs CBC:**
- **ECB (Electronic Codebook)**: Same plaintext blocks → same ciphertext blocks (**Never use!**)
- **CBC (Cipher Block Chaining)**: Uses IV and chains blocks (secure)

**Try AES Online:**
- CyberChef AES: https://gchq.github.io/CyberChef/#recipe=AES_Encrypt
- Online AES Tool: https://www.devglan.com/online-tools/aes-encryption-decryption

### DES and 3DES (Legacy)
- **DES**: 56-bit key (broken in 1990s) - don't use!
- **3DES**: Applies DES three times, still in legacy banking

---

## Asymmetric Encryption (Public Key Cryptography)

### RSA (Rivest-Shamir-Adleman)
Based on difficulty of factoring large numbers.

**Key Generation Example:**
```
1. Choose primes: p = 61, q = 53
2. Calculate n = p × q = 3233
3. Calculate φ(n) = (p-1)(q-1) = 3120
4. Choose e = 17, calculate d = 2753
Result: Public (n=3233, e=17), Private (n=3233, d=2753)
```

**Encryption Example:**
```
Message "HI" = 7273
Encrypt: c = 7273^17 mod 3233 = 1570
Decrypt: m = 1570^2753 mod 3233 = 7273
```

**Try RSA Online:**
- RSA Calculator: https://www.cs.drexel.edu/~jpopyack/Courses/CSP/Fa17/notes/10.1_Cryptography/RSA_Express_EncryptDecrypt_v2.html

### Elliptic Curve Cryptography (ECC)
Same security as RSA with smaller keys (RSA-2048 ≈ ECC-224). Used in Bitcoin, Signal, modern TLS.

---

## How HTTPS Actually Works (Hybrid Systems)

Real-world systems combine symmetric and asymmetric encryption for efficiency.

**HTTPS Process:**
```
1. Client → Server: "Hello, I want secure connection"
2. Server → Client: "Here's my RSA public key and certificate"
3. Client generates: AES-256 key (random 32 bytes)
4. Client → Server: RSA_encrypt(AES_session_key, server_public_key)
5. Server decrypts AES key with RSA private key
6. Client ↔ Server: AES_encrypt(data, AES_session_key)
```

**Why Hybrid:** RSA is slow for large data, AES is fast but needs shared key.

---

## Digital Signatures & Certificates

### RSA Signatures
Prove authenticity and integrity without revealing private key.

**How It Works:**
```
Signing: signature = hash(message)^private_key mod n
Verification: hash = signature^public_key mod n
```

### Digital Certificates
A digital certificate is an electronic document that authenticates the identity of a user, device, server, or website using cryptography and public key infrastructure (PKI). Issued by a trusted certificate authority (CA), it enables secure data exchange and helps ensure that communications are confidential and only accessible to authorized parties.

- Trust infrastructure - Root CA vouches for intermediate CA vouches for website.

---

## Modern Cryptographic Attacks

### Padding Oracle Attacks
Exploit error messages from padding validation in CBC mode. Server reveals info through different error messages.

### Timing Attacks
```python
# Insecure password comparison
def check_password(input, correct):
    for i in range(len(input)):
        if input[i] != correct[i]:
            return False  # Returns immediately - timing leak!
    return True
```

---

## CTF-Relevant Modern Crypto

### Common CTF Scenarios

**Weak RSA Implementations:**
- Small prime factors (n = 143 = 11 × 13)
- Common modulus attack
- Low public exponent (e=3)

**AES Implementation Flaws:**
- ECB mode - patterns visible in encrypted images
- Key reuse in stream ciphers
- Weak random number generation

**Hash Extension Attacks:**
- Given hash(secret + known_data), can compute hash(secret + known_data + attacker_data)

### Online Tools
- **dCode RSA Decoder**: https://www.dcode.fr/rsa-cipher
- **RSA Calculator**: https://www.cs.drexel.edu/~jpopyack/Courses/CSP/Fa17/notes/10.1_Cryptography/RSA_Express_EncryptDecrypt_v2.html
- **Factorization**: http://factordb.com/
- **CyberChef**: https://gchq.github.io/CyberChef/
- **Hash Length Extension**: https://github.com/iagox86/hash_extender

---

## Real-World Applications

**Banking**: HSMs for key storage, RSA-2048 signatures, AES-256 encryption
**Messaging**: Signal (Curve25519), WhatsApp (Signal protocol), Telegram (MTProto)
**Blockchain**: Bitcoin (SHA-256, ECDSA), Ethereum (Keccak-256, secp256k1)
**Standards**: TLS 1.3, FIPS 140-2, NIST Post-Quantum Cryptography