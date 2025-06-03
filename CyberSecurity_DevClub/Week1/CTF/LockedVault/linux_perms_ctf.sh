#!/bin/bash

# Linux Permissions CTF Challenge - Scripts with encoded content
# Challenge: "LockedVault"

set -e

CHALLENGE_DIR="/home/aryan/dcCTF/challenges/LockedVault"
FLAG="dcCTF{y0u_4r3_4_p3rm1ss10n_h4ck3r_n0w}"

echo "Setting up LockedVault CTF Challenge..."

# Clean up and create challenge directory
rm -rf "$CHALLENGE_DIR"
mkdir -p "$CHALLENGE_DIR"
cd "$CHALLENGE_DIR"

echo "=== Creating challenge scripts ==="

# Script 1 - Cannot be read initially, contains encoded password
cat > step1.sh << 'EOF'
#!/bin/bash
# First step - decode the password
if [ "$EUID" -eq 0 ]; then
    echo "ERROR: Don't run this as root/sudo!"
    echo "Run as: bash $0"
    exit 1
fi
password="dW5sb2NrMTIz"
echo "Step 1 complete!"
echo "Password found: $(echo $password | base64 -d)"
echo "Now make step2.sh executable and run it with this password!"
EOF

# Script 2 - Cannot be executed initially, requires password from step1
cat > step2.sh << 'EOF'
#!/bin/bash
if [ "$EUID" -eq 0 ]; then
    echo "ERROR: Don't run this as root/sudo!"
    echo "Run as: bash $0"
    exit 1
fi
echo "Enter the password from step1:"
read -r input
if [ "$(printf $input | base64)" = "dW5sb2NrMTIz" ]; then
    echo "Correct! Moving to final step..."
    key="ZmluYWxfa2V5XzQ1Ng=="
    echo "Key for step3: $(echo $key | base64 -d)"
else
    echo "Wrong password! Go back to step1."
    exit 1
fi
EOF

# Script 3 - Cannot be read initially, contains encoded flag that needs key
cat > step3.sh << 'EOF'
#!/bin/bash
if [ "$EUID" -eq 0 ]; then
    echo "ERROR: Don't run this as root/sudo!"
    echo "Run as: bash $0"
    exit 1
fi
echo "Enter the key from step2:"
read -r input
if [ "$(printf $input | base64)" = "ZmluYWxfa2V5XzQ1Ng==" ]; then
    # XOR encoded flag with key
    encoded_flag="ZGNDVEZ7eTB1XzRyM180X3Azcm0xc3MxMG5faDBja2NyX24wd30="
    echo "Vault unlocked!"
    echo "Flag: $(echo $encoded_flag | base64 -d)"
else
    echo "Wrong key! Complete the previous steps first."
    exit 1
fi
EOF

# README explaining the challenge
cat > README.txt << 'EOF'
LOCKED VAULT CHALLENGE

You have 3 scripts but they have restrictive permissions:
- step1.sh: Cannot be read (need to fix permissions to see the password)
- step2.sh: Cannot be executed (need to make executable and use password from step1)  
- step3.sh: Cannot be read (need to fix permissions and use key from step2)

Each script checks for the correct input from the previous step.
Even if you bypass permissions with sudo, you still need to solve each step properly!

Start by fixing permissions on step1.sh and reading it.
EOF

echo "=== Setting permissions normally for archive creation ==="
chmod 755 "$CHALLENGE_DIR"
chmod 644 step1.sh step2.sh step3.sh README.txt

echo "=== Creating archive ==="
cd /home/aryan/dcCTF/challenges/
tar -czf LockedVault.tar.gz LockedVault/

echo "=== Setting challenge permissions in archive ==="
cd /tmp
tar -xzf /home/aryan/dcCTF/challenges/LockedVault.tar.gz

# Set restrictive permissions
chmod 000 LockedVault/step1.sh     # Cannot read (contains encoded password)
chmod 644 LockedVault/step2.sh     # Cannot execute (needs password check)
chmod 000 LockedVault/step3.sh     # Cannot read (contains encoded flag)
chmod 644 LockedVault/README.txt   # Readable instructions

# Repack with challenge permissions
tar -czf /home/aryan/dcCTF/challenges/LockedVault.tar.gz LockedVault/
rm -rf LockedVault/

echo "=== Challenge Complete! ==="
echo ""
echo "Archive: /home/aryan/dcCTF/challenges/LockedVault.tar.gz"
echo ""
echo "=== Why sudo won't help ==="
echo "- Even with sudo, scripts require correct passwords/keys"
echo "- Passwords are base64 encoded in scripts"
echo "- Each step validates input from previous step"
echo "- Must solve sequentially to get the flag"
echo ""
echo "=== Intended Solution ==="
echo "1. chmod +r step1.sh && cat step1.sh"
echo "2. Decode base64 password: echo 'dW5sb2NrMTIz' | base64 -d"
echo "3. chmod +x step2.sh && ./step2.sh (enter: unlock123)"
echo "4. Decode base64 key: echo 'ZmluYWxfa2V5XzQ1Ng==' | base64 -d"  
echo "5. chmod +r step3.sh && cat step3.sh"
echo "6. chmod +x step3.sh && ./step3.sh (enter: final_key_456)"
echo ""
echo "Flag: $FLAG"