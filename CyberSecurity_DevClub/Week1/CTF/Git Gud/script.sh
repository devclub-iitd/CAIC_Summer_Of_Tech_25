#!/bin/bash
set -e

CHALLENGE_DIR="git_forensics_challenge"
FLAG_CONTENT="dcCTF{g17_0bj3ct5_n3v3r_d13}"
FLAG_FILE="flag.txt"

echo "[*] Creating challenge in: $CHALLENGE_DIR"
rm -rf "$CHALLENGE_DIR"
mkdir "$CHALLENGE_DIR"
cd "$CHALLENGE_DIR"

# Init a Git repo
git init &> /dev/null

# First commit: some junk
echo "This is a sample README." > README.md
git add README.md
git commit -m "Initial commit" &> /dev/null

# Add flag in a commit
echo "$FLAG_CONTENT" > "$FLAG_FILE"
git add "$FLAG_FILE"
git commit -m "Add sensitive file" &> /dev/null

# Remove flag and commit something else
git rm "$FLAG_FILE"
git commit -m "Remove flag" &> /dev/null

# Make a couple of distraction commits
echo "Just some boring changes." >> README.md
git add README.md
git commit -m "Update README again" &> /dev/null

# Danger: remove all reflogs and unreachable commits
# This is optional – skip if you want it easier
git reflog expire --expire=now --all
git gc --prune=now --aggressive

cd ..
zip -r "${CHALLENGE_DIR}.zip" "$CHALLENGE_DIR" > /dev/null
echo "[+] Challenge zipped as ${CHALLENGE_DIR}.zip"

