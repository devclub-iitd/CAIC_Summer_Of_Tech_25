#!/bin/bash
set -euo pipefail

TEMP_FLAG_COPY="flag.png"
OUTPUT_FILE="challenge.img"
OUTPUT_FILE_SIZE_MB=10
MOUNT_DIR="./mnt_loop"

truncate -s "${OUTPUT_FILE_SIZE_MB}M" "$OUTPUT_FILE"
sudo mkfs.ext4 -F "$OUTPUT_FILE" &> /dev/null
mkdir -p "$MOUNT_DIR"
sudo mount -o loop "$OUTPUT_FILE" "$MOUNT_DIR"
sudo cp "$TEMP_FLAG_COPY" "$MOUNT_DIR/flag.png"
sudo rm "$MOUNT_DIR/flag.png"
sync
sudo umount "$MOUNT_DIR"
rmdir "$MOUNT_DIR"
