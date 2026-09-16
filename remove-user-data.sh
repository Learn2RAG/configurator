#!/bin/sh
set -eu

TARGET_DIR="$HOME/.local/share/Learn2RAG"
FORCE=0

# Check for -y or --yes flag when we call it from unistall will send -y
for arg in "$@"; do
    case "$arg" in
        -y|--yes) FORCE=1 ;;
    esac
done

if [ ! -d "$TARGET_DIR" ]; then
    echo "No user data found at $TARGET_DIR."
    exit 0
fi

if [ "$FORCE" -ne 1 ]; then
    printf "WARNING: This will permanently delete all user data in %s.\nAre you sure? (y/N): " "$TARGET_DIR"
    read -r CONFIRM
    case "$CONFIRM" in
        [yY][eE][sS]|[yY]) ;;
        *)
            echo "Aborted. No data was deleted."
            exit 1
            ;;
    esac
fi

echo "Deleting user data..."
rm -rf "$TARGET_DIR"
echo "User data removed."