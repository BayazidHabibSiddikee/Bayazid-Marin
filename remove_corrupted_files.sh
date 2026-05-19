#!/bin/bash
./check_corrupted_files.sh | grep '✗ ' | sed 's/[[:space:]]*✗[[:space:]]*//;s/: No valid text chunks after filtering//' | while read -r file; do
    # Target the file inside the doc directory
    target_file="doc/$file"
    if [ -f "$target_file" ]; then
        rm -v "$target_file"
    else
        echo "⚠️ Not found in doc/: $target_file"
    fi
done
