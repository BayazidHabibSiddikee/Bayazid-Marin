import os
import subprocess
import re

def clean_skipped_docs():
    print("Evaluating current knowledge base output...")
    # Fixed the keyword argument typo here
    result = subprocess.run(['./check_corrupted_files.sh'], capture_output=True, text=True, check=True)
    
    skipped_pattern = re.compile(r'✗\s*(.*?):\s*No valid text chunks')
    deleted_count = 0
    
    for line in result.stdout.split('\n'):
        match = skipped_pattern.search(line)
        if match:
            filename = match.group(1).strip()
            # Pointing explicitly inside the doc/ subdirectory
            file_path = os.path.join("doc", filename) 
            
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"🗑️ Deleted: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"❌ Failed to delete {file_path}: {e}")
            else:
                print(f"⚠️ File not found inside doc/: {filename}")
                
    print(f"\n✨ Done! Successfully removed {deleted_count} unindexable documents.")

if __name__ == "__main__":
    clean_skipped_docs()
