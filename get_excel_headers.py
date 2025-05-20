import os
import pandas as pd
import sys

def get_excel_headers(directory_path):
    all_unique_headers = set()

    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at {directory_path}", file=sys.stderr)
        return []

    for filename in os.listdir(directory_path):
        if filename.endswith(".xlsx"):
            file_path = os.path.join(directory_path, filename)
            try:
                # Read only the header row to get column names
                df_header = pd.read_excel(file_path, nrows=0, engine='openpyxl')
                headers = df_header.columns.tolist()
                for header in headers:
                    # Ensure header is a string and strip any leading/trailing whitespace
                    all_unique_headers.add(str(header).strip())
            except Exception as e:
                print(f"Error reading headers from {file_path}: {e}", file=sys.stderr)
    
    return sorted(list(all_unique_headers))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_directory = sys.argv[1]
        unique_headers = get_excel_headers(data_directory)
        # Print each header on a new line for better readability if it's a long list
        for header in unique_headers:
            print(header)
    else:
        print("Usage: python get_excel_headers.py <directory_path>", file=sys.stderr) 