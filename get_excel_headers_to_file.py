import os
import pandas as pd
import sys

def get_excel_headers_to_file(directory_path, output_file_path):
    all_unique_headers = set()

    if not os.path.isdir(directory_path):
        error_message = f"Error: Directory not found at {directory_path}"
        print(error_message, file=sys.stderr)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(error_message + "\n")
        return

    for filename in os.listdir(directory_path):
        if filename.endswith(".xlsx"):
            file_path = os.path.join(directory_path, filename)
            try:
                df_header = pd.read_excel(file_path, nrows=0, engine='openpyxl')
                headers = df_header.columns.tolist()
                for header in headers:
                    all_unique_headers.add(str(header).strip())
            except Exception as e:
                print(f"Error reading headers from {file_path}: {e}", file=sys.stderr)
    
    sorted_headers = sorted(list(all_unique_headers))
    
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            if not sorted_headers:
                f.write("No headers found or error occurred.\n")
            for header in sorted_headers:
                f.write(header + "\n")
        print(f"Headers written to {output_file_path}")
    except Exception as e:
        print(f"Error writing headers to file {output_file_path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        data_dir = sys.argv[1]
        out_file = sys.argv[2]
        get_excel_headers_to_file(data_dir, out_file)
    else:
        print("Usage: python get_excel_headers_to_file.py <directory_path> <output_file_path>", file=sys.stderr) 