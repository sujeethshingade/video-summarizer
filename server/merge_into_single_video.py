import os
import re
import subprocess
from datetime import datetime

def extract_timestamp(filename):
    pattern = r'ScreenRecording_File_(\d{8})_(\d{6})_'
    match = re.search(pattern, filename)
    if match:
        date_str, time_str = match.groups()
        timestamp_str = f"{date_str}_{time_str}"
        try:
            return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except ValueError as e:
            print(f"Error parsing timestamp for {filename}: {e}")
    return None

def main():
    # Prompt user for directory
    directory = input("Enter the path to the directory containing WebM files: ").strip()
    if not os.path.isdir(directory):
        print("Invalid directory path.")
        return

    # Get all webm files in the specified directory
    webm_files = [f for f in os.listdir(directory) if f.endswith('.webm')]
    if not webm_files:
        print("No WebM files found in the directory.")
        return
    
    print(f"Found {len(webm_files)} WebM files.")

    # Extract timestamps and create list of (filename, timestamp) tuples
    files_with_timestamps = []
    invalid_files = []
    
    for file in webm_files:
        timestamp = extract_timestamp(file)
        if timestamp:
            files_with_timestamps.append((file, timestamp))
        else:
            invalid_files.append(file)
    
    if invalid_files:
        print(f"Warning: Could not extract timestamps from {len(invalid_files)} files:")
        for f in invalid_files:
            print(f"  - {f}")
    
    # Sort files by timestamp
    sorted_files = sorted(files_with_timestamps, key=lambda x: x[1])
    
    if not sorted_files:
        print("No files with valid timestamps found.")
        return
    
    # Print the sorted order for verification
    print("\nFiles will be concatenated in the following order:")
    for i, (file, timestamp) in enumerate(sorted_files, 1):
        print(f"{i}. {file} - {timestamp}")
    
    # Confirm with user
    confirm = input("\nDoes this order look correct? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Operation cancelled.")
        return
    
    # Create a file list for ffmpeg with full paths
    concat_list_path = os.path.join(directory, 'concat_list.txt')
    with open(concat_list_path, 'w') as f:
        for file, _ in sorted_files:
            # Use absolute paths in the concat file
            abs_path = os.path.abspath(os.path.join(directory, file))
            f.write(f"file '{abs_path}'\n")
    
    # Output filename with current timestamp
    output_file = os.path.join(directory, f"combined_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.webm")
    
    # Run ffmpeg to concatenate the files
    try:
        print("\nStarting video concatenation with ffmpeg...")
        subprocess.run([
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_list_path,
            '-c', 'copy',
            output_file
        ], check=True)
        print(f"Successfully created combined video: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg execution: {e}")
    except FileNotFoundError:
        print("ffmpeg not found. Please make sure ffmpeg is installed and in your PATH.")
    
    # Clean up the temporary file
    os.remove(concat_list_path)

if __name__ == "__main__":
    main()