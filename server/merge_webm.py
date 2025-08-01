import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
import tempfile

def parse_timestamp_from_filename(filename):
    pattern = r'ScreenRecording_File_(\d{8})_(\d{6})_.*\.webm'
    match = re.search(pattern, filename)
    
    if match:
        date_str = match.group(1)  # YYYYMMDD
        time_str = match.group(2)  # HHMMSS
        
        # Parse to datetime
        datetime_str = f"{date_str}_{time_str}"
        return datetime.strptime(datetime_str, "%Y%m%d_%H%M%S")
    return None

def group_videos_by_time_gap(video_files, max_gap_minutes=10):
    if not video_files:
        return []
    
    # Sort by timestamp
    sorted_videos = sorted(video_files, key=lambda x: x[1])
    
    groups = []
    current_group = [sorted_videos[0]]
    
    for i in range(1, len(sorted_videos)):
        current_file, current_time = sorted_videos[i]
        prev_file, prev_time = sorted_videos[i-1]
        
        time_diff = (current_time - prev_time).total_seconds() / 60
        
        if time_diff <= max_gap_minutes:
            current_group.append(sorted_videos[i])
        else:
            groups.append(current_group)
            current_group = [sorted_videos[i]]
    
    groups.append(current_group)
    return groups

def merge_videos_with_ffmpeg(video_group, output_path):
    # If only one video then copy it
    if len(video_group) == 1:
        subprocess.run(['cp', video_group[0][0], output_path], check=True)
        return
    
    # Create temporary file list for ffmpeg
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for video_file, _ in video_group:
            f.write(f"file '{video_file}'\n")
        temp_file_list = f.name
    
    try:
        # Use ffmpeg to concatenate videos
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0', 
            '-i', temp_file_list, 
            '-c', 'copy',  
            '-y',  
            output_path
        ]
        subprocess.run(cmd, check=True)
    finally:
        # Clean up temporary file
        os.unlink(temp_file_list)

def process_webm_files(input_folder):
    input_path = Path(input_folder)
    merged_dir = input_path / "merged"
    merged_dir.mkdir(exist_ok=True)
    
    webm_files = []
    for file_path in input_path.glob("*.webm"):
        timestamp = parse_timestamp_from_filename(file_path.name)
        if timestamp:
            webm_files.append((str(file_path), timestamp))
    
    if not webm_files:
        print("No valid WebM files found in the specified format.")
        return
    
    print(f"Found {len(webm_files)} WebM files")
    
    video_groups = group_videos_by_time_gap(webm_files)
    print(f"Created {len(video_groups)} groups based on time gaps")
    
    for i, group in enumerate(video_groups):
        if not group:
            continue
            
        start_time = min(video[1] for video in group)
        end_time = max(video[1] for video in group)
        
        start_str = start_time.strftime("%Y%m%d_%H%M%S")
        end_str = end_time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"merged_{start_str}_to_{end_str}.webm"
        output_path = merged_dir / output_filename
        
        print(f"Merging group {i+1}: {len(group)} files -> {output_filename}")
        print(f"Time range: {start_time} to {end_time}")
        
        try:
            merge_videos_with_ffmpeg(group, str(output_path))
            print(f"  Successfully created: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"  Error merging group {i+1}: {e}")
        except Exception as e:
            print(f"  Unexpected error for group {i+1}: {e}")

if __name__ == "__main__":
    # Change input folder path containing WebM files
    input_folder = "/Users/sujeethshingade/Downloads"
    
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except subprocess.CalledProcessError:
        print("Error: ffmpeg is not installed")
        exit(1)
    except FileNotFoundError:
        print("Error: ffmpeg is not installed")
        exit(1)
    
    print(f"Processing WebM files in: {input_folder}")
    process_webm_files(input_folder)
    print("Processing complete!")