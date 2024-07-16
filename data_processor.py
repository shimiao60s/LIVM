import tensorflow as tf
import glob
import requests
import re
import subprocess
import os
import sys

# Ensure necessary paths as arguments
if len(sys.argv) < 4:
    print("Usage: python script.py <download_path> <output_path> <audio_output_path>")
    sys.exit(1)

download_path = sys.argv[1]
output_path = sys.argv[2]
audio_output_path = sys.argv[3]
segment_length = 15  # Default segment length in seconds

# Check and create output directories if they don't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(audio_output_path):
    os.makedirs(audio_output_path)

# Load dataset and define feature descriptions
filenames = [YOUR_TFRECORD_FILES_HERE]  # User should replace placeholder with actual TFRecord file paths
raw_dataset = tf.data.TFRecordDataset(filenames)
feature_description = {
    'id': tf.io.FixedLenFeature([], tf.string),
    'labels': tf.io.VarLenFeature(tf.int64),
    'mean_rgb': tf.io.FixedLenFeature([1024], tf.float32),
    'mean_audio': tf.io.FixedLenFeature([128], tf.float32),
}

def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)

parsed_dataset = raw_dataset.map(_parse_function)

def get_full_video_id(short_id):
    url = f"http://data.yt8m.org/2/j/i/{short_id[:2]}/{short_id}.js"
    response = requests.get(url)
    if response.status_code == 200:
        match = re.search(r'i\("\w+",\s*"(\w+)"\);', response.text)
        if match:
            return match.group(1)
    else:
        print(f"Failed to retrieve data for {short_id}")
        return None

video_ids = []

# This id is for "music video"
TARGET_LABEL_ID = 14  

for parsed_record in parsed_dataset:
    labels = tf.sparse.to_dense(parsed_record['labels'])
    if TARGET_LABEL_ID in labels.numpy():
        short_video_id = parsed_record['id'].numpy().decode("utf-8")
        full_video_id = get_full_video_id(short_video_id)
        if full_video_id:
            video_ids.append(full_video_id)
        if len(video_ids) >= 1500:
            break

# Save the video IDs to a file
with open("video_ids.txt", "w") as file:
    for video_id in video_ids:
        file.write(f"{video_id}\n")

# Video processing function using FFmpeg
def process_video_segment_ffmpeg(video_id):
    video_path = os.path.join(download_path, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        print(f"Video file not found for ID {video_id}")
        return

    try:
        duration_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
        result = subprocess.run(duration_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        duration = float(result.stdout)

        segment_count = 0

        for start in range(0, int(duration), segment_length):
            end = start + segment_length
            if end > duration:
                end = duration
            if end - start < segment_length:
                continue

            segment_filename = os.path.join(output_path, f"{video_id}_{segment_count:02d}.mp4")

            if not os.path.exists(segment_filename):
                cmd = ["ffmpeg", "-i", video_path, "-ss", str(start), "-t", str(segment_length), "-c", "copy", segment_filename]
                subprocess.run(cmd)
                segment_count += 1

    except Exception as e:
        print(f"An error occurred for video {video_id}: {str(e)}")

# Main execution loop
for video_id in video_ids:
    process_video_segment_ffmpeg(video_id)

print("All videos have been processed.")

# Function to extract audio from video segments
def extract_audio_from_segment(segment_path):
    try:
        base_name = os.path.basename(segment_path)
        audio_filename = os.path.splitext(base_name)[0] + ".wav"
        audio_filepath = os.path.join(audio_output_path, audio_filename)

        if not os.path.exists(audio_filepath):
            audio_cmd = ["ffmpeg", "-i", segment_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", audio_filepath]
            subprocess.run(audio_cmd)

    except Exception as e:
        print(f"An error occurred while extracting audio: {str(e)}")

segment_files = glob.glob(os.path.join(output_path, "*.mp4"))
for segment_file in segment_files:
    extract_audio_from_segment(segment_file)

print("All audio extraction completed.")
