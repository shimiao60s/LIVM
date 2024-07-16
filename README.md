
## Dataset for TIVM audio-video matching model

The dataset, including both audio and video, has been segmented and is available for download from Google Drive. You can access and download the segmented data using the link below:

- **Download Segmented Audio and Video Data**:
  [Segmented Data Download](https://drive.google.com/drive/folders/1ZuDgWVb_aSlcsM8TgVTlyBZql82Z8n8F?usp=sharing)

Additionally, you can use our data_processor.py to download, segment, and customize the amount of data according to your needs. 

## For data_processor.py： 
### Video and Audio Processing Toolkit

This toolkit processes video files to segment them and extract audio. It leverages TensorFlow to handle TFRecord datasets and uses FFmpeg for video and audio manipulation.

### Features

- Extract specific video IDs based on TensorFlow record datasets.
- Segment videos into smaller clips.
- Extract audio from video segments.

### Prerequisites

- Python 3.x
- FFmpeg installed on your system (required for video and audio processing)

## Contact

If you have any questions or encounter issues using the dataset or the code, feel free to contact us through GitHub Issues or send an email to [sliu823@gatech.edu].
