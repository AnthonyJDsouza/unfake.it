import cv2
import pandas as pd
import os
from PIL import Image as PILImage
import uuid
from tqdm import tqdm
# import huggingface_hub
import csv
import argparse


def frame_generator(input_folder, csv_path, output_dir, interval_sec = 0.5):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    labels_csv = os.path.join(output_dir, 'labels.csv')

    headers = ['video', 'image', 'label']

    with open(labels_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos into frames"):
            video_name = row['video']
            video_path = os.path.join(input_folder, video_name)
            if not os.path.exists(video_path):
                print(f"Error: {video_name} at {video_path} not found")
                continue

            label = row['label']

            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                print(f"Error: Could not find {video_name}")
                continue

            fps = video.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(fps * interval_sec))
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_count = 0
            while True:
                success, frame = video.read()
                if not success:
                    break

                if frame_count % frame_interval == 0:
                    frame_filename = f"{video_name[:-4]}_{uuid.uuid4().hex}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

                    # image = PILImage.open(frame_path)
                    # yield {
                    #     'video_name': video_name,
                    #     'image': image,
                    #     'label': label
                    # }
                    # print(f"processing {video_name} labelled {label}")
                    writer.writerow([f'{video_name}', f"{frame_path}", f"{label}"])
                frame_count += 1
                if frame_count >= total_frames:
                    break

            video.release()

# frame_generator('data/video/', 'data/video/dataset.csv', 'data/outputjpg/')

parser = argparse.ArgumentParser()

parser.add_argument('--csv', type=str, required=True)
parser.add_argument('--input_folder', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
args = parser.parse_args()

frame_generator(input_folder=args.input_folder, csv_path=args.csv, output_dir=args.output_dir)
