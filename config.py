import os

# Tự động lấy thư mục gốc chứa file config.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Trỏ vào thư mục data
DATA_DIR = os.path.join(BASE_DIR, "data")

W_CLIP = 0.5
W_OCR = 0.2
W_OD = 0.3

# Đường dẫn cụ thể tới các thư mục con
KEYFRAME_DIR = os.path.join(DATA_DIR, "keyframes")
OBJECTS_DIR = os.path.join(DATA_DIR, "objects")
OCR_DIR = os.path.join(DATA_DIR, "OCR_data")
CLIP_FEATURES_DIR = os.path.join(DATA_DIR, "clip-features-32")
MAP_KEYFRAMES_DIR = os.path.join(DATA_DIR, "map-keyframes")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
