import os
import sys
import glob
import json
import csv
import re
import bisect
import pickle
from pathlib import Path
import numpy as np
import faiss

# Đảm bảo in tiếng Việt an toàn trên Windows console
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from config import DATA_DIR, CLIP_FEATURES_DIR, OBJECTS_DIR, OCR_DIR, MAP_KEYFRAMES_DIR, ASR_DIR

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 80
HNSW_EF_SEARCH = 64

CACHE_FAISS_PATH = os.path.join(DATA_DIR, "cache_faiss.index")
CACHE_META_CLIP_PATH = os.path.join(DATA_DIR, "cache_meta_clip.pkl")
CACHE_META_OD_PATH = os.path.join(DATA_DIR, "cache_meta_od.pkl")
CACHE_META_OCR_PATH = os.path.join(DATA_DIR, "cache_meta_ocr.pkl")
CACHE_MAPPER_PATH = os.path.join(DATA_DIR, "cache_keyframe_mapper.pkl")
CACHE_ASR_INDEX_PATH = os.path.join(DATA_DIR, "cache_asr_index.pkl")

# =====================================================================
# 1. HỆ THỐNG CHUYỂN ĐỔI ID VIDEO & KEYFRAME (THEO MAP-KEYFRAMES)
# =====================================================================
def _clean_video_id(raw_name):
    """Chuẩn hóa ID video, loại bỏ mọi phần mở rộng tệp tin."""
    name = str(raw_name)
    for ext in ['.npy', '.mp4', '.avi', '.mov', '.mkv', '.json', '.csv']:
        if name.endswith(ext):
            return name[:-len(ext)]
    return name

class KeyframeMapper:
    """
    Hệ thống chuyển đổi đa chiều ID Keyframe chuẩn AIC theo map-keyframes:
    - n: Số thứ tự keyframe (1, 2, 3...) -> tương ứng file ảnh 001.jpg, 002.jpg
    - frame_idx: Chỉ số frame tuyệt đối trong file video gốc .mp4 (0, 90, 261, 351...)
    - pts_time: Mốc thời gian thực theo giây trong video (0.0, 3.0, 8.7...)
    - pts_time_ms: Mốc thời gian thực theo mili-giây (0, 3000, 8700...)
    - fps: Tốc độ khung hình của video (25.0, 30.0...)
    """
    def __init__(self, map_keyframes_dir=MAP_KEYFRAMES_DIR, auto_load=True):
        self.map_keyframes_dir = map_keyframes_dir
        self.n_to_meta = {}
        self.fidx_to_n = {}
        self.timeline = {}
        
        if auto_load:
            self.load()

    def load(self, force_reload=False):
        if not force_reload and os.path.exists(CACHE_MAPPER_PATH):
            try:
                with open(CACHE_MAPPER_PATH, "rb") as f:
                    data = pickle.load(f)
                    self.n_to_meta = data["n_to_meta"]
                    self.fidx_to_n = data["fidx_to_n"]
                    self.timeline = data["timeline"]
                return
            except Exception as e:
                pass

        if not Path(self.map_keyframes_dir).exists():
            return

        csv_files = sorted(glob.glob(os.path.join(self.map_keyframes_dir, "*.csv")))
        self.n_to_meta = {}
        self.fidx_to_n = {}
        self.timeline = {}

        for file_path in csv_files:
            video_id = _clean_video_id(os.path.basename(file_path))
            n_map = {}
            fidx_map = {}
            time_list = []

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        n = int(row["n"])
                        pts_time = float(row["pts_time"])
                        frame_idx = int(row["frame_idx"])
                        fps = float(row.get("fps", 25.0))

                        n_map[n] = {
                            "pts_time": pts_time,
                            "frame_idx": frame_idx,
                            "fps": fps
                        }
                        fidx_map[frame_idx] = n
                        time_list.append((pts_time, n, frame_idx))

                time_list.sort(key=lambda x: x[0])
                self.n_to_meta[video_id] = n_map
                self.fidx_to_n[video_id] = fidx_map
                self.timeline[video_id] = time_list
            except Exception:
                pass

        try:
            with open(CACHE_MAPPER_PATH, "wb") as f:
                pickle.dump({
                    "n_to_meta": self.n_to_meta,
                    "fidx_to_n": self.fidx_to_n,
                    "timeline": self.timeline
                }, f)
        except Exception:
            pass

    def get_info(self, video_id, n):
        """Lấy toàn bộ thông tin (n, frame_idx, pts_time, pts_time_ms, fps) của keyframe n."""
        v_key = _clean_video_id(video_id)
        v_map = self.n_to_meta.get(v_key)
        if v_map and int(n) in v_map:
            data = v_map[int(n)]
            return {
                "video_id": v_key,
                "n": int(n),
                "frame_idx": data["frame_idx"],
                "pts_time": data["pts_time"],
                "pts_time_ms": int(round(data["pts_time"] * 1000)),
                "fps": data["fps"]
            }
        pts_sec = float(n) / 25.0
        return {
            "video_id": v_key,
            "n": int(n),
            "frame_idx": int(n),
            "pts_time": pts_sec,
            "pts_time_ms": int(round(pts_sec * 1000)),
            "fps": 25.0
        }

    def n_to_frame_idx(self, video_id, n):
        """Đổi từ n (1, 2, 3...) sang frame_idx tuyệt đối trong file video mp4."""
        return self.get_info(video_id, n)["frame_idx"]

    def n_to_pts_time(self, video_id, n):
        """Đổi từ n sang mốc thời gian (giây)."""
        return self.get_info(video_id, n)["pts_time"]

    def n_to_pts_ms(self, video_id, n):
        """Đổi từ n sang mốc thời gian (mili-giây)."""
        return self.get_info(video_id, n)["pts_time_ms"]

    def frame_idx_to_n(self, video_id, frame_idx):
        """Đổi từ frame_idx sang n (keyframe ID gần nhất)."""
        v_key = _clean_video_id(video_id)
        fmap = self.fidx_to_n.get(v_key, {})
        if frame_idx in fmap:
            return fmap[frame_idx]
        
        t_list = self.timeline.get(v_key, [])
        if not t_list:
            return int(frame_idx)
        
        best_n = min(t_list, key=lambda item: abs(item[2] - frame_idx))[1]
        return best_n

    def pts_time_to_n(self, video_id, pts_time):
        """Đổi từ thời gian (giây) sang n (keyframe ID) bằng Binary Search."""
        v_key = _clean_video_id(video_id)
        t_list = self.timeline.get(v_key, [])
        if not t_list:
            return max(1, int(round(pts_time * 25)))

        times = [x[0] for x in t_list]
        idx = bisect.bisect_left(times, pts_time)
        if idx == 0:
            return t_list[0][1]
        if idx >= len(t_list):
            return t_list[-1][1]

        before = t_list[idx - 1]
        after = t_list[idx]
        if abs(before[0] - pts_time) <= abs(after[0] - pts_time):
            return before[1]
        return after[1]

    def format_submission_line(self, video_id, n, mode="frame_idx"):
        """Định dạng kết quả nộp bài thi AIC (DRES / VBS): video_id,frame_idx hoặc video_id,pts_time_ms."""
        info = self.get_info(video_id, n)
        vid = info["video_id"]
        if mode == "frame_idx":
            return f"{vid},{info['frame_idx']}"
        elif mode == "pts_time_ms":
            return f"{vid},{info['pts_time_ms']}"
        elif mode == "pts_time":
            return f"{vid},{info['pts_time']:.3f}"
        else:
            return f"{vid},{info['n']}"

mapper = KeyframeMapper()


# =====================================================================
# 2. MODULE TIỀN XỬ LÝ & LẬP CHỈ MỤC ASR (SPEECH TRANSCRIPT)
# =====================================================================
BOILERPLATE_PATTERNS = [
    r"hãy đăng ký kênh.*",
    r"đăng ký kênh để ủng hộ.*",
    r"cảm ơn quý vị đã theo dõi.*",
    r"nhớ like và subscribe.*",
    r"xin kính chào tạm biệt.*"
]

def clean_asr_text(text):
    """Làm sạch văn bản ASR tiếng Việt: loại bỏ quảng cáo, ký tự lạ và chuẩn hóa khoảng trắng."""
    if not text:
        return ""
    txt = str(text).lower()
    for pat in BOILERPLATE_PATTERNS:
        txt = re.sub(pat, "", txt, flags=re.IGNORECASE)
    txt = re.sub(r'[^\w\s\d]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt

def extract_keywords(text):
    """Tách từ đơn và cụm từ liền kề (unigram + bigram) để lập chỉ mục tìm kiếm."""
    cleaned = clean_asr_text(text)
    words = cleaned.split()
    tokens = set(words)
    for i in range(len(words) - 1):
        tokens.add(f"{words[i]} {words[i+1]}")
    return tokens

class ASRPreprocessor:
    """
    Tiền xử lý và lập chỉ mục Inverted Index cho ASR:
    - Làm sạch và chuẩn hóa toàn bộ lời thoại
    - Khớp mốc thời gian [start - 15s, end + 15s] với timeline của map-keyframes
    - Lưu chỉ mục cache_asr_index.pkl để truy vấn siêu tốc (< 1ms)
    """
    def __init__(self, asr_dir=ASR_DIR, map_keyframes_dir=MAP_KEYFRAMES_DIR):
        self.asr_dir = asr_dir
        self.mapper = KeyframeMapper(map_keyframes_dir=map_keyframes_dir)
        self.inverted_index = {}
        self.cleaned_asr_data = {}

    def process_all(self, padding_sec=15.0, save_cache=True):
        json_files = sorted(glob.glob(os.path.join(self.asr_dir, "*.json")))
        if not json_files:
            fallback = os.path.join(DATA_DIR, "asr_data")
            if Path(fallback).exists():
                json_files = sorted(glob.glob(os.path.join(fallback, "*.json")))
                self.asr_dir = fallback

        print(f"[*] Đang xử lý {len(json_files)} file ASR JSON...")

        total_captions = 0
        self.inverted_index = {}
        self.cleaned_asr_data = {}

        for fpath in json_files:
            video_id = _clean_video_id(os.path.basename(fpath))
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    captions = json.load(f)
            except Exception:
                continue

            cleaned_caps = []
            for cap in captions:
                raw_txt = cap.get("text", "")
                cleaned_txt = clean_asr_text(raw_txt)
                if not cleaned_txt:
                    continue

                start = float(cap.get("start", 0.0))
                end = float(cap.get("end", 0.0))
                
                cleaned_caps.append({
                    "start": start,
                    "end": end,
                    "text": cleaned_txt
                })
                total_captions += 1

                t_list = self.mapper.timeline.get(video_id, [])
                lo = max(0.0, start - padding_sec)
                hi = end + padding_sec

                matched_keyframes = [
                    item[1] for item in t_list if lo <= item[0] <= hi
                ]

                keywords = extract_keywords(cleaned_txt)
                for kw in keywords:
                    if kw not in self.inverted_index:
                        self.inverted_index[kw] = []
                    for kf_n in matched_keyframes:
                        self.inverted_index[kw].append((video_id, kf_n))

            self.cleaned_asr_data[video_id] = cleaned_caps

        for kw in self.inverted_index:
            self.inverted_index[kw] = list(set(self.inverted_index[kw]))

        print(f"[*] Đã lập chỉ mục xong {len(self.inverted_index):,} từ khóa từ {total_captions:,} câu thoại.")

        if save_cache:
            try:
                with open(CACHE_ASR_INDEX_PATH, "wb") as f:
                    pickle.dump({
                        "inverted_index": self.inverted_index,
                        "cleaned_asr_data": self.cleaned_asr_data
                    }, f)
                print(f"[*] Đã lưu cache ASR Inverted Index tại {CACHE_ASR_INDEX_PATH}")
            except Exception as e:
                print(f"[!] Lỗi lưu cache ASR: {e}")

        return self.inverted_index, self.cleaned_asr_data

    @staticmethod
    def load_cache():
        if os.path.exists(CACHE_ASR_INDEX_PATH):
            try:
                with open(CACHE_ASR_INDEX_PATH, "rb") as f:
                    data = pickle.load(f)
                return data["inverted_index"], data["cleaned_asr_data"]
            except Exception:
                pass
        return None, None


# =====================================================================
# 3. TỔNG HỢP TẤT CẢ CÔNG ĐOẠN TIỀN XỬ LÝ (BUILD ALL CACHES)
# =====================================================================
def build_all_caches():
    print("=" * 60)
    print("🚀 BẮT ĐẦU TIỀN XỬ LÝ TOÀN BỘ DỮ LIỆU AIC 2026")
    print("=" * 60)

    # 1. CLIP & FAISS
    print("\n[1/5] Xây dựng FAISS Vector Index (HNSW)...")
    npy_files = sorted(glob.glob(os.path.join(CLIP_FEATURES_DIR, "*.npy")))
    all_features = []
    metadata_clip = []
    key_to_faiss_id = {}

    for file_path in npy_files:
        try:
            features = np.load(file_path, allow_pickle=True).astype("float32")
            for frame_id, feature in enumerate(features, start=1):
                all_features.append(feature)
                metadata_clip.append((os.path.basename(file_path), frame_id))
        except Exception as e:
            print(f"Lỗi đọc {file_path}: {e}")

    if len(all_features) > 0:
        all_features = np.asarray(all_features, dtype="float32")
        faiss.normalize_L2(all_features)
        dim = all_features.shape[1]
        index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        index.hnsw.efSearch = HNSW_EF_SEARCH
        index.add(all_features)
        faiss.write_index(index, CACHE_FAISS_PATH)

        for internal_id, (raw_name, fid) in enumerate(metadata_clip):
            vkey = _clean_video_id(raw_name)
            key_to_faiss_id[(vkey, fid)] = internal_id

        with open(CACHE_META_CLIP_PATH, "wb") as f:
            pickle.dump({"all_features": all_features, "metadata_clip": metadata_clip, "key_to_faiss_id": key_to_faiss_id}, f)
        print(f" -> Đã nén {index.ntotal:,} vector CLIP vào {CACHE_FAISS_PATH}")

    # 2. Object Detection
    print("\n[2/5] Gộp dữ liệu Object Detection JSON...")
    metadata_OD = []
    if Path(OBJECTS_DIR).exists():
        for video_dir in Path(OBJECTS_DIR).iterdir():
            if not video_dir.is_dir(): continue
            for json_file in video_dir.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    metadata_OD.append({
                        "video": video_dir.name,
                        "frame_id": int(json_file.stem.split("_")[-1]),
                        "scores": data.get("detection_scores", []),
                        "classes": data.get("detection_class_entities", [])
                    })
                except Exception: pass
    with open(CACHE_META_OD_PATH, "wb") as f:
        pickle.dump(metadata_OD, f)
    print(f" -> Đã gộp {len(metadata_OD):,} frames OD vào {CACHE_META_OD_PATH}")

    # 3. OCR
    print("\n[3/5] Gộp dữ liệu OCR JSON...")
    loaded_ocr_data = []
    if Path(OCR_DIR).exists():
        for json_file in Path(OCR_DIR).glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    loaded_ocr_data.extend(json.load(f))
            except Exception: pass
    with open(CACHE_META_OCR_PATH, "wb") as f:
        pickle.dump(loaded_ocr_data, f)
    print(f" -> Đã lưu {len(loaded_ocr_data):,} mục OCR vào {CACHE_META_OCR_PATH}")

    # 4. Map-Keyframes & KeyframeMapper
    print("\n[4/5] Xây dựng bản đồ chuyển đổi ID Keyframe (map-keyframes)...")
    km = KeyframeMapper(auto_load=False)
    km.load(force_reload=True)
    print(f" -> Đã lập bản đồ thời gian cho {len(km.n_to_meta):,} videos vào {CACHE_MAPPER_PATH}")

    # 5. ASR Inverted Index
    print("\n[5/5] Tiền xử lý & Lập chỉ mục ngược ASR (Speech Transcript)...")
    asrp = ASRPreprocessor()
    asrp.process_all(save_cache=True)

    print("\n" + "=" * 60)
    print("🎉 HOÀN TẤT TIỀN XỬ LÝ TOÀN BỘ 5 MÔ HÌNH VÀ BẢN ĐỒ ID KEYFRAME!")
    print("=" * 60)

if __name__ == "__main__":
    build_all_caches()
