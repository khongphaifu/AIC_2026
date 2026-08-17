import os, glob, json, pickle
import numpy as np
import faiss
from pathlib import Path
from config import DATA_DIR, CLIP_FEATURES_DIR, OBJECTS_DIR, OCR_DIR

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 80
HNSW_EF_SEARCH = 64

CACHE_FAISS_PATH = os.path.join(DATA_DIR, "cache_faiss.index")
CACHE_META_CLIP_PATH = os.path.join(DATA_DIR, "cache_meta_clip.pkl")
CACHE_META_OD_PATH = os.path.join(DATA_DIR, "cache_meta_od.pkl")
CACHE_META_OCR_PATH = os.path.join(DATA_DIR, "cache_meta_ocr.pkl")

def _make_video_key(raw_name):
    name = str(raw_name)
    for ext in ['.npy', '.mp4', '.avi', '.mov', '.mkv']:
        if name.endswith(ext):
            return name[:-len(ext)]
    return name

def build_caches():
    print("[1/3] Đang nén và xây dựng FAISS Index...")
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
            vkey = _make_video_key(raw_name)
            key_to_faiss_id[(vkey, fid)] = internal_id
            
        with open(CACHE_META_CLIP_PATH, "wb") as f:
            pickle.dump({"all_features": all_features, "metadata_clip": metadata_clip, "key_to_faiss_id": key_to_faiss_id}, f)
        print(f" -> Đã lưu {index.ntotal} vectors vào {CACHE_FAISS_PATH}")

    print("[2/3] Đang gộp toàn bộ Object Detection JSON...")
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
    print(f" -> Đã gộp {len(metadata_OD)} frames OD vào {CACHE_META_OD_PATH}")

    print("[3/3] Đang gộp dữ liệu OCR...")
    loaded_ocr_data = []
    if Path(OCR_DIR).exists():
        for json_file in Path(OCR_DIR).glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    loaded_ocr_data.extend(json.load(f))
            except Exception: pass
    with open(CACHE_META_OCR_PATH, "wb") as f:
        pickle.dump(loaded_ocr_data, f)
    print(f" -> Đã lưu {len(loaded_ocr_data)} mục OCR vào {CACHE_META_OCR_PATH}")

    print("\n🎉 HOÀN TẤT! Dữ liệu đã được nén nhị phân siêu tốc.")

if __name__ == "__main__":
    build_caches()
