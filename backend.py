import os, glob, json, re, pickle
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from collections import deque

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

def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Đang khởi tạo AI Model trên: {device.upper()}")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return device, model, processor

def load_faiss_database():
    # 1. Ưu tiên nạp siêu tốc từ cache FAISS nhị phân đã lưu (< 0.2s)
    if os.path.exists(CACHE_FAISS_PATH) and os.path.exists(CACHE_META_CLIP_PATH):
        try:
            print(f"[*] ⚡ Nạp FAISS siêu tốc từ cache: {CACHE_FAISS_PATH}")
            index = faiss.read_index(CACHE_FAISS_PATH)
            with open(CACHE_META_CLIP_PATH, "rb") as f:
                data = pickle.load(f)
            print(f"[*] Đã nạp xong {index.ntotal} vector FAISS từ cache.")
            return data["all_features"], data["metadata_clip"], index, data["key_to_faiss_id"]
        except Exception as e:
            print(f"[!] Lỗi khi đọc cache FAISS ({e}), chuyển sang nạp thô từ .npy...")

    # 2. Fallback: nạp từ file .npy nếu chưa có cache
    print(f"[*] Đang nạp đặc trưng CLIP từ thư mục: {CLIP_FEATURES_DIR}")
    npy_files = sorted(glob.glob(os.path.join(CLIP_FEATURES_DIR, "*.npy")))
    print(f"[DEBUG] Tìm thấy {len(npy_files)} file .npy trong {CLIP_FEATURES_DIR}")

    all_features = []
    metadata_clip = []

    for file_path in npy_files:
        try:
            features = np.load(file_path, allow_pickle=True).astype("float32")
            for frame_id, feature in enumerate(features, start=1):
                all_features.append(feature)
                metadata_clip.append((os.path.basename(file_path), frame_id))
        except Exception as e:
            print(f"[DEBUG] Lỗi khi đọc {file_path}: {e}")

    index = None
    key_to_faiss_id = {}

    if len(all_features) > 0:
        all_features = np.asarray(all_features, dtype="float32")
        faiss.normalize_L2(all_features)
        dim = all_features.shape[1]

        index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        index.hnsw.efSearch = HNSW_EF_SEARCH
        index.add(all_features)
        print(f"[*] Đã nạp xong {index.ntotal} vector FAISS (HNSW, M={HNSW_M}).")

        for internal_id, (raw_name, fid) in enumerate(metadata_clip):
            vkey = _make_video_key(raw_name)
            key_to_faiss_id[(vkey, fid)] = internal_id
            
        # Tự động lưu cache cho các lần chạy sau
        try:
            faiss.write_index(index, CACHE_FAISS_PATH)
            with open(CACHE_META_CLIP_PATH, "wb") as f:
                pickle.dump({"all_features": all_features, "metadata_clip": metadata_clip, "key_to_faiss_id": key_to_faiss_id}, f)
            print(f"[*] Đã tự động tạo cache FAISS tại {CACHE_FAISS_PATH}")
        except Exception as e:
            print(f"[!] Không thể lưu cache FAISS: {e}")
    else:
        all_features = np.asarray([], dtype="float32")

    return all_features, metadata_clip, index, key_to_faiss_id

def load_od_metadata():
    # 1. Ưu tiên nạp siêu tốc từ cache gộp (< 0.2s)
    if os.path.exists(CACHE_META_OD_PATH):
        try:
            print(f"[*] ⚡ Nạp Object Detection siêu tốc từ cache: {CACHE_META_OD_PATH}")
            with open(CACHE_META_OD_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[!] Lỗi khi đọc cache OD ({e}), chuyển sang nạp từ JSON...")

    metadata_OD = []
    if not Path(OBJECTS_DIR).exists():
        return metadata_OD
    print(f"[*] Đang nạp dữ liệu OD từ: {OBJECTS_DIR}")
    for video_dir in Path(OBJECTS_DIR).iterdir():
        if not video_dir.is_dir():
            continue
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
            except Exception as e:
                print(f"[DEBUG] Lỗi khi đọc {json_file}: {e}")
                
    # Tự động lưu cache OD
    try:
        with open(CACHE_META_OD_PATH, "wb") as f:
            pickle.dump(metadata_OD, f)
        print(f"[*] Đã tự động tạo cache OD tại {CACHE_META_OD_PATH}")
    except Exception as e:
        print(f"[!] Không thể lưu cache OD: {e}")

    return metadata_OD

def load_ocr_metadata():
    # 1. Ưu tiên nạp siêu tốc từ cache gộp (< 0.1s)
    if os.path.exists(CACHE_META_OCR_PATH):
        try:
            print(f"[*] ⚡ Nạp OCR siêu tốc từ cache: {CACHE_META_OCR_PATH}")
            with open(CACHE_META_OCR_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[!] Lỗi khi đọc cache OCR ({e}), chuyển sang nạp từ JSON...")

    loaded_ocr_data = []
    if not Path(OCR_DIR).exists():
        return loaded_ocr_data
    print(f"[*] Đang nạp dữ liệu OCR từ: {OCR_DIR}")
    for json_file in Path(OCR_DIR).glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                loaded_ocr_data.extend(json.load(f))
        except Exception as e:
            print(f"[DEBUG] Lỗi khi đọc {json_file}: {e}")
            
    # Tự động lưu cache OCR
    try:
        with open(CACHE_META_OCR_PATH, "wb") as f:
            pickle.dump(loaded_ocr_data, f)
        print(f"[*] Đã tự động tạo cache OCR tại {CACHE_META_OCR_PATH}")
    except Exception as e:
        print(f"[!] Không thể lưu cache OCR: {e}")

    return loaded_ocr_data

DEFAULT_CLIP_CANDIDATE_POOL = 500

class AICSearchEngine:
    MODULE_FUNCS = {
        "clip": "search_clip",
        "od": "search_od",
        "ocr": "search_ocr",
    }

    def __init__(self, device, model, processor, all_features, metadata_clip,
                 index, key_to_faiss_id, metadata_OD, loaded_ocr_data):
        self.device = device
        self.model = model
        self.processor = processor
        self.all_features = all_features
        self.metadata_clip = metadata_clip
        self.index = index
        self.key_to_faiss_id = key_to_faiss_id
        self.metadata_OD = metadata_OD
        self.loaded_ocr_data = loaded_ocr_data

    def min_max_normalize(self, scores):
        scores = np.asarray(scores, dtype=np.float32)
        if scores.size == 0:
            return scores
        min_sc, max_sc = scores.min(), scores.max()
        if max_sc == min_sc:
            return np.zeros_like(scores)
        return (scores - min_sc) / (max_sc - min_sc)

    def _ocr_similarity(self, query, text):
        query_words = re.findall(r'\w+', str(query).lower())
        text_words = set(re.findall(r'\w+', str(text).lower()))
        if not query_words:
            return 0.0
        return sum(1 for word in query_words if word in text_words) / len(query_words)

    def _make_video_key(self, raw_name):
        name = str(raw_name)
        for ext in ['.npy', '.mp4', '.avi', '.mov', '.mkv']:
            if name.endswith(ext):
                return name[:-len(ext)]
        return name

    def _normalize_allowed(self, allowed_frames):
        if allowed_frames is None:
            return None
        return {(self._make_video_key(v), int(f)) for v, f in allowed_frames}

    def _encode_text(self, text_query):
        inputs = self.processor(text=text_query, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_outputs = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None)
            )
            pooled = text_outputs[1]
            text_features = self.model.text_projection(pooled)
            text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features.cpu().numpy().astype("float32")

    def results_to_allowed_frames(self, fusion_output):
        if isinstance(fusion_output, dict):
            items = fusion_output.get("results", [])
        else:
            items = fusion_output or []
        return [(item["video_id"], item["frame_id"]) for item in items]

    def search_clip(self, text_query, allowed_frames=None, ef_search=None, candidate_pool=None):
        results = []
        if self.index is None or not text_query:
            return results

        allowed_set = self._normalize_allowed(allowed_frames)
        query_feat = self._encode_text(text_query)

        selector = None
        if allowed_set is not None:
            faiss_ids = [self.key_to_faiss_id[k] for k in allowed_set if k in self.key_to_faiss_id]
            if not faiss_ids:
                return results
            selector = faiss.IDSelectorArray(np.array(faiss_ids, dtype=np.int64))
            k = min(len(faiss_ids), candidate_pool or DEFAULT_CLIP_CANDIDATE_POOL)
        else:
            k = min(self.index.ntotal, candidate_pool or DEFAULT_CLIP_CANDIDATE_POOL)

        params = faiss.SearchParametersHNSW(
            efSearch=ef_search or max(HNSW_EF_SEARCH, k),
            sel=selector
        )
        dist, ids = self.index.search(query_feat, k, params=params)

        raw_pairs = [(fid, sc) for fid, sc in zip(ids[0], dist[0]) if fid != -1]
        if not raw_pairs:
            return results

        fids, scores = zip(*raw_pairs)
        norm_c = self.min_max_normalize(np.asarray(scores))
        for fid, sc in zip(fids, norm_c):
            video_key = self._make_video_key(self.metadata_clip[fid][0])
            results.append({
                "video_id": video_key,
                "frame_id": int(self.metadata_clip[fid][1]),
                "score": float(sc)
            })

        return results

    def search_od(self, query_objects, allowed_frames=None):
        results = []
        if not query_objects:
            return results

        allowed_set = self._normalize_allowed(allowed_frames)
        objects = {obj.lower().strip() for obj in query_objects if obj.strip()}

        raw = []
        for frame in self.metadata_OD:
            video_key = self._make_video_key(frame["video"])
            if allowed_set is not None and (video_key, frame["frame_id"]) not in allowed_set:
                continue

            object_conf = {obj: 0.0 for obj in objects}
            for score, obj_class in zip(frame["scores"], frame["classes"]):
                oc = obj_class.lower()
                if oc in object_conf:
                    object_conf[oc] = max(object_conf[oc], float(score))

            raw.append({
                "video_id": video_key,
                "frame_id": frame["frame_id"],
                "raw_score": float(np.mean(list(object_conf.values())) if object_conf else 0.0)
            })

        if not raw:
            return results

        norm_od = self.min_max_normalize([r["raw_score"] for r in raw])
        for r, sc in zip(raw, norm_od):
            results.append({"video_id": r["video_id"], "frame_id": r["frame_id"], "score": float(sc)})
        return results

    def search_ocr(self, text_query, allowed_frames=None):
        results = []
        if not (text_query and self.loaded_ocr_data):
            return results

        allowed_set = self._normalize_allowed(allowed_frames)
        items = self.loaded_ocr_data
        if allowed_set is not None:
            items = [
                item for item in items
                if (self._make_video_key(item["video_id"]), item["frame_n"]) in allowed_set
            ]
        if not items:
            return results

        raw_ocr = [self._ocr_similarity(text_query, item["text"]) for item in items]
        norm_ocr = self.min_max_normalize(raw_ocr)
        for item, sc in zip(items, norm_ocr):
            results.append({
                "video_id": self._make_video_key(item["video_id"]),
                "frame_id": item["frame_n"],
                "score": float(sc)
            })
        return results

    def _fuse_modules(self, modules, allowed_frames=None):
        if not modules:
            raise ValueError("modules rỗng — không có module nào để fusion.")

        module_scores = {}
        weights = {}

        for name, cfg in modules.items():
            if name not in self.MODULE_FUNCS:
                raise ValueError(f"Module '{name}' không tồn tại. Chọn trong {list(self.MODULE_FUNCS)}")

            cfg = dict(cfg or {})
            weight = float(cfg.pop("weight", 1.0))
            cfg.setdefault("allowed_frames", allowed_frames)

            func = getattr(self, self.MODULE_FUNCS[name])
            raw_results = func(**cfg)

            module_scores[name] = {(r["video_id"], r["frame_id"]): r["score"] for r in raw_results}
            weights[name] = weight

        all_keys = set()
        for scores in module_scores.values():
            all_keys.update(scores.keys())

        combined = {}
        per_module = {}
        for key in all_keys:
            total = 0.0
            breakdown = {}
            for name, scores in module_scores.items():
                sc = scores.get(key, 0.0)
                breakdown[name] = sc
                total += weights[name] * sc
            combined[key] = total
            per_module[key] = breakdown

        return combined, per_module

    def late_fusion(self, modules, top_k=10, allowed_frames=None):
        if not modules:
            return {"status": "error", "message": "Chưa chọn module nào để fusion."}

        combined, per_module = self._fuse_modules(modules, allowed_frames=allowed_frames)

        fusion_results = []
        for key, final_score in combined.items():
            video_id, frame_id = key
            per_module_scores = {f"{name}_score": sc for name, sc in per_module[key].items()}
            fusion_results.append({
                "video_id": video_id,
                "frame_id": frame_id,
                "score": float(final_score),
                **per_module_scores
            })

        fusion_results.sort(key=lambda x: x["score"], reverse=True)
        final = fusion_results[:top_k]
        for rank, res in enumerate(final, 1):
            res["rank"] = rank
        return {"status": "success", "results": final}

    def multistage_query(self, stages, allowed_frames=None, return_intermediate=True):
        if not stages:
            return {"status": "error", "message": "Chưa có giai đoạn (stage) nào để chạy."}

        current_allowed = allowed_frames
        stage_outputs = []

        for i, stage in enumerate(stages, start=1):
            modules = stage.get("modules")
            if not modules:
                raise ValueError(f"Stage {i} thiếu 'modules' — phải chỉ định ít nhất 1 module để chạy.")

            top_k = stage.get("top_k", 100)
            stage_allowed = stage.get("allowed_frames", current_allowed)

            stage_result = self.late_fusion(modules, top_k=top_k, allowed_frames=stage_allowed)
            stage_outputs.append(stage_result)

            current_allowed = self.results_to_allowed_frames(stage_result)

            if not current_allowed:
                break
                
        if not stage_outputs:
            return {"status": "error", "message": "Không có kết quả."}

        final_result = stage_outputs[-1]
        if return_intermediate:
            return {"status": "success", "results": final_result["results"], "stages": stage_outputs}
        return {"status": "success", "results": final_result["results"]}

    def search_kis(self, stages, allowed_frames=None, return_intermediate=False):
        return self.multistage_query(
            stages=stages,
            allowed_frames=allowed_frames,
            return_intermediate=return_intermediate
        )

    def _compute_event_scores(self, event_modules, allowed_frames=None):
        combined, _ = self._fuse_modules(event_modules, allowed_frames=allowed_frames)
        return combined

    def search_trake(self, events, allowed_frames=None, top_k_videos=5, min_gap=1, max_gap=None):
        if not events:
            return {"status": "error", "message": "Cần ít nhất 1 event."}

        N = len(events)

        event_score_maps = [
            self._compute_event_scores(ev, allowed_frames=allowed_frames) for ev in events
        ]

        videos = {}
        for score_map in event_score_maps:
            for (vid, fid) in score_map.keys():
                videos.setdefault(vid, set()).add(fid)

        NEG_INF = float("-inf")
        video_results = []

        for vid, frame_set in videos.items():
            frames = sorted(frame_set)
            M = len(frames)
            if M < N:
                continue

            score_mat = [
                [score_map.get((vid, f), 0.0) for f in frames]
                for score_map in event_score_maps
            ]

            dp_prev = score_mat[0][:]
            all_dp = [dp_prev]
            all_choice = [[-1] * M]

            for i in range(1, N):
                dp_cur = [NEG_INF] * M
                choice_cur = [-1] * M

                dq = deque()
                right = -1

                for j in range(M):
                    high = frames[j] - min_gap
                    low = frames[j] - max_gap if max_gap is not None else NEG_INF

                    while right + 1 < M and frames[right + 1] <= high:
                        right += 1
                        while dq and dp_prev[dq[-1]] <= dp_prev[right]:
                            dq.pop()
                        dq.append(right)

                    while dq and frames[dq[0]] < low:
                        dq.popleft()

                    if dq and dp_prev[dq[0]] > NEG_INF:
                        best_p = dq[0]
                        dp_cur[j] = score_mat[i][j] + dp_prev[best_p]
                        choice_cur[j] = best_p

                dp_prev = dp_cur
                all_dp.append(dp_cur)
                all_choice.append(choice_cur)

            best_final = max(all_dp[-1])
            if best_final == NEG_INF:
                continue

            best_j = all_dp[-1].index(best_final)
            path_frames = [None] * N
            path_frames[N - 1] = frames[best_j]
            cur_j = best_j
            for i in range(N - 1, 0, -1):
                prev_j = all_choice[i][cur_j]
                path_frames[i - 1] = frames[prev_j]
                cur_j = prev_j

            video_results.append({
                "video_id": vid,
                "score": float(best_final),
                "frames": path_frames,
            })

        video_results.sort(key=lambda x: x["score"], reverse=True)
        final = video_results[:top_k_videos]
        for rank, r in enumerate(final, 1):
            r["rank"] = rank

        return {"status": "success", "results": final}
