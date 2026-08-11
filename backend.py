import os, glob, json, re
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from config import CLIP_FEATURES_DIR, OBJECTS_DIR, OCR_DIR

class AICSearchEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Đang khởi tạo AI Model trên: {self.device.upper()}")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

        self.all_features, self.metadata_clip, self.index = [], [], None
        self._load_faiss_database()
        
        self.metadata_OD = []
        self._load_od_metadata()
        
        self.loaded_ocr_data = []
        self._load_ocr_metadata()

    def _load_faiss_database(self):
        print(f"[*] Đang nạp đặc trưng CLIP từ: {CLIP_FEATURES_DIR}")
        npy_files = sorted(glob.glob(os.path.join(CLIP_FEATURES_DIR, "*.npy")))
        for file_path in npy_files:
            try:
                features = np.load(file_path, allow_pickle=True).astype("float32")
                for frame_id, feature in enumerate(features, start=1):
                    self.all_features.append(feature)
                    self.metadata_clip.append((os.path.basename(file_path), frame_id))
            except: pass
        if len(self.all_features) > 0:
            self.all_features = np.asarray(self.all_features, dtype="float32")
            faiss.normalize_L2(self.all_features)
            self.index = faiss.IndexFlatIP(self.all_features.shape[1])
            self.index.add(self.all_features)
            print(f"[*] Đã nạp xong {self.index.ntotal} vector FAISS.")

    def _load_od_metadata(self):
        if not Path(OBJECTS_DIR).exists(): return
        print(f"[*] Đang nạp dữ liệu OD từ: {OBJECTS_DIR}")
        for video_dir in Path(OBJECTS_DIR).iterdir():
            if not video_dir.is_dir(): continue
            for json_file in video_dir.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.metadata_OD.append({
                        "video": video_dir.name,
                        "frame_id": int(json_file.stem.split("_")[-1]),
                        "scores": data.get("detection_scores", []),
                        "classes": data.get("detection_class_entities", [])
                    })
                except: pass

    def _load_ocr_metadata(self):
        if not Path(OCR_DIR).exists(): return
        print(f"[*] Đang nạp dữ liệu OCR từ: {OCR_DIR}")
        for json_file in Path(OCR_DIR).glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    self.loaded_ocr_data.extend(json.load(f))
            except: pass

    def min_max_normalize(self, scores):
        scores = np.asarray(scores, dtype=np.float32)
        if scores.size == 0: return scores
        min_sc, max_sc = scores.min(), scores.max()
        if max_sc == min_sc: return np.zeros_like(scores)
        return (scores - min_sc) / (max_sc - min_sc)

    def _ocr_similarity(self, query, text):
        query_words = re.findall(r'\w+', str(query).lower())
        text_words = set(re.findall(r'\w+', str(text).lower()))
        if not query_words: return 0.0
        return sum(1 for word in query_words if word in text_words) / len(query_words)

    def _calculate_od_scores(self, query_objects):
        query_objects = {obj.lower().strip() for obj in query_objects if obj.strip()}
        results = []
        for frame in self.metadata_OD:
            object_conf = {obj: 0.0 for obj in query_objects}
            for score, obj_class in zip(frame["scores"], frame["classes"]):
                oc = obj_class.lower()
                if oc in query_objects:
                    object_conf[oc] = max(object_conf[oc], float(score))
            results.append({
                "video": frame["video"], "frame_id": frame["frame_id"],
                "od_score": float(np.mean(list(object_conf.values())) if object_conf else 0.0)
            })
        return results

    def _make_video_key(self, raw_name):
        name = str(raw_name)
        for ext in ['.npy', '.mp4', '.avi', '.mov', '.mkv']:
            if name.endswith(ext): return name[:-len(ext)]
        return name

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

    def search_kis(self, text_query, query_objects, top_k=10, w_clip=0.5, w_od=0.3, w_ocr=0.2):
        clip_results = []
        if w_clip > 0 and self.index and text_query:
            query_feat = self._encode_text(text_query)
            dist, ids = self.index.search(query_feat, self.index.ntotal)
            norm_c = self.min_max_normalize(dist[0])
            for fid, sc in zip(ids[0], norm_c):
                video_key = self._make_video_key(self.metadata_clip[fid][0])
                clip_results.append({
                    "video": video_key,
                    "frame_id": int(self.metadata_clip[fid][1]),
                    "clip_score": float(sc)
                })

        od_dict = {}
        if w_od > 0 and query_objects:
            od_res = self._calculate_od_scores(query_objects)
            if od_res:
                norm_od = self.min_max_normalize([r["od_score"] for r in od_res])
                for r, sc in zip(od_res, norm_od):
                    od_dict[(self._make_video_key(r["video"]), r["frame_id"])] = float(sc)

        ocr_dict = {}
        if w_ocr > 0 and text_query and self.loaded_ocr_data:
            raw_ocr = [self._ocr_similarity(text_query, item["text"]) for item in self.loaded_ocr_data]
            norm_ocr = self.min_max_normalize(raw_ocr)
            for item, sc in zip(self.loaded_ocr_data, norm_ocr):
                ocr_dict[(self._make_video_key(item["video_id"]), item["frame_n"])] = float(sc)

        fusion_results = []
        for item in clip_results:
            key = (item["video"], item["frame_id"])
            cs = item["clip_score"]
            os_ = ocr_dict.get(key, 0.0)
            ds = od_dict.get(key, 0.0)
            final_score = (w_clip * cs) + (w_ocr * os_) + (w_od * ds)

            fusion_results.append({
                "video_id": item["video"], "frame_id": item["frame_id"],
                "score": float(final_score),
                "clip_score": cs, "ocr_score": os_, "od_score": ds
            })

        fusion_results.sort(key=lambda x: x["score"], reverse=True)
        final = fusion_results[:top_k]
        for rank, res in enumerate(final, 1): res["rank"] = rank
        return {"status": "success", "results": final}
