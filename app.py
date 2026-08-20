import os
import streamlit as st
import importlib
import math
import backend
from config import KEYFRAME_DIR, VIDEOS_DIR
from preprocess import mapper, ASRPreprocessor

st.set_page_config(page_title="AIC 2026 Studio - vitrivr Style", page_icon="⚡", layout="wide")

# CSS dark-mode vitrivr cao cấp & Thẻ ảnh click trực tiếp 100% sạch sẽ
st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #e6edf3; }
    
    .panel-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #58a6ff;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Thẻ Card vitrivr keyframe chuẩn gốc */
    .vitrivr-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 4px;
        text-align: center;
        margin-bottom: 8px;
        transition: transform 0.12s ease, border-color 0.12s ease, box-shadow 0.12s ease;
        cursor: pointer;
        user-select: none;
    }
    .vitrivr-card:hover {
        border-color: #58a6ff;
        transform: translateY(-3px);
        box-shadow: 0 4px 14px rgba(88, 166, 255, 0.35);
    }
    .card-img {
        width: 100%;
        aspect-ratio: 16/9;
        object-fit: cover;
        border-radius: 4px;
        display: block;
        background-color: #0d1117;
        pointer-events: none;
    }
    .score-badge {
        background: #238636;
        color: #ffffff;
        font-weight: 700;
        font-size: 0.72rem;
        padding: 1px 6px;
        border-radius: 6px;
        display: inline-block;
        margin-bottom: 2px;
        pointer-events: none;
    }
    .meta-tag-primary {
        font-size: 0.78rem;
        font-weight: 600;
        color: #58a6ff;
        display: block;
        margin-top: 3px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        pointer-events: none;
    }
    .meta-tag-secondary {
        font-size: 0.72rem;
        color: #8b949e;
        display: block;
        margin-bottom: 2px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        pointer-events: none;
    }
    
    /* Ẩn hoàn toàn container của nút trigger ngầm dưới thẻ card */
    div[data-testid="stElementContainer"]:has(.vitrivr-card) + div[data-testid="stElementContainer"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper fragment an toàn
def safe_fragment(func):
    if hasattr(st, "fragment"):
        return st.fragment(func)
    return func

@st.cache_resource(show_spinner="🧠 Đang nạp AI Models & Dữ liệu (CLIP, OD, OCR, ASR)...")
def load_resources():
    importlib.reload(backend)
    device, model, processor = backend.load_clip_model()
    all_features, metadata_clip, index, key_to_faiss_id = backend.load_faiss_database()
    metadata_OD = backend.load_od_metadata()
    loaded_ocr_data = backend.load_ocr_metadata()
    map_keyframes = backend.load_map_keyframes()
    asr_data = backend.load_asr_metadata()
    return device, model, processor, all_features, metadata_clip, index, key_to_faiss_id, metadata_OD, loaded_ocr_data, map_keyframes, asr_data

device, model, processor, all_features, metadata_clip, index, key_to_faiss_id, metadata_OD, loaded_ocr_data, map_keyframes, asr_data = load_resources()
engine = backend.AICSearchEngine(device, model, processor, all_features, metadata_clip, index, key_to_faiss_id, metadata_OD, loaded_ocr_data, map_keyframes, asr_data, kf_mapper=mapper)

# URL tĩnh trực tiếp phục vụ HTTP siêu tốc và browser caching
def get_image_url(video_id, frame_id):
    video_name = video_id.replace('.mp4', '')
    fmt_frame = f"{int(frame_id):03d}.jpg"
    return f"/app/static/{video_name}/{fmt_frame}"

# Cache video trên ổ đĩa
_VIDEO_PATH_CACHE = {}
def get_video_path(video_id):
    if video_id in _VIDEO_PATH_CACHE:
        return _VIDEO_PATH_CACHE[video_id]
    if VIDEOS_DIR is None:
        _VIDEO_PATH_CACHE[video_id] = None
        return None
    
    vname = video_id if video_id.endswith(".mp4") else f"{video_id}.mp4"
    path = os.path.join(VIDEOS_DIR, vname)
    if os.path.exists(path):
        _VIDEO_PATH_CACHE[video_id] = path
        return path
    _VIDEO_PATH_CACHE[video_id] = None
    return None

# Helper danh sách videos
@st.cache_data
def get_unique_videos_list(_metadata_clip):
    vids = set()
    for raw_name, _ in _metadata_clip:
        vids.add(backend._make_video_key(raw_name))
    return sorted(list(vids))

unique_videos = get_unique_videos_list(metadata_clip)

# Helper lấy keyframes
@st.cache_data
def get_dataset_keyframes(_metadata_clip, video_filter):
    if video_filter == "Tất cả videos" or not video_filter:
        return [(backend._make_video_key(item[0]), item[1]) for item in _metadata_clip]
    else:
        return [
            (backend._make_video_key(item[0]), item[1])
            for item in _metadata_clip
            if backend._make_video_key(item[0]) == video_filter
        ]

# =====================================================================
# THUẬT TOÁN TỰ ĐỘNG CÂN BẰNG TRỌNG SỐ CHO TỔNG BẰNG 100%
# =====================================================================
def rebalance_weights_on_change(target, changed_key, new_val):
    active = [m for m in ['clip', 'od', 'ocr', 'asr'] if target.get(f'use_{m}', False)]
    if not active or len(active) == 1:
        if active:
            target[f'weight_{active[0]}'] = 100
        return
    if changed_key not in active:
        return

    others = [m for m in active if m != changed_key]
    new_val = max(0, min(100, int(new_val)))
    target[f'weight_{changed_key}'] = new_val
    remaining = 100 - new_val

    current_others_sum = sum(target.get(f'weight_{m}', 0) for m in others)
    new_others = {}
    if current_others_sum > 0:
        allocated = 0
        for m in others[:-1]:
            w = int(round(target.get(f'weight_{m}', 0) * remaining / current_others_sum))
            new_others[m] = w
            allocated += w
        new_others[others[-1]] = remaining - allocated
    else:
        each = remaining // len(others)
        allocated = 0
        for m in others[:-1]:
            new_others[m] = each
            allocated += each
        new_others[others[-1]] = remaining - allocated

    for m, w in new_others.items():
        target[f'weight_{m}'] = max(0, min(100, w))

def rebalance_weights_on_toggle(target, toggled_key, is_on):
    target[f'use_{toggled_key}'] = is_on
    active = [m for m in ['clip', 'od', 'ocr', 'asr'] if target.get(f'use_{m}', False)]
    if not active:
        for m in ['clip', 'od', 'ocr', 'asr']:
            target[f'weight_{m}'] = 0
        return
    if len(active) == 1:
        for m in ['clip', 'od', 'ocr', 'asr']:
            target[f'weight_{m}'] = 100 if m == active[0] else 0
        return

    if not is_on:
        target[f'weight_{toggled_key}'] = 0
        rem_sum = sum(target.get(f'weight_{m}', 0) for m in active)
        if rem_sum > 0:
            allocated = 0
            for m in active[:-1]:
                w = int(round(target.get(f'weight_{m}', 0) * 100 / rem_sum))
                target[f'weight_{m}'] = w
                allocated += w
            target[f'weight_{active[-1]}'] = 100 - allocated
        else:
            each = 100 // len(active)
            allocated = 0
            for m in active[:-1]:
                target[f'weight_{m}'] = each
                allocated += each
            target[f'weight_{active[-1]}'] = 100 - allocated
    else:
        target_val = 100 // len(active)
        others = [m for m in active if m != toggled_key]
        target[f'weight_{toggled_key}'] = target_val
        rem_budget = 100 - target_val
        old_sum = sum(target.get(f'weight_{m}', 0) for m in others)
        if old_sum > 0:
            allocated = 0
            for m in others[:-1]:
                w = int(round(target.get(f'weight_{m}', 0) * rem_budget / old_sum))
                target[f'weight_{m}'] = w
                allocated += w
            target[f'weight_{others[-1]}'] = rem_budget - allocated
        else:
            each = rem_budget // len(others)
            allocated = 0
            for m in others[:-1]:
                target[f'weight_{m}'] = each
                allocated += each
            target[f'weight_{others[-1]}'] = rem_budget - allocated


# ==================== KHỞI TẠO STATE BỀN VỮNG TRONG RAM ====================
if "query_type" not in st.session_state:
    st.session_state.query_type = "🎯 Textual KIS (Nhiều giai đoạn)"

if "kis_stages" not in st.session_state:
    st.session_state.kis_stages = [
        {
            "top_k": 100,
            "use_clip": True, "weight_clip": 100, "query_clip": "",
            "use_od": False, "weight_od": 0, "query_od": "",
            "use_ocr": False, "weight_ocr": 0, "query_ocr": "",
            "use_asr": False, "weight_asr": 0, "query_asr": "", "padding_asr": 15
        }
    ]

if "active_kis_stage_idx" not in st.session_state:
    st.session_state.active_kis_stage_idx = 0

if "trake_events" not in st.session_state:
    st.session_state.trake_events = [
        {
            "use_clip": True, "weight_clip": 100, "query_clip": "",
            "use_od": False, "weight_od": 0, "query_od": "",
            "use_ocr": False, "weight_ocr": 0, "query_ocr": "",
            "use_asr": False, "weight_asr": 0, "query_asr": "", "padding_asr": 15
        },
        {
            "use_clip": True, "weight_clip": 100, "query_clip": "",
            "use_od": False, "weight_od": 0, "query_od": "",
            "use_ocr": False, "weight_ocr": 0, "query_ocr": "",
            "use_asr": False, "weight_asr": 0, "query_asr": "", "padding_asr": 15
        }
    ]

if "active_trake_event_idx" not in st.session_state:
    st.session_state.active_trake_event_idx = 0

if "trake_top_k_videos" not in st.session_state:
    st.session_state.trake_top_k_videos = 5

if "trake_min_gap" not in st.session_state:
    st.session_state.trake_min_gap = 1

if "trake_max_gap" not in st.session_state:
    st.session_state.trake_max_gap = 50

if "active_video" not in st.session_state:
    st.session_state.active_video = None

if "kis_results" not in st.session_state:
    st.session_state.kis_results = None

if "trake_results" not in st.session_state:
    st.session_state.trake_results = None

if "gallery_page" not in st.session_state:
    st.session_state.gallery_page = 1

if "gallery_filter_video" not in st.session_state:
    st.session_state.gallery_filter_video = "Tất cả videos"

if "col2_view_mode" not in st.session_state:
    st.session_state.col2_view_mode = "🖼️ Thư viện Keyframes"

# Callbacks điều khiển Giai đoạn / Video
def set_active_kis_stage(idx):
    st.session_state.active_kis_stage_idx = idx

def add_kis_stage():
    st.session_state.kis_stages.append({
        "top_k": 10,
        "use_clip": True, "weight_clip": 100, "query_clip": "",
        "use_od": False, "weight_od": 0, "query_od": "",
        "use_ocr": False, "weight_ocr": 0, "query_ocr": "",
        "use_asr": False, "weight_asr": 0, "query_asr": "", "padding_asr": 15
    })
    st.session_state.active_kis_stage_idx = len(st.session_state.kis_stages) - 1

def remove_kis_stage():
    if len(st.session_state.kis_stages) > 1:
        st.session_state.kis_stages.pop()
        if st.session_state.active_kis_stage_idx >= len(st.session_state.kis_stages):
            st.session_state.active_kis_stage_idx = len(st.session_state.kis_stages) - 1

def set_active_trake_event(idx):
    st.session_state.active_trake_event_idx = idx

def add_trake_event():
    st.session_state.trake_events.append({
        "use_clip": True, "weight_clip": 100, "query_clip": "",
        "use_od": False, "weight_od": 0, "query_od": "",
        "use_ocr": False, "weight_ocr": 0, "query_ocr": "",
        "use_asr": False, "weight_asr": 0, "query_asr": "", "padding_asr": 15
    })
    st.session_state.active_trake_event_idx = len(st.session_state.trake_events) - 1

def remove_trake_event():
    if len(st.session_state.trake_events) > 1:
        st.session_state.trake_events.pop()
        if st.session_state.active_trake_event_idx >= len(st.session_state.trake_events):
            st.session_state.active_trake_event_idx = len(st.session_state.trake_events) - 1

def play_video_callback(video_id, frame_id, start_sec):
    st.session_state.active_video = {"video_id": video_id, "frame_id": frame_id, "start_sec": start_sec}

def close_video_callback():
    st.session_state.active_video = None


# Callbacks tự động cân bằng trọng số KIS
def on_kis_toggle(s_idx, model):
    stg = st.session_state.kis_stages[s_idx]
    is_checked = st.session_state.get(f"c3_chk_{model}_{s_idx}", False)
    rebalance_weights_on_toggle(stg, model, is_checked)
    for m in ['clip', 'od', 'ocr', 'asr']:
        st.session_state[f"c3_w_{m}_{s_idx}"] = stg[f"weight_{m}"]

def on_kis_weight_change(s_idx, model):
    stg = st.session_state.kis_stages[s_idx]
    new_val = st.session_state.get(f"c3_w_{model}_{s_idx}", 0)
    rebalance_weights_on_change(stg, model, new_val)
    for m in ['clip', 'od', 'ocr', 'asr']:
        st.session_state[f"c3_w_{m}_{s_idx}"] = stg[f"weight_{m}"]

# Callbacks tự động cân bằng trọng số TRAKE
def on_trake_toggle(e_idx, model):
    ev = st.session_state.trake_events[e_idx]
    is_checked = st.session_state.get(f"c3_tr_chk_{model}_{e_idx}", False)
    rebalance_weights_on_toggle(ev, model, is_checked)
    for m in ['clip', 'od', 'ocr', 'asr']:
        st.session_state[f"c3_tr_w_{m}_{e_idx}"] = ev[f"weight_{m}"]

def on_trake_weight_change(e_idx, model):
    ev = st.session_state.trake_events[e_idx]
    new_val = st.session_state.get(f"c3_tr_w_{model}_{e_idx}", 0)
    rebalance_weights_on_change(ev, model, new_val)
    for m in ['clip', 'od', 'ocr', 'asr']:
        st.session_state[f"c3_tr_w_{m}_{e_idx}"] = ev[f"weight_{m}"]


# =========================================================
# BỐ CỤC 3 CỘT: CỘT 1 (ĐIỀU KHIỂN) - CỘT 2 (GALLERY) - CỘT 3 (CHI TIẾT)
# =========================================================
col_left_main, col_mid_main, col_right_main = st.columns([3.0, 5.2, 3.8], gap="medium")

# ---------------------------------------------------------
# CÔ LẬP CỘT 1 & CỘT 3 TRONG FRAGMENT (CHUYỂN ĐỔI 0MS INSTANT)
# ---------------------------------------------------------
@safe_fragment
def render_controls():
    # -----------------------------------------------------
    # CỘT 1: CHỌN TRUY VẤN & DANH SÁCH GIAI ĐOẠN / EVENT
    # -----------------------------------------------------
    with col_left_main:
        st.markdown("<div class='panel-title'>⚙️ CỘT 1: CHỌN TRUY VẤN</div>", unsafe_allow_html=True)
        
        q_types = ["🎯 Textual KIS (Nhiều giai đoạn)", "⏱️ TRAKE (Chuỗi sự kiện)"]
        selected_q = st.radio(
            "Loại truy vấn:",
            q_types,
            index=0 if "KIS" in st.session_state.query_type else 1,
            label_visibility="collapsed"
        )
        st.session_state.query_type = selected_q
        st.divider()

        if "KIS" in selected_q:
            st.markdown(f"**Danh sách Giai đoạn ({len(st.session_state.kis_stages)}):**")
            
            for s_idx in range(len(st.session_state.kis_stages)):
                is_active = (s_idx == st.session_state.active_kis_stage_idx)
                label = f"{'👉 ' if is_active else ''}Giai đoạn {s_idx + 1}"
                st.button(
                    label,
                    key=f"btn_kis_stage_select_{s_idx}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True,
                    on_click=set_active_kis_stage,
                    args=(s_idx,)
                )

            col_add, col_rem = st.columns(2)
            with col_add:
                st.button("➕ Thêm", key="add_kis_stage_btn", use_container_width=True, on_click=add_kis_stage, disabled=(len(st.session_state.kis_stages) >= 5))
            with col_rem:
                st.button("➖ Bớt", key="rem_kis_stage_btn", use_container_width=True, on_click=remove_kis_stage, disabled=(len(st.session_state.kis_stages) <= 1))

            st.caption("💡 Click vào từng giai đoạn ở trên để chỉnh chi tiết Model ở **Cột 3**.")
            st.divider()
            
            if st.button("🚀 TÌM KIẾM KIS", type="primary", use_container_width=True, key="btn_exec_kis"):
                stages_payload = []
                valid = True
                for idx, stg in enumerate(st.session_state.kis_stages):
                    w_c = stg.get("weight_clip", 0) if stg.get("use_clip", False) else 0
                    q_c = stg.get("query_clip", "").strip()
                    
                    w_d = stg.get("weight_od", 0) if stg.get("use_od", False) else 0
                    q_d = stg.get("query_od", "").strip()
                    
                    w_r = stg.get("weight_ocr", 0) if stg.get("use_ocr", False) else 0
                    q_r = stg.get("query_ocr", "").strip()

                    w_a = stg.get("weight_asr", 0) if stg.get("use_asr", False) else 0
                    q_a = stg.get("query_asr", "").strip()
                    pad_a = float(stg.get("padding_asr", 15))

                    w_tot = w_c + w_d + w_r + w_a
                    if w_tot == 0:
                        st.error(f"Giai đoạn {idx+1}: Hãy chọn ít nhất 1 model có trọng số > 0%!")
                        valid = False
                        break

                    modules = {}
                    if stg.get("use_clip") and q_c and w_c > 0:
                        modules["clip"] = {"weight": w_c / w_tot, "text_query": q_c}
                    if stg.get("use_od") and q_d and w_d > 0:
                        objs = [x.strip() for x in q_d.split(",") if x.strip()]
                        modules["od"] = {"weight": w_d / w_tot, "query_objects": objs}
                    if stg.get("use_ocr") and q_r and w_r > 0:
                        modules["ocr"] = {"weight": w_r / w_tot, "text_query": q_r}
                    if stg.get("use_asr") and q_a and w_a > 0:
                        modules["asr"] = {"weight": w_a / w_tot, "text_query": q_a, "padding_sec": pad_a}

                    if not modules:
                        st.error(f"Giai đoạn {idx+1}: Hãy nhập nội dung tìm kiếm cho các model đang bật!")
                        valid = False
                        break

                    stages_payload.append({"modules": modules, "top_k": int(stg.get("top_k", 100))})

                if valid:
                    with st.spinner("🚀 Đang chạy tìm kiếm KIS..."):
                        res = engine.search_kis(stages_payload, return_intermediate=False)
                        st.session_state.kis_results = res
                        st.session_state.col2_view_mode = "🎯 Kết quả tìm kiếm"
                        if hasattr(st, "rerun"):
                            st.rerun(scope="app") if hasattr(st.rerun, "__code__") and "scope" in st.rerun.__code__.co_varnames else st.rerun()

        else:
            st.markdown("**Cấu hình chung TRAKE:**")
            top_k_videos = st.number_input("Top K Videos:", min_value=1, value=st.session_state.trake_top_k_videos, key="trake_top_k_input")
            st.session_state.trake_top_k_videos = top_k_videos

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                min_gap = st.number_input("Min gap:", min_value=0, value=st.session_state.trake_min_gap, key="trake_min_gap_input")
                st.session_state.trake_min_gap = min_gap
            with col_g2:
                max_gap_val = st.number_input("Max gap (0 = ∞):", min_value=0, value=st.session_state.trake_max_gap, key="trake_max_gap_input")
                st.session_state.trake_max_gap = max_gap_val

            st.markdown(f"**Danh sách Event ({len(st.session_state.trake_events)}):**")
            for e_idx in range(len(st.session_state.trake_events)):
                is_active = (e_idx == st.session_state.active_trake_event_idx)
                label = f"{'👉 ' if is_active else ''}Event {e_idx + 1}"
                st.button(
                    label,
                    key=f"btn_trake_ev_select_{e_idx}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True,
                    on_click=set_active_trake_event,
                    args=(e_idx,)
                )

            col_add, col_rem = st.columns(2)
            with col_add:
                st.button("➕ Thêm Event", key="add_trake_ev_btn", use_container_width=True, on_click=add_trake_event, disabled=(len(st.session_state.trake_events) >= 10))
            with col_rem:
                st.button("➖ Bớt", key="rem_trake_ev_btn", use_container_width=True, on_click=remove_trake_event, disabled=(len(st.session_state.trake_events) <= 1))

            st.caption("💡 Click vào từng Event ở trên để chỉnh chi tiết Model ở **Cột 3**.")
            st.divider()

            if st.button("🚀 CHẠY TRAKE", type="primary", use_container_width=True, key="btn_exec_trake"):
                events_payload = []
                valid = True
                for idx, ev in enumerate(st.session_state.trake_events):
                    w_c = ev.get("weight_clip", 0) if ev.get("use_clip", False) else 0
                    q_c = ev.get("query_clip", "").strip()
                    
                    w_d = ev.get("weight_od", 0) if ev.get("use_od", False) else 0
                    q_d = ev.get("query_od", "").strip()
                    
                    w_r = ev.get("weight_ocr", 0) if ev.get("use_ocr", False) else 0
                    q_r = ev.get("query_ocr", "").strip()

                    w_a = ev.get("weight_asr", 0) if ev.get("use_asr", False) else 0
                    q_a = ev.get("query_asr", "").strip()
                    pad_a = float(ev.get("padding_asr", 15))

                    w_tot = w_c + w_d + w_r + w_a
                    if w_tot == 0:
                        st.error(f"Event {idx+1}: Hãy chọn ít nhất 1 model có trọng số > 0%!")
                        valid = False
                        break

                    modules = {}
                    if ev.get("use_clip") and q_c and w_c > 0:
                        modules["clip"] = {"weight": w_c / w_tot, "text_query": q_c}
                    if ev.get("use_od") and q_d and w_d > 0:
                        objs = [x.strip() for x in q_d.split(",") if x.strip()]
                        modules["od"] = {"weight": w_d / w_tot, "query_objects": objs}
                    if ev.get("use_ocr") and q_r and w_r > 0:
                        modules["ocr"] = {"weight": w_r / w_tot, "text_query": q_r}
                    if ev.get("use_asr") and q_a and w_a > 0:
                        modules["asr"] = {"weight": w_a / w_tot, "text_query": q_a, "padding_sec": pad_a}

                    if not modules:
                        st.error(f"Event {idx+1}: Hãy nhập nội dung tìm kiếm cho các model đang bật!")
                        valid = False
                        break

                    events_payload.append(modules)

                if valid:
                    with st.spinner("🚀 Đang chạy thuật toán TRAKE..."):
                        max_gap_param = st.session_state.trake_max_gap if st.session_state.trake_max_gap > 0 else None
                        res = engine.search_trake(
                            events_payload,
                            top_k_videos=int(st.session_state.trake_top_k_videos),
                            min_gap=int(st.session_state.trake_min_gap),
                            max_gap=max_gap_param
                        )
                        st.session_state.trake_results = res
                        st.session_state.col2_view_mode = "🎯 Kết quả tìm kiếm"
                        if hasattr(st, "rerun"):
                            st.rerun(scope="app") if hasattr(st.rerun, "__code__") and "scope" in st.rerun.__code__.co_varnames else st.rerun()

        st.divider()
        c_act1, c_act2 = st.columns(2)
        with c_act1:
            if st.button("♻️ Nạp lại Cache", use_container_width=True):
                st.cache_resource.clear()
                st.cache_data.clear()
                _VIDEO_PATH_CACHE.clear()
                st.rerun()
        with c_act2:
            if st.button("⚡ Index ASR", use_container_width=True):
                with st.spinner("Đang lập chỉ mục ASR..."):
                    pre = ASRPreprocessor()
                    pre.process_all()
                    st.success("✅ Đã tạo chỉ mục ASR Index!")
                    st.cache_resource.clear()
                    st.rerun()

    # -----------------------------------------------------
    # CỘT 3: CHỈNH CHI TIẾT GIAI ĐOẠN / EVENT ĐANG CHỌN
    # -----------------------------------------------------
    with col_right_main:
        if "KIS" in st.session_state.query_type:
            curr_idx = st.session_state.active_kis_stage_idx
            if curr_idx >= len(st.session_state.kis_stages):
                curr_idx = len(st.session_state.kis_stages) - 1
                st.session_state.active_kis_stage_idx = curr_idx
            
            stg = st.session_state.kis_stages[curr_idx]
            st.markdown(f"<div class='panel-title'>🛠️ CỘT 3: CHI TIẾT GIAI ĐOẠN {curr_idx + 1}</div>", unsafe_allow_html=True)
            
            top_k_val = st.number_input(
                f"Top K giữ lại chuyển sang Giai đoạn sau:",
                min_value=1,
                max_value=5000,
                value=int(stg.get("top_k", 100)),
                key=f"c3_kis_topk_{curr_idx}"
            )
            stg["top_k"] = top_k_val

            st.markdown("**Chọn models & Phân bổ trọng số (Tổng = 100%):**")
            
            # Đồng bộ state widget
            for m in ['clip', 'od', 'ocr', 'asr']:
                if f"c3_chk_{m}_{curr_idx}" not in st.session_state:
                    st.session_state[f"c3_chk_{m}_{curr_idx}"] = stg.get(f"use_{m}", False)
                if f"c3_w_{m}_{curr_idx}" not in st.session_state:
                    st.session_state[f"c3_w_{m}_{curr_idx}"] = int(stg.get(f"weight_{m}", 0))

            # 1. CLIP
            use_c = st.checkbox(
                "Mô hình CLIP (Text-Image)",
                key=f"c3_chk_clip_{curr_idx}",
                on_change=on_kis_toggle,
                args=(curr_idx, "clip")
            )
            if use_c:
                st.slider(
                    "Trọng số CLIP (%)", 0, 100,
                    key=f"c3_w_clip_{curr_idx}",
                    on_change=on_kis_weight_change,
                    args=(curr_idx, "clip")
                )
                stg["query_clip"] = st.text_area("CLIP Prompt / Query:", value=stg.get("query_clip", ""), height=65, key=f"c3_qc_{curr_idx}")
            
            st.divider()
            # 2. OD
            use_d = st.checkbox(
                "Mô hình Object Detection (OD)",
                key=f"c3_chk_od_{curr_idx}",
                on_change=on_kis_toggle,
                args=(curr_idx, "od")
            )
            if use_d:
                st.slider(
                    "Trọng số OD (%)", 0, 100,
                    key=f"c3_w_od_{curr_idx}",
                    on_change=on_kis_weight_change,
                    args=(curr_idx, "od")
                )
                stg["query_od"] = st.text_input("Vật thể (cách nhau dấu phẩy):", value=stg.get("query_od", ""), key=f"c3_qd_{curr_idx}")

            st.divider()
            # 3. OCR
            use_r = st.checkbox(
                "Mô hình OCR (Text in frame)",
                key=f"c3_chk_ocr_{curr_idx}",
                on_change=on_kis_toggle,
                args=(curr_idx, "ocr")
            )
            if use_r:
                st.slider(
                    "Trọng số OCR (%)", 0, 100,
                    key=f"c3_w_ocr_{curr_idx}",
                    on_change=on_kis_weight_change,
                    args=(curr_idx, "ocr")
                )
                stg["query_ocr"] = st.text_input("Chữ cần tìm trên hình:", value=stg.get("query_ocr", ""), key=f"c3_qr_{curr_idx}")

            st.divider()
            # 4. ASR
            use_a = st.checkbox(
                "Mô hình ASR (Speech Transcript)",
                key=f"c3_chk_asr_{curr_idx}",
                on_change=on_kis_toggle,
                args=(curr_idx, "asr")
            )
            if use_a:
                st.slider(
                    "Trọng số ASR (%)", 0, 100,
                    key=f"c3_w_asr_{curr_idx}",
                    on_change=on_kis_weight_change,
                    args=(curr_idx, "asr")
                )
                stg["query_asr"] = st.text_input("Lời thoại / Giọng nói (ASR):", value=stg.get("query_asr", ""), key=f"c3_qa_{curr_idx}")
                stg["padding_asr"] = st.slider("Phạm vi mở rộng (± giây):", 0, 60, value=int(stg.get("padding_asr", 15)), key=f"c3_pa_{curr_idx}")

            w_c = stg.get("weight_clip", 0) if stg.get("use_clip") else 0
            w_d = stg.get("weight_od", 0) if stg.get("use_od") else 0
            w_r = stg.get("weight_ocr", 0) if stg.get("use_ocr") else 0
            w_a = stg.get("weight_asr", 0) if stg.get("use_asr") else 0
            w_tot = w_c + w_d + w_r + w_a
            if w_tot > 0:
                parts = []
                if stg.get("use_clip") and w_c > 0:
                    parts.append(f"CLIP ({w_c}%)")
                if stg.get("use_od") and w_d > 0:
                    parts.append(f"OD ({w_d}%)")
                if stg.get("use_ocr") and w_r > 0:
                    parts.append(f"OCR ({w_r}%)")
                if stg.get("use_asr") and w_a > 0:
                    parts.append(f"ASR ({w_a}%)")
                st.info(f"✨ Tỷ lệ: **{' | '.join(parts)}** (Tổng: **{w_tot}%**)")
            else:
                st.warning("⚠️ Hãy bật ít nhất 1 model để tìm kiếm")

        else:
            curr_idx = st.session_state.active_trake_event_idx
            if curr_idx >= len(st.session_state.trake_events):
                curr_idx = len(st.session_state.trake_events) - 1
                st.session_state.active_trake_event_idx = curr_idx

            ev = st.session_state.trake_events[curr_idx]
            st.markdown(f"<div class='panel-title'>🛠️ CỘT 3: CHI TIẾT EVENT {curr_idx + 1}</div>", unsafe_allow_html=True)

            for m in ['clip', 'od', 'ocr', 'asr']:
                if f"c3_tr_chk_{m}_{curr_idx}" not in st.session_state:
                    st.session_state[f"c3_tr_chk_{m}_{curr_idx}"] = ev.get(f"use_{m}", False)
                if f"c3_tr_w_{m}_{curr_idx}" not in st.session_state:
                    st.session_state[f"c3_tr_w_{m}_{curr_idx}"] = int(ev.get(f"weight_{m}", 0))

            # 1. CLIP
            use_c = st.checkbox(
                "Mô hình CLIP (Text-Image)",
                key=f"c3_tr_chk_clip_{curr_idx}",
                on_change=on_trake_toggle,
                args=(curr_idx, "clip")
            )
            if use_c:
                st.slider(
                    "Trọng số CLIP (%)", 0, 100,
                    key=f"c3_tr_w_clip_{curr_idx}",
                    on_change=on_trake_weight_change,
                    args=(curr_idx, "clip")
                )
                ev["query_clip"] = st.text_area("Mô tả sự kiện (CLIP Query):", value=ev.get("query_clip", ""), height=65, key=f"c3_tr_qc_{curr_idx}")

            st.divider()
            # 2. OD
            use_d = st.checkbox(
                "Mô hình Object Detection (OD)",
                key=f"c3_tr_chk_od_{curr_idx}",
                on_change=on_trake_toggle,
                args=(curr_idx, "od")
            )
            if use_d:
                st.slider(
                    "Trọng số OD (%)", 0, 100,
                    key=f"c3_tr_w_od_{curr_idx}",
                    on_change=on_trake_weight_change,
                    args=(curr_idx, "od")
                )
                ev["query_od"] = st.text_input("Vật thể xuất hiện:", value=ev.get("query_od", ""), key=f"c3_tr_qd_{curr_idx}")

            st.divider()
            # 3. OCR
            use_r = st.checkbox(
                "Mô hình OCR (Text in frame)",
                key=f"c3_tr_chk_ocr_{curr_idx}",
                on_change=on_trake_toggle,
                args=(curr_idx, "ocr")
            )
            if use_r:
                st.slider(
                    "Trọng số OCR (%)", 0, 100,
                    key=f"c3_tr_w_ocr_{curr_idx}",
                    on_change=on_trake_weight_change,
                    args=(curr_idx, "ocr")
                )
                ev["query_ocr"] = st.text_input("Chữ cần tìm trên hình:", value=ev.get("query_ocr", ""), key=f"c3_tr_qr_{curr_idx}")

            st.divider()
            # 4. ASR
            use_a = st.checkbox(
                "Mô hình ASR (Speech Transcript)",
                key=f"c3_tr_chk_asr_{curr_idx}",
                on_change=on_trake_toggle,
                args=(curr_idx, "asr")
            )
            if use_a:
                st.slider(
                    "Trọng số ASR (%)", 0, 100,
                    key=f"c3_tr_w_asr_{curr_idx}",
                    on_change=on_trake_weight_change,
                    args=(curr_idx, "asr")
                )
                ev["query_asr"] = st.text_input("Lời thoại / Giọng nói (ASR):", value=ev.get("query_asr", ""), key=f"c3_tr_qa_{curr_idx}")
                ev["padding_asr"] = st.slider("Phạm vi mở rộng (± giây):", 0, 60, value=int(ev.get("padding_asr", 15)), key=f"c3_tr_pa_{curr_idx}")

            w_c = ev.get("weight_clip", 0) if ev.get("use_clip") else 0
            w_d = ev.get("weight_od", 0) if ev.get("use_od") else 0
            w_r = ev.get("weight_ocr", 0) if ev.get("use_ocr") else 0
            w_a = ev.get("weight_asr", 0) if ev.get("use_asr") else 0
            w_tot = w_c + w_d + w_r + w_a
            if w_tot > 0:
                parts = []
                if ev.get("use_clip") and w_c > 0:
                    parts.append(f"CLIP ({w_c}%)")
                if ev.get("use_od") and w_d > 0:
                    parts.append(f"OD ({w_d}%)")
                if ev.get("use_ocr") and w_r > 0:
                    parts.append(f"OCR ({w_r}%)")
                if ev.get("use_asr") and w_a > 0:
                    parts.append(f"ASR ({w_a}%)")
                st.info(f"✨ Tỷ lệ: **{' | '.join(parts)}** (Tổng: **{w_tot}%**)")
            else:
                st.warning("⚠️ Hãy bật ít nhất 1 model để tìm kiếm")

# Render Fragment Cột 1 & Cột 3
render_controls()


# ---------------------------------------------------------
# CÔ LẬP CỘT 2 TRONG FRAGMENT (CLICK ẢNH LÀ PHÁT VIDEO TRỰC TIẾP)
# ---------------------------------------------------------
@safe_fragment
def render_gallery_panel():
    with col_mid_main:
        st.markdown("<div class='panel-title'>🎬 CỘT 2: VITRIVR MULTIMEDIA GALLERY & PLAYER</div>", unsafe_allow_html=True)
        
        # 1. Trình phát Video Player
        if st.session_state.active_video is not None:
            v_info = st.session_state.active_video
            v_path = get_video_path(v_info["video_id"])
            kf_info = mapper.get_info(v_info["video_id"], v_info["frame_id"])
            
            c_title, c_close = st.columns([8, 2])
            with c_title:
                st.markdown(f"**🎞️ Đang phát:** `{v_info['video_id']}` | **Keyframe:** `{kf_info['n']}` | **Frame Index:** `{kf_info['frame_idx']}` | **Time:** `{kf_info['pts_time']:.2f}s`")
            with c_close:
                st.button("❌ Đóng Video", key="btn_close_v", on_click=close_video_callback, use_container_width=True)

            if v_path and os.path.exists(v_path):
                st.video(v_path, start_time=int(kf_info["pts_time"]), autoplay=True)
            else:
                st.warning(f"⚠️ Chưa tìm thấy file video `{v_info['video_id']}.mp4` trong `data/videos`. Khi bạn copy video vào thư mục này, video sẽ tự động phát từ mốc `{kf_info['pts_time']:.2f}s` (Frame {kf_info['frame_idx']})!")
                
            st.divider()

        # 2. Thanh điều hướng Gallery / Kết quả
        c_mode1, c_mode2 = st.columns([7, 3])
        with c_mode1:
            view_modes = ["🖼️ Thư viện Keyframes"]
            has_results = (st.session_state.kis_results is not None) or (st.session_state.trake_results is not None)
            if has_results:
                view_modes.insert(0, "🎯 Kết quả tìm kiếm")

            st.session_state.col2_view_mode = st.radio(
                "Chế độ Cột 2:",
                view_modes,
                index=0 if st.session_state.col2_view_mode in view_modes else 0,
                horizontal=True,
                label_visibility="collapsed"
            )
        with c_mode2:
            if has_results and st.session_state.col2_view_mode == "🎯 Kết quả tìm kiếm":
                if st.button("🗑️ Xóa kết quả", use_container_width=True):
                    st.session_state.kis_results = None
                    st.session_state.trake_results = None
                    st.session_state.col2_view_mode = "🖼️ Thư viện Keyframes"
                    st.rerun()

        # Render Kết quả
        if st.session_state.col2_view_mode == "🎯 Kết quả tìm kiếm":
            if "KIS" in st.session_state.query_type and st.session_state.kis_results:
                res = st.session_state.kis_results
                if res.get("status") == "success":
                    results = res.get("results", [])
                    st.success(f"🎯 Tìm thấy {len(results)} kết quả KIS (Click vào ảnh để xem video):")
                    
                    # Nút hiển thị chuỗi nộp bài thi AIC
                    with st.expander("📋 Xem danh sách ID nộp bài thi (DRES / VBS Format)"):
                        sub_lines = [mapper.format_submission_line(item['video_id'], item['frame_id'], mode="frame_idx") for item in results]
                        st.code("\n".join(sub_lines), language="text")

                    for i in range(0, len(results), 3):
                        cols = st.columns(3)
                        for j in range(3):
                            idx = i + j
                            if idx < len(results):
                                item = results[idx]
                                with cols[j]:
                                    img_url = get_image_url(item['video_id'], item['frame_id'])
                                    kf_info = mapper.get_info(item['video_id'], item['frame_id'])
                                    start_sec = int(kf_info['pts_time'])
                                    score_pct = item['score'] * 100
                                    
                                    st.markdown(
                                        f"<div class='vitrivr-card' onclick=\"const b=this.parentElement.querySelector('button'); if(b) b.click();\">"
                                        f"<div class='score-badge'>{score_pct:.1f}%</div>"
                                        f"<img src='{img_url}' class='card-img' onerror=\"this.onerror=null; this.src='https://placehold.co/240x135?text=F+{item['frame_id']}';\" />"
                                        f"<span class='meta-tag-primary'><b>{item['video_id']}</b> | KF {kf_info['n']}</span>"
                                        f"<span class='meta-tag-secondary'>Frame {kf_info['frame_idx']} ({kf_info['pts_time']:.1f}s)</span>"
                                        f"</div>",
                                        unsafe_allow_html=True
                                    )
                                    st.button(
                                        f"play_res_{item['video_id']}_{item['frame_id']}",
                                        key=f"p_res_{item['video_id']}_{item['frame_id']}_{idx}",
                                        on_click=play_video_callback,
                                        args=(item['video_id'], item['frame_id'], start_sec)
                                    )
                else:
                    st.error(res.get("message", "Lỗi tìm kiếm KIS."))

            elif st.session_state.trake_results:
                res = st.session_state.trake_results
                if res.get("status") == "success":
                    results = res.get("results", [])
                    st.success(f"⏱️ Tìm thấy {len(results)} chuỗi TRAKE khớp nhất (Click vào ảnh để xem video):")
                    for rank_idx, item in enumerate(results, 1):
                        st.markdown(f"**#{rank_idx} {item['video_id']}** (Score: `{item['score']:.4f}`)")
                        frames = item["frames"]
                        ev_cols = st.columns(len(frames))
                        for ev_idx, fid in enumerate(frames):
                            with ev_cols[ev_idx]:
                                img_url = get_image_url(item['video_id'], fid)
                                kf_info = mapper.get_info(item['video_id'], fid)
                                start_sec = int(kf_info['pts_time'])
                                st.markdown(
                                    f"<div class='vitrivr-card' onclick=\"const b=this.parentElement.querySelector('button'); if(b) b.click();\">"
                                    f"<img src='{img_url}' class='card-img' onerror=\"this.onerror=null; this.src='https://placehold.co/240x135?text=F+{fid}';\" />"
                                    f"<span class='meta-tag-primary'>Event {ev_idx+1} | KF {kf_info['n']}</span>"
                                    f"<span class='meta-tag-secondary'>Frame {kf_info['frame_idx']} ({kf_info['pts_time']:.1f}s)</span>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                                st.button(
                                    f"play_tr_{item['video_id']}_{fid}",
                                    key=f"p_tr_{item['video_id']}_{fid}_{ev_idx}",
                                    on_click=play_video_callback,
                                    args=(item['video_id'], fid, start_sec)
                                )

                        item_vid = item["video_id"]
                        first_f = frames[0]
                        first_info = mapper.get_info(item_vid, first_f)
                        last_f = frames[-1]
                        last_info = mapper.get_info(item_vid, last_f)
                        st.button(
                            f"🎬 Chiếu toàn bộ sự kiện (KF {first_info['n']} ➔ KF {last_info['n']} / {first_info['pts_time']:.1f}s - {last_info['pts_time']:.1f}s)",
                            key=f"p_all_{item_vid}_{rank_idx}",
                            use_container_width=True,
                            on_click=play_video_callback,
                            args=(item_vid, first_f, int(first_info['pts_time']))
                        )
                        st.divider()
                else:
                    st.error(res.get("message", "Lỗi tìm kiếm TRAKE."))

        # Render Gallery Kho
        else:
            c_filter, c_size = st.columns([7, 3])
            with c_filter:
                video_options = ["Tất cả videos"] + unique_videos
                selected_vid = st.selectbox(
                    "Lọc theo Video:",
                    video_options,
                    index=video_options.index(st.session_state.gallery_filter_video) if st.session_state.gallery_filter_video in video_options else 0,
                    key="sb_gallery_video"
                )
                if selected_vid != st.session_state.gallery_filter_video:
                    st.session_state.gallery_filter_video = selected_vid
                    st.session_state.gallery_page = 1
                    st.rerun()

            with c_size:
                page_size = st.selectbox("Số lượng / trang:", [12, 24, 48], index=0)

            all_gallery_items = get_dataset_keyframes(metadata_clip, st.session_state.gallery_filter_video)
            total_items = len(all_gallery_items)
            total_pages = max(1, math.ceil(total_items / page_size))

            if st.session_state.gallery_page > total_pages:
                st.session_state.gallery_page = 1

            c_prev, c_page_info, c_next = st.columns([2, 6, 2])
            with c_prev:
                if st.button("◀️ Trước", use_container_width=True, disabled=(st.session_state.gallery_page <= 1)):
                    st.session_state.gallery_page -= 1
                    st.rerun()
            with c_page_info:
                st.markdown(f"<div style='text-align:center; padding-top:6px;'><small>Trang <b>{st.session_state.gallery_page} / {total_pages}</b> ({total_items:,} Keyframes - Click ảnh để phát)</small></div>", unsafe_allow_html=True)
            with c_next:
                if st.button("Sau ▶️", use_container_width=True, disabled=(st.session_state.gallery_page >= total_pages)):
                    st.session_state.gallery_page += 1
                    st.rerun()

            start_idx = (st.session_state.gallery_page - 1) * page_size
            end_idx = min(start_idx + page_size, total_items)
            current_page_items = all_gallery_items[start_idx:end_idx]

            for i in range(0, len(current_page_items), 3):
                cols = st.columns(3)
                for j in range(3):
                    idx = i + j
                    if idx < len(current_page_items):
                        vid_id, fid = current_page_items[idx]
                        with cols[j]:
                            img_url = get_image_url(vid_id, fid)
                            kf_info = mapper.get_info(vid_id, fid)
                            start_sec = int(kf_info['pts_time'])
                            
                            st.markdown(
                                f"<div class='vitrivr-card' onclick=\"const b=this.parentElement.querySelector('button'); if(b) b.click();\">"
                                f"<img src='{img_url}' class='card-img' onerror=\"this.onerror=null; this.src='https://placehold.co/240x135?text=F+{fid}';\" />"
                                f"<span class='meta-tag-primary'><b>{vid_id}</b> | KF {kf_info['n']}</span>"
                                f"<span class='meta-tag-secondary'>Frame {kf_info['frame_idx']} ({kf_info['pts_time']:.1f}s)</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                            st.button(
                                f"play_gal_{vid_id}_{fid}",
                                key=f"p_gal_{vid_id}_{fid}_{idx}",
                                on_click=play_video_callback,
                                args=(vid_id, fid, start_sec)
                            )

# Render Fragment Cột 2
render_gallery_panel()