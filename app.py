import os
import streamlit as st
import importlib
import backend
from config import KEYFRAME_DIR

st.set_page_config(page_title="AIC 2026", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0B0F19; color: white; }
    .hero-title { text-align: center; font-size: 3rem; font-weight: 800; color: #00F2FE; }
    .result-card { background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 10px; margin-bottom: 10px;}
    .metric-box { text-align: center; padding: 5px; background: rgba(0,0,0,0.3); border-radius: 8px; margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="🧠 Đang nạp AI Models...")
def get_search_engine():
    importlib.reload(backend)
    return backend.AICSearchEngine()

engine = get_search_engine()

def get_real_image_path(video_id, frame_id):
    if KEYFRAME_DIR is None:
        return None
        
    video_name = video_id.replace('.mp4', '')
    prefix = video_name.split('_')[0] if '_' in video_name else ''
    
    possible_dirs = [
        KEYFRAME_DIR,
        os.path.join(KEYFRAME_DIR, f"Keyframes_{prefix}"),
        os.path.join(KEYFRAME_DIR, prefix),
    ]
    
    for parent_dir in possible_dirs:
        if not os.path.exists(parent_dir):
            continue
            
        for fmt in ["{:03d}.jpg", "{:04d}.jpg", "{:05d}.jpg", "{:06d}.jpg", "{}.jpg"]:
            path = os.path.join(parent_dir, video_name, fmt.format(int(frame_id)))
            if os.path.exists(path):
                return path
                
    return None

st.markdown("<h1 class='hero-title'>AIC 2026 RETRIEVAL</h1>", unsafe_allow_html=True)

tab_kis, tab_qa, tab_trake = st.tabs(["🎯 Dạng 1: Textual KIS", "❓ Dạng 2: Hỏi-Đáp (Q&A)", "⏱️ Dạng 3: TRAKE"])

with tab_kis:
    col1, col2 = st.columns([8, 2])
    with col1:
        query_kis = st.text_area("Mô tả sự kiện (Chữ):", "A photograph taken from above...", height=60, label_visibility="collapsed")
        query_objs_str = st.text_input("Vật thể hỗ trợ OD (cách nhau bởi dấu phẩy):", "cat, dog")

        with st.expander("⚙️ Tùy chỉnh Bật/Tắt Model & Trọng số", expanded=True):
            c1, c2, c3 = st.columns(3)
            use_clip = c1.checkbox("Dùng CLIP (Mặc định 0.5)", value=True)
            use_od = c2.checkbox("Dùng OD (Mặc định 0.3)", value=True)
            use_ocr = c3.checkbox("Dùng OCR (Mặc định 0.2)", value=True)

            w_clip = 0.5 if use_clip else 0.0
            w_od = 0.3 if use_od else 0.0
            w_ocr = 0.2 if use_ocr else 0.0

            total_w = w_clip + w_od + w_ocr
            if total_w > 0:
                w_clip /= total_w
                w_od /= total_w
                w_ocr /= total_w
                st.caption(f"✨ Trọng số phân bổ thực tế: **CLIP ({w_clip:.2f}) - OD ({w_od:.2f}) - OCR ({w_ocr:.2f})**")

    with col2:
        if st.button("♻️ Nạp lại Backend"):
            st.cache_resource.clear()
            st.rerun()
        btn_kis = st.button("⚡ TÌM KIẾM", key="btn_kis", use_container_width=True)

    if btn_kis and total_w > 0:
        with st.spinner("🚀 Đang chạy thuật toán..."):
            obj_list = [obj.strip() for obj in query_objs_str.split(",")] if query_objs_str.strip() else []
            res = engine.search_kis(query_kis, obj_list, top_k=10, w_clip=w_clip, w_od=w_od, w_ocr=w_ocr)

            if res.get("status") == "success":
                with st.expander("📄 Xem log điểm số"):
                    log_text = "\n".join([f"Rank {item['rank']}: {item['video_id']}, frame={item['frame_id']}, SCORE={item['score']:.6f}" for item in res['results']])
                    st.code(log_text, language="text")

                results_sorted = sorted(res['results'], key=lambda x: x['score'], reverse=True)
                for i in range(0, 10, 5):
                    cols = st.columns(5)
                    for j in range(5):
                        idx = i + j
                        if idx < len(results_sorted):
                            item = results_sorted[idx]
                            with cols[j]:
                                st.markdown(f"<div class='result-card'><div class='metric-box'><div style='color:#4CAF50; font-weight:bold;'>{item['score']*100:.1f}%</div><div style='font-size:0.8rem;'>{item['video_id']}</div></div></div>", unsafe_allow_html=True)
                                real_img = get_real_image_path(item['video_id'], item['frame_id'])
                                if real_img:
                                    st.image(real_img, use_container_width=True)
                                else:
                                    st.image(f"https://placehold.co/300x200?text={item['frame_id']}", use_container_width=True)
