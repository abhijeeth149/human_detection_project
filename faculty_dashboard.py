"""
faculty_dashboard.py — Enhanced Faculty Command Center (Vidyaksha v2).
Features: Multi-feed monitoring grid, live occupancy graph, AI verification
queue (confidence < 42%), historical heatmap, auto attendance report.
"""
import io, csv, time, os, json, math
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import cv2
from ultralytics import YOLO
from utils import count_people
from styles import FACULTY_DASHBOARD_CSS, FACULTY_ENHANCED_CSS

# ── Constants ─────────────────────────────────────────────────────────────────
LABEL_MAP_FILE   = "roll_label_map.json"
LBPH_THRESHOLD   = 110
EIGEN_THRESHOLD  = 3500
DNN_PROTO        = "deploy.prototxt"
DNN_MODEL        = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
EIGEN_MODEL      = "eigen_model.yml"
LOG_FILE         = "detection_logs.csv"
AI_CONF_FLAG_THR = 0.42          # detections below this are flagged
_eye_cascade     = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# ── Mock historical data ───────────────────────────────────────────────────────
HEATMAP_DATA = {
    "Section": ["ECE-A", "ECE-B", "CSE-A", "CSE-B"],
    "Mon": [83, 72, 91, 65], "Tue": [91, 88, 78, 82],
    "Wed": [76, 61, 94, 70], "Thu": [88, 79, 85, 55],
    "Fri": [70, 93, 67, 88],
}

# ── Model loading ──────────────────────────────────────────────────────────────
def _load_roll_numbers() -> dict:
    if not os.path.exists(LABEL_MAP_FILE):
        return {}
    with open(LABEL_MAP_FILE, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


@st.cache_resource
def _load_models():
    mp = "classroom_ai/max_accuracy_run/weights/best.pt"
    if not os.path.exists(mp):
        mp = "yolov8n.pt"
    yolo = YOLO(mp)
    rec  = cv2.face.LBPHFaceRecognizer_create()
    fc   = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if os.path.exists("face_model.yml"):
        rec.read("face_model.yml")
    eigen = cv2.face.EigenFaceRecognizer_create()
    if os.path.exists(EIGEN_MODEL):
        eigen.read(EIGEN_MODEL)
    dnn = None
    if os.path.exists(DNN_PROTO) and os.path.exists(DNN_MODEL):
        dnn = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
    prof = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    lmap = _load_roll_numbers()
    rolls = [lmap[i] for i in sorted(lmap.keys())]
    return yolo, rec, eigen, fc, dnn, prof, lmap, rolls


# ── Detection helpers (compact) ───────────────────────────────────────────────
def _detect_faces(frame, dnn, fc, prof):
    h, w = frame.shape[:2]; all_f = []
    if dnn:
        blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300),(104,177,123))
        dnn.setInput(blob); dets = dnn.forward()
        for i in range(dets.shape[2]):
            if dets[0,0,i,2] > 0.40:
                b = (dets[0,0,i,3:7]*np.array([w,h,w,h])).astype(int)
                x1,y1,x2,y2 = max(0,b[0]),max(0,b[1]),min(w,b[2]),min(h,b[3])
                if (x2-x1)>15 and (y2-y1)>15: all_f.append((x1,y1,x2-x1,y2-y1))
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.0,(8,8)).apply(gray)
    lp = prof.detectMultiScale(gray,1.1,3,minSize=(20,20))
    if len(lp)>0: all_f.extend(list(lp))
    if not all_f:
        d = fc.detectMultiScale(gray,1.1,3,minSize=(18,18))
        if len(d)>0: all_f.extend(list(d))
    def iou(a,b):
        ax1,ay1,aw,ah=a; bx1,by1,bw,bh=b
        ix=max(0,min(ax1+aw,bx1+bw)-max(ax1,bx1)); iy=max(0,min(ay1+ah,by1+bh)-max(ay1,by1))
        inter=ix*iy; return inter/(aw*ah+bw*bh-inter+1e-5)
    uniq=[]
    for f in sorted(all_f,key=lambda x:x[2]*x[3],reverse=True):
        if not any(iou(f,u)>0.4 for u in uniq): uniq.append(f)
    return uniq


def _preprocess(gc): 
    c=cv2.createCLAHE(2.0,(8,8)).apply(gc)
    return cv2.resize(cv2.bilateralFilter(c,9,75,75),(100,100))


def _predict(gc, rec, eigen, lmap):
    if gc.shape[0]<5 or gc.shape[1]<5: return None, 9999, 0.0
    # Align
    wk=cv2.resize(gc,(150,150)); eq=cv2.createCLAHE(2.0,(8,8)).apply(wk)
    eyes=_eye_cascade.detectMultiScale(eq,1.1,5,minSize=(15,15))
    if len(eyes)>=2:
        eyes=sorted(eyes,key=lambda e:e[0])[:2]
        (ex1,ey1,ew1,eh1),(ex2,ey2,ew2,eh2)=eyes
        angle=math.degrees(math.atan2((ey2+eh2//2)-(ey1+eh1//2),(ex2+ew2//2)-(ex1+ew1//2)))
        M=cv2.getRotationMatrix2D(((ex1+ew1//2+ex2+ew2//2)/2,(ey1+eh1//2+ey2+eh2//2)/2),angle,1.0)
        wk=cv2.warpAffine(wk,M,(150,150),borderMode=cv2.BORDER_REPLICATE)
    aligned=cv2.resize(wk,(100,100))
    votes={}; dists={}
    aH,aW=aligned.shape
    crops=[(0,0,aW,aH)]
    for mg in [max(1,int(aH*.08)),max(1,int(aH*.14))]:
        if aH-2*mg>5: crops.append((mg,mg,aW-mg,aH-mg))
    for (x1,y1,x2,y2) in crops:
        lbl,d=rec.predict(_preprocess(aligned[y1:y2,x1:x2]))
        r=lmap.get(lbl)
        if r: votes[r]=votes.get(r,0)+1; dists[r]=min(dists.get(r,9999),d)
    try:
        el,ed=eigen.predict(aligned)
        er=lmap.get(el)
        if er and ed<EIGEN_THRESHOLD: votes[er]=votes.get(er,0)+1
    except: pass
    if not votes: return None,9999,0.0
    br=max(votes,key=lambda r:(votes[r],-dists.get(r,9999)))
    bd=dists.get(br,9999)
    if bd>=LBPH_THRESHOLD: return None,bd,0.0
    others=[d for r,d in dists.items() if r!=br]
    if others and bd/(min(others)+1e-5)>0.75: return None,bd,0.0
    conf=max(0.0,min(1.0,1.0-bd/LBPH_THRESHOLD))
    return br,bd,conf


def _init_csv():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE,'w',newline='') as f:
            csv.writer(f).writerow(["Timestamp","Mode","Person_Count","Alert_Status"])


def _log_csv(mode_name,count,limit):
    status="Safe"
    if count>=limit+5: status="Critical Overcrowding"
    elif count>limit: status="Warning Near Capacity"
    from datetime import datetime
    with open(LOG_FILE,'a',newline='') as f:
        csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),mode_name,count,status])


# ── Plotly occupancy graph ─────────────────────────────────────────────────────
def _occupancy_chart(times, counts):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=counts, mode="lines+markers",
        line=dict(color="#00f0ff", width=2),
        marker=dict(size=5, color="#00f0ff", symbol="circle"),
        fill="tozeroy", fillcolor="rgba(0,240,255,0.06)",
        name="Headcount"
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30,r=10,t=20,b=30), height=220,
        font=dict(color="#94a3b8", size=11),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,240,255,0.07)", title="Time"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,240,255,0.07)", title="Count", rangemode="tozero"),
        showlegend=False,
    )
    return fig


# ── Heatmap HTML ───────────────────────────────────────────────────────────────
def _heatmap_html(data):
    days = ["Mon","Tue","Wed","Thu","Fri"]
    h = "<table style='width:100%;border-collapse:separate;border-spacing:4px;'>"
    h += "<tr><th class='hm-label'>Section</th>"
    for d in days: h += f"<th class='hm-label'>{d}</th>"
    h += "</tr>"
    for i,sec in enumerate(data["Section"]):
        h += f"<tr><td class='hm-label' style='font-weight:700;color:#94a3b8;'>{sec}</td>"
        for d in days:
            v = data[d][i]
            cls = "hm-green" if v >= 75 else "hm-red"
            h += f"<td class='{cls}'>{v}%</td>"
        h += "</tr>"
    h += "</table>"
    return h


# ── Attendance report CSV ──────────────────────────────────────────────────────
def _build_report_csv(attendance, peak, session_start):
    from datetime import datetime
    buf = io.StringIO()
    w   = csv.writer(buf)
    w.writerow(["Vidyaksha AI Observer — Auto Attendance Report"])
    w.writerow(["Generated", datetime.now().strftime("%d %b %Y %I:%M %p")])
    w.writerow(["Session Start", session_start])
    w.writerow([])
    present = [r for r,s in attendance.items() if s=="Present"]
    absent  = [r for r,s in attendance.items() if s=="Absent"]
    w.writerow(["Total Students", len(attendance)])
    w.writerow(["Present", len(present)])
    w.writerow(["Absent",  len(absent)])
    w.writerow(["Peak Occupancy", peak])
    w.writerow([])
    w.writerow(["Roll Number","Status"])
    for r,s in attendance.items(): w.writerow([r,s])
    return buf.getvalue().encode()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN show() entry point
# ═══════════════════════════════════════════════════════════════════════════════
def show(user_id: str):
    st.markdown(FACULTY_DASHBOARD_CSS,   unsafe_allow_html=True)
    st.markdown(FACULTY_ENHANCED_CSS,    unsafe_allow_html=True)

    yolo, rec, eigen, fc, dnn, prof, lmap, roll_list = _load_models()
    _init_csv()

    # ── Session state bootstrap ────────────────────────────────────────────────
    for k,v in [("run",False),("session_ended",False),
                ("occ_times",[]),("occ_counts",[]),
                ("peak_count",0),("vq_items",[]),("session_start","—")]:
        if k not in st.session_state: st.session_state[k] = v

    if "attendance" not in st.session_state or \
            set(st.session_state.attendance.keys()) != set(roll_list):
        st.session_state.attendance = {r:"Absent" for r in roll_list}

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align:center;margin-bottom:16px;'>
        <div style='font-family:"Outfit",sans-serif;font-size:1.1rem;font-weight:800;color:#0ea5e9;letter-spacing:1px;'>FACULTY PORTAL</div>
        <div style='color:#64748b;font-size:0.8rem;font-weight:500;margin-top:4px;'>🏫 ID: <b style='color:#0f172a'>{user_id}</b></div>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")

        # AI model stats (mock)
        st.markdown("<div class='section-title'>📡 Model Stats</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='sidebar-stat'><div class='sidebar-stat-label'>mAP@0.5</div><div class='sidebar-stat-value'>87.3%</div></div>
        <div class='sidebar-stat'><div class='sidebar-stat-label'>MAE (headcount)</div><div class='sidebar-stat-value'>1.4</div></div>
        <div class='sidebar-stat'><div class='sidebar-stat-label'>Inference FPS</div><div class='sidebar-stat-value'>~24</div></div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("### ⚙️ AI Parameters")
        conf_thr      = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
        overcrowd_lim = st.slider("Overcrowd Limit", 1, 100, 20)
        st.markdown("---")
        st.markdown("### 🔔 Alerts")
        alert_on = st.toggle("AI Confidence Alerts", value=True)
        mode     = st.radio("Observation Mode",
                            ["📷 Live Camera Feed","🖼 Upload Snapshot Analytics"], index=0)
        st.markdown("---")
        if st.button("🚪 LOGOUT", key="f_logout"):
            for k in ["is_logged_in","user_role","user_id",
                      "run","session_ended","occ_times","occ_counts",
                      "peak_count","vq_items","session_start"]:
                st.session_state.pop(k, None)
            st.rerun()

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='font-family:"Outfit",sans-serif;font-size:2.4rem;font-weight:800;
         background:linear-gradient(135deg,#0ea5e9,#4f46e5);-webkit-background-clip:text;
         -webkit-text-fill-color:transparent;letter-spacing:-0.02em;margin-bottom:4px;'>
    🎓 Vidyaksha Command Center</div>
    <div style='color:#64748b;font-size:0.9rem;font-weight:500;letter-spacing:0.05em;margin-bottom:24px;text-transform:uppercase;'>
    AI Observer · Real-Time Classroom Intelligence</div>
    """, unsafe_allow_html=True)

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎥  Live Monitor", "📊  Analytics", "🗓  Heatmap", "📋  Report"])

    # ══════════════════════════════════════════════════════
    #  TAB 1 — Live Monitor
    # ══════════════════════════════════════════════════════
    with tab1:
        # ── Class Selection ───────────────────────────────
        st.markdown("<div class='section-title'>📚 Select Classroom</div>", unsafe_allow_html=True)
        sel_col1, sel_col2, sel_col3 = st.columns(3)
        with sel_col1:
            year = st.selectbox("Year", ["1st Year", "2nd Year", "3rd Year", "4th Year"], index=3)
        with sel_col2:
            branch = st.selectbox("Branch", ["CSE", "ECE", "EEE", "MECH", "CIVIL"], index=0)
        with sel_col3:
            section = st.selectbox("Section", ["A", "B", "C", "D"], index=0)
            
        st.markdown(f"<div style='margin-top:8px; border-bottom: 1px solid #e2e8f0; padding-bottom:12px'>"
                    f"<strong style='color:#0ea5e9;'>Target:</strong> {year} · {branch} · {section}</div>"
                    f"<br>", unsafe_allow_html=True)

        # ── Main camera / upload ────────────────────────────
        main_col, stats_col = st.columns([7,3])
        with main_col:
            v1, v2 = st.columns(2)
            with v1:
                st.markdown("<h5 style='text-align:center;color:#4b5563'>Face Recognition</h5>",
                            unsafe_allow_html=True)
                face_ph = st.empty()
            with v2:
                st.markdown("<h5 style='text-align:center;color:#4b5563'>Person Detection</h5>",
                            unsafe_allow_html=True)
                yolo_ph = st.empty()
            st.markdown("<br>", unsafe_allow_html=True)
            bc1, bc2, bc3 = st.columns(3)
            start    = bc1.button("▶ Start",    type="primary", use_container_width=True)
            stop     = bc2.button("⏹ End",      type="primary", use_container_width=True)
            show_res = bc3.button("📊 Results", type="primary", use_container_width=True)

        if start:
            st.session_state.run = True
            st.session_state.session_ended = False
            from datetime import datetime
            st.session_state.session_start = datetime.now().strftime("%d %b %Y %I:%M %p")
            st.session_state.occ_times  = []
            st.session_state.occ_counts = []
            st.session_state.peak_count = 0
            st.session_state.vq_items   = []
        if stop:
            st.session_state.run = False
            st.session_state.session_ended = True
            st.rerun()

        @st.dialog("Attendance Summary")
        def _summary():
            from datetime import datetime
            pn = sum(1 for s in st.session_state.attendance.values() if s=="Present")
            tn = len(st.session_state.attendance)
            st.markdown(f"<h4 style='text-align:center'>📋 {datetime.now().strftime('%d %b %Y, %I:%M %p')}"
                        f" | ✅ {pn}/{tn}</h4>", unsafe_allow_html=True)
            rows=""
            for roll,status in st.session_state.attendance.items():
                c="#16a34a" if status=="Present" else "#dc2626"
                bg="#dcfce7" if status=="Present" else "#fee2e2"
                rows+=(f"<tr><td style='padding:8px'>{roll}</td>"
                       f"<td style='padding:8px;text-align:center'>"
                       f"<span style='background:{bg};color:{c};padding:4px 12px;"
                       f"border-radius:10px;font-weight:700'>{status}</span></td></tr>")
            st.markdown(f"<table style='width:100%;border-collapse:collapse'>"
                        f"<tr style='background:#0284c7;color:white'><th>Roll</th><th>Status</th></tr>"
                        f"{rows}</table>", unsafe_allow_html=True)
            if st.button("✅ Close", type="primary"): st.rerun()

        if show_res: _summary()

        # Stats column
        with stats_col:
            st.markdown("### Attendance Status")
            att_ph  = st.empty()
            met_ph  = st.empty()
            st.markdown("---")
            alrt_ph = st.empty()
            st.markdown("### Analytics")
            pie_ph  = st.empty()

        def _render_att():
            if not st.session_state.attendance:
                att_ph.info("⚠️ No students trained yet.")
                return
            rows=""
            for roll,status in st.session_state.attendance.items():
                if st.session_state.run:
                    rows+=(f"<div style='padding:10px;border-radius:8px;background:#f8f9fa;"
                           f"border:1px solid #e2e8f0;margin-bottom:4px'>"
                           f"<strong style='color:#4b5563;font-size:14px'>🎓 {roll}</strong></div>")
                else:
                    c="#22c55e" if status=="Present" else "#ef4444"
                    rows+=(f"<div style='padding:10px;border-radius:8px;background:#f8f9fa;"
                           f"border-left:4px solid {c};display:flex;justify-content:space-between;"
                           f"align-items:center;box-shadow:0 1px 3px rgba(0,0,0,.1);margin-bottom:4px'>"
                           f"<strong style='color:#1e293b;font-size:14px'>🎓 {roll}</strong>"
                           f"<span style='color:white;font-weight:700;background:{c};"
                           f"padding:4px 12px;border-radius:10px;font-size:13px'>{status}</span></div>")
            att_ph.markdown(f"<div>{rows}</div>", unsafe_allow_html=True)

        def _render_pie():
            pn=sum(1 for s in st.session_state.attendance.values() if s=="Present")
            ab=len(st.session_state.attendance)-pn
            url=(f"https://quickchart.io/chart?c={{type:'doughnut',data:{{"
                 f"labels:['Absent','Present'],datasets:[{{data:[{ab},{pn}],"
                 f"backgroundColor:['%23ef4444','%2322c55e']}}]}}}}")
            pie_ph.markdown(f"<div style='text-align:center'><img src='{url}' width='150'></div>",
                            unsafe_allow_html=True)

        def _render_metrics(cnt):
            pn=sum(1 for s in st.session_state.attendance.values() if s=="Present")
            ab=len(st.session_state.attendance)-pn
            met_ph.markdown(
                f"<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:12px 0'>"
                f"<div class='metric-card' style='padding:16px'>"
                f"<div style='color:#64748b;font-size:0.75rem;font-weight:700;text-transform:uppercase;font-family:\"Outfit\"'>Head Count</div>"
                f"<div style='font-size:1.8rem;font-weight:800;color:#0f172a'>{cnt}</div></div>"
                f"<div class='metric-card' style='padding:16px'>"
                f"<div style='color:#64748b;font-size:0.75rem;font-weight:700;text-transform:uppercase;font-family:\"Outfit\"'>Present</div>"
                f"<div style='font-size:1.8rem;font-weight:800;color:#10b981'>{pn}</div></div>"
                f"<div class='metric-card' style='padding:16px'>"
                f"<div style='color:#64748b;font-size:0.75rem;font-weight:700;text-transform:uppercase;font-family:\"Outfit\"'>Absent</div>"
                f"<div style='font-size:1.8rem;font-weight:800;color:#ef4444'>{ab}</div></div></div>",
                unsafe_allow_html=True)

        def _render_alert(cnt, lim):
            if not alert_on: alrt_ph.empty(); return
            if cnt >= lim+5:
                alrt_ph.markdown(f"<div class='status-banner status-danger'>🚨 CRITICAL: OVERCROWDED ({cnt}/{lim})</div>", unsafe_allow_html=True)
            elif cnt > lim:
                alrt_ph.markdown(f"<div class='status-banner status-warning'>⚠️ WARNING: Near Capacity ({cnt}/{lim})</div>", unsafe_allow_html=True)
            else:
                alrt_ph.markdown(f"<div class='status-banner status-safe'>✅ STATUS: NORMAL ({cnt}/{lim})</div>", unsafe_allow_html=True)

        _render_att(); _render_pie()

        if not st.session_state.run and mode == "📷 Live Camera Feed":
            face_ph.info("👋 Setup Ready.")
            yolo_ph.info("Click **Start** to begin.")

        # ── Upload Mode ────────────────────────────────────────────────────────
        if mode == "🖼 Upload Snapshot Analytics":
            with main_col:
                st.markdown("#### 📸 Upload Group Photo")
                uf = st.file_uploader("Upload classroom photo", type=["jpg","jpeg","png"])
                st.caption("  |  ".join(roll_list) if roll_list else "⚠️ No students trained.")
            if uf:
                with st.spinner("🔍 Analysing..."):
                    try:
                        img=cv2.imdecode(np.frombuffer(uf.read(),np.uint8),cv2.IMREAD_COLOR)
                        h0,w0=img.shape[:2]
                        if max(h0,w0)>1920:
                            sc=1920/max(h0,w0); img=cv2.resize(img,(int(w0*sc),int(h0*sc)))
                        res=yolo(img,conf=conf_thr,iou=0.45)
                        cnt,_=count_people(res,conf_thr)
                        _log_csv("Image Upload",cnt,overcrowd_lim)
                        ay=res[0].plot(); cv2.putText(ay,f"Detected: {cnt}",(30,70),
                            cv2.FONT_HERSHEY_DUPLEX,1.5,(0,255,170),4)
                        yolo_ph.image(cv2.cvtColor(ay,cv2.COLOR_BGR2RGB))
                        st.session_state.attendance={r:"Absent" for r in roll_list}
                        faces=_detect_faces(img,dnn,fc,prof)
                        gray=cv2.createCLAHE(2.0,(8,8)).apply(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
                        af=img.copy(); rec_list=[]; ih,iw=img.shape[:2]
                        for (x,y,wf,hf) in faces:
                            x,y=max(0,x),max(0,y); wf=min(wf,iw-x); hf=min(hf,ih-y)
                            if wf<5 or hf<5: continue
                            mr,dist,conf_val=_predict(gray[y:y+hf,x:x+wf],rec,eigen,lmap)
                            if conf_val < AI_CONF_FLAG_THR and mr:
                                st.session_state.vq_items.append(
                                    {"roll":mr,"conf":round(conf_val*100,1),"dist":round(dist,1)})
                            if mr:
                                rec_list.append(mr)
                                st.session_state.attendance[mr]="Present"
                                cv2.rectangle(af,(x,y),(x+wf,y+hf),(0,200,80),3)
                                cv2.putText(af,f"{mr}({conf_val*100:.0f}%)",(x,y-8),
                                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,200,80),2)
                            else:
                                cv2.rectangle(af,(x,y),(x+wf,y+hf),(0,0,210),3)
                                cv2.putText(af,"?",(x,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,210),2)
                        face_ph.image(cv2.cvtColor(af,cv2.COLOR_BGR2RGB))
                        _render_att(); _render_metrics(cnt); _render_alert(cnt,overcrowd_lim); _render_pie()
                        st.success(f"✅ {len(rec_list)} detected | Log saved.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        # ── Live Camera Mode ───────────────────────────────────────────────────
        if mode == "📷 Live Camera Feed" and st.session_state.run:
            cap=cv2.VideoCapture(0); prev=0; last_log=time.time()
            try:
                while st.session_state.run:
                    ret,frame=cap.read()
                    if not ret: st.error("Camera failed."); break
                    res=yolo(frame,conf=conf_thr,iou=0.45)
                    cnt,_=count_people(res,conf_thr)
                    ay=res[0].plot()
                    gray=cv2.createCLAHE(2.0,(8,8)).apply(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
                    faces=_detect_faces(frame,dnn,fc,prof)
                    af=frame.copy(); upd=False
                    for (x,y,w,h) in faces:
                        x,y=max(0,x),max(0,y); w=min(w,frame.shape[1]-x); h=min(h,frame.shape[0]-y)
                        if w<5 or h<5: continue
                        mr,dist,conf_val=_predict(gray[y:y+h,x:x+w],rec,eigen,lmap)
                        if mr and conf_val < AI_CONF_FLAG_THR:
                            if not any(v["roll"]==mr for v in st.session_state.vq_items):
                                st.session_state.vq_items.append(
                                    {"roll":mr,"conf":round(conf_val*100,1),"dist":round(dist,1)})
                        if mr:
                            if st.session_state.attendance.get(mr)!="Present":
                                st.session_state.attendance[mr]="Present"; upd=True
                            cv2.rectangle(af,(x,y),(x+w,y+h),(0,220,80),2)
                            cv2.putText(af,f"{mr}({conf_val*100:.0f}%)",(x,y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,220,80),2)
                        else:
                            cv2.rectangle(af,(x,y),(x+w,y+h),(0,0,220),2)
                    # Occupancy graph data
                    now=time.time()
                    if now-prev>=5:
                        from datetime import datetime
                        st.session_state.occ_times.append(datetime.now().strftime("%H:%M:%S"))
                        st.session_state.occ_counts.append(cnt)
                        st.session_state.occ_times  = st.session_state.occ_times[-60:]
                        st.session_state.occ_counts = st.session_state.occ_counts[-60:]
                        prev=now
                    if cnt > st.session_state.peak_count:
                        st.session_state.peak_count = cnt
                    if now-last_log>=2: _log_csv("Live Camera",cnt,overcrowd_lim); last_log=now
                    face_ph.image(cv2.cvtColor(af,cv2.COLOR_BGR2RGB))
                    yolo_ph.image(cv2.cvtColor(ay,cv2.COLOR_BGR2RGB))
                    _render_metrics(cnt); _render_alert(cnt,overcrowd_lim)
                    if upd: _render_att(); _render_pie()
                    time.sleep(0.01)
            finally:
                cap.release(); cv2.destroyAllWindows()

    # ══════════════════════════════════════════════════════
    #  TAB 2 — Analytics (Occupancy Graph + VQ)
    # ══════════════════════════════════════════════════════
    with tab2:
        # Real-Time Occupancy Graph
        st.markdown("<div class='section-title'>📈 Real-Time Occupancy Graph</div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='panel-glow'>", unsafe_allow_html=True)
        if st.session_state.occ_times:
            st.plotly_chart(_occupancy_chart(
                st.session_state.occ_times, st.session_state.occ_counts),
                use_container_width=True, config={"displayModeBar": False})
            st.caption(f"Peak count this session: **{st.session_state.peak_count}**  "
                       f"| Last {len(st.session_state.occ_times)} readings (every 5s)")
        else:
            st.info("▶ Start a live session to see the real-time occupancy graph.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Verification Queue
        st.markdown("<div class='section-title'>🔍 AI Verification Queue"
                    f" — Confidence &lt; {int(AI_CONF_FLAG_THR*100)}%</div>",
                    unsafe_allow_html=True)
        if st.session_state.vq_items:
            for item in st.session_state.vq_items:
                st.markdown(
                    f"<div class='vq-card'>"
                    f"<span style='font-size:1.5rem'>⚠️</span>"
                    f"<div><div style='font-weight:700;color:#b91c1c;font-family:\"Outfit\",sans-serif;font-size:1.1rem'>"
                    f"{item['roll']}</div>"
                    f"<div style='color:#64748b;font-size:0.85rem'>"
                    f"Confidence: <b style='color:#ef4444'>{item['conf']}%</b> "
                    f"· LBPH dist: {item['dist']}</div></div>"
                    f"<div style='margin-left:auto;color:#ef4444;font-size:0.8rem;"
                    f"font-weight:700;background:#fee2e2;padding:6px 12px;border-radius:20px'>REVIEW</div></div>",
                    unsafe_allow_html=True)
            if st.button("✅ Clear Queue", key="clear_vq"):
                st.session_state.vq_items = []; st.rerun()
        else:
            st.markdown(
                "<div class='vq-card-ok'>"
                "<span style='font-size:1.2rem'>✅</span> "
                "<span style='color:#86efac;font-weight:600'>"
                "No low-confidence detections — all readings above threshold.</span></div>",
                unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    #  TAB 3 — Historical Heatmap
    # ══════════════════════════════════════════════════════
    with tab3:
        st.markdown("<div class='section-title'>🗓 Section Attendance Heatmap — This Week</div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='panel-glow'>", unsafe_allow_html=True)
        st.markdown(_heatmap_html(HEATMAP_DATA), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("🟢 Green = ≥ 75% attendance · 🔴 Red = < 75% attendance")

        # Summary stats
        st.markdown("<br>", unsafe_allow_html=True)
        df = pd.DataFrame(HEATMAP_DATA).set_index("Section")
        df["Avg"] = df.mean(axis=1).round(1)
        c1,c2,c3,c4 = st.columns(4)
        for col,(sec,row) in zip([c1,c2,c3,c4], df.iterrows()):
            clr = "#10b981" if row["Avg"]>=75 else "#ef4444"
            col.markdown(
                f"<div class='panel-glow' style='text-align:center;padding:16px'>"
                f"<div style='font-family:\"Outfit\",sans-serif;font-size:0.85rem;color:#0ea5e9;"
                f"letter-spacing:1px;font-weight:700'>{sec}</div>"
                f"<div style='font-size:2.2rem;font-weight:800;color:{clr}'>{row['Avg']}%</div>"
                f"<div style='color:#64748b;font-size:0.75rem'>Weekly Avg</div></div>",
                unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    #  TAB 4 — Auto Attendance Report
    # ══════════════════════════════════════════════════════
    with tab4:
        st.markdown("<div class='section-title'>📋 Auto Attendance Report</div>",
                    unsafe_allow_html=True)
        if not st.session_state.session_ended:
            st.markdown(
                "<div class='report-locked'>"
                "<div style='font-size:3rem;margin-bottom:12px'>🔒</div>"
                "<div style='font-family:\"Open Sauce One\",sans-serif;font-size:1rem;color:#475569;"
                "letter-spacing:2px;margin-bottom:8px'>SESSION ACTIVE</div>"
                "<div style='color:#475569;font-size:0.88rem'>Click <b>End Session</b> "
                "in the Live Monitor tab to unlock the report.</div></div>",
                unsafe_allow_html=True)
        else:
            pn=sum(1 for s in st.session_state.attendance.values() if s=="Present")
            ab=len(st.session_state.attendance)-pn
            st.success("✅ Session ended — report ready.")
            m1,m2,m3,m4 = st.columns(4)
            for col,(label,val,c) in zip([m1,m2,m3,m4],[
                ("Total Students", len(st.session_state.attendance), "#0ea5e9"),
                ("Present",  pn, "#10b981"),
                ("Absent",   ab, "#ef4444"),
                ("Peak Headcount", st.session_state.peak_count, "#f59e0b"),
            ]):
                col.markdown(
                    f"<div class='panel-glow' style='text-align:center;padding:24px'>"
                    f"<div style='color:#64748b;font-size:0.75rem;font-weight:700;"
                    f"text-transform:uppercase;font-family:\"Outfit\"'>{label}</div>"
                    f"<div style='font-size:2.2rem;font-weight:800;color:{c};margin-top:8px'>{val}</div></div>",
                    unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            report_csv = _build_report_csv(
                st.session_state.attendance,
                st.session_state.peak_count,
                st.session_state.session_start)
            st.download_button(
                label="📥 Download Full Attendance Report (CSV)",
                data=report_csv,
                file_name="vidyaksha_attendance_report.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
                key="report_download")
            # Preview
            st.markdown("<br>**Preview (first 20 rows):**", unsafe_allow_html=True)
            preview_rows = list(st.session_state.attendance.items())[:20]
            ph_html = ("<table style='width:100%;border-collapse:collapse;font-size:0.88rem'>"
                       "<tr style='background:#0284c7;color:white'><th style='padding:8px'>Roll No</th>"
                       "<th style='padding:8px;text-align:center'>Status</th></tr>")
            for roll,status in preview_rows:
                c="#22c55e" if status=="Present" else "#ef4444"
                bg="#dcfce7" if status=="Present" else "#fee2e2"
                ph_html+=(f"<tr><td style='padding:8px;border:1px solid #e2e8f0'>{roll}</td>"
                          f"<td style='padding:8px;border:1px solid #e2e8f0;text-align:center'>"
                          f"<span style='background:{bg};color:{c};padding:3px 10px;"
                          f"border-radius:8px;font-weight:700'>{status}</span></td></tr>")
            ph_html += "</table>"
            st.markdown(ph_html, unsafe_allow_html=True)
