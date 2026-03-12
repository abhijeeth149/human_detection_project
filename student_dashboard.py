"""
student_dashboard.py — Student Dashboard (Premium Light Edition)
Features: Today's Schedule, Subject Analytics, 75% Calculator, Risk Alerts & History.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from styles import STUDENT_DASHBOARD_CSS


# ── Mock Data Generation ────────────────────────────────────────────────────────
def _generate_mock_data(user_id):
    """Generate realistic-looking mock data for the student."""
    subjects = [
        {"name": "Artificial Intelligence", "conducted": 42, "attended": 38, "color": "#0ea5e9"}, # sky
        {"name": "Data Structures", "conducted": 40, "attended": 28, "color": "#f43f5e"},       # rose
        {"name": "Computer Networks", "conducted": 38, "attended": 30, "color": "#10b981"},     # emerald
        {"name": "Operating Systems", "conducted": 45, "attended": 36, "color": "#6366f1"},     # indigo
        {"name": "Database Management", "conducted": 35, "attended": 29, "color": "#f59e0b"}      # amber
    ]
    
    # Calculate percentages
    for sub in subjects:
        sub["percentage"] = round((sub["attended"] / sub["conducted"]) * 100, 1)
        sub["status"] = "SAFE" if sub["percentage"] >= 75.0 else "AT RISK"
        
    # Generate mock history
    history = []
    base_date = datetime.now() - timedelta(days=14)
    for i in range(14):
        date = base_date + timedelta(days=i)
        if date.weekday() < 5:  # form mon-fri sequence
            history.append({"Date": date.strftime("%d %b %Y"), "Subject": "Data Structures", "Time": "09:00 AM", "Status": "Absent" if i % 4 == 0 else "Present"})
            history.append({"Date": date.strftime("%d %b %Y"), "Subject": "Artificial Intelligence", "Time": "11:00 AM", "Status": "Present"})
            
    history.reverse()

    # Generate mock daily schedule
    schedule = [
        {"time": "09:00 AM", "title": "Data Structures Lecture", "status": "past", "room": "Room 302"},
        {"time": "11:00 AM", "title": "Artificial Intelligence Lab", "status": "past", "room": "Lab 4"},
        {"time": "02:00 PM", "title": "Database Management", "status": "future", "room": "Room 105"},
        {"time": "04:00 PM", "title": "Mentoring Session", "status": "future", "room": "Faculty Cabin"},
    ]
    
    return subjects, history, schedule


# ── Render Charts ─────────────────────────────────────────────────────────────
def _render_attendance_chart(subjects):
    fig = go.Figure()
    
    sorted_subs = sorted(subjects, key=lambda x: x["percentage"], reverse=True)
    names = [sub["name"] for sub in sorted_subs]
    percentages = [sub["percentage"] for sub in sorted_subs]
    colors = [sub["color"] for sub in sorted_subs]
    
    fig.add_trace(go.Bar(
        x=names,
        y=percentages,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"<b>{p}%</b>" for p in percentages],
        textposition='outside',
        textfont=dict(color='#334155', size=13, family="Outfit"),
        hoverinfo='y'
    ))
    
    # Add 75% threshold line
    fig.add_hline(y=75, line_dash="dash", line_color="#ef4444", line_width=2,
                  annotation_text="75% Mandate", annotation_position="top right", 
                  annotation_font=dict(color="#ef4444", family="Outfit", size=12))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        yaxis=dict(
            range=[0, 115], 
            gridcolor="rgba(0,0,0,0.05)", 
            zerolinecolor="rgba(0,0,0,0.1)",
            tickfont=dict(color="#64748b", family="Outfit")
        ),
        xaxis=dict(
            tickangle=-15, 
            tickfont=dict(color="#475569", family="Outfit", size=11),
            gridcolor="rgba(0,0,0,0)"
        ),
        showlegend=False
    )
    return fig


# ── Main Entry ────────────────────────────────────────────────────────────────
def show(user_id: str):
    """Render the Student Dashboard Portal."""
    st.markdown(STUDENT_DASHBOARD_CSS, unsafe_allow_html=True)
    
    subjects, history, schedule = _generate_mock_data(user_id)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align:center; margin-bottom:24px; margin-top:16px;'>
            <div style='background:#f1f5f9; width:80px; height:80px; border-radius:50%; margin:0 auto 12px auto; display:flex; align-items:center; justify-content:center; font-size:2.5rem; box-shadow:inset 0 2px 4px rgba(0,0,0,0.05);'>
                👨‍🎓
            </div>
            <div style='font-family:"Outfit",sans-serif; font-size:1.2rem; color:#0f172a; font-weight:800;'>
                Hello, Student!
            </div>
            <div style='color:#64748b; font-size:0.85rem; margin-top:4px;'>
                Reg. No: <strong style='color:#334155;'>{user_id}</strong>
            </div>
        </div>
        <hr style='border-color: rgba(0,0,0,0.05); margin:16px 0;'>
        """, unsafe_allow_html=True)
        
        # Calculate overall attendance
        total_conducted = sum(s["conducted"] for s in subjects)
        total_attended = sum(s["attended"] for s in subjects)
        overall_perc = round((total_attended / total_conducted) * 100, 1) if total_conducted > 0 else 0
        stat_color = "#10b981" if overall_perc >= 75 else "#ef4444"
        
        st.markdown(f"""
        <div style='background:#ffffff; border:1px solid rgba(0,0,0,0.05); border-radius:16px; padding:20px; text-align:center; margin-bottom:24px; box-shadow:0 4px 6px rgba(0,0,0,0.02);'>
            <div style='color:#64748b; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;'>Overall Attendance</div>
            <div style='color:{stat_color}; font-size:2.5rem; font-weight:800; font-family:"Outfit"; line-height:1;'>{overall_perc}%</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚪 LOGOUT", key="student_logout"):
            for key in ["is_logged_in", "user_role", "user_id"]:
                st.session_state.pop(key, None)
            st.rerun()

    # ── Header ──
    st.markdown("<div class='student-header'>VIDYAKSHA</div>", unsafe_allow_html=True)
    st.markdown("<div class='student-sub'>Student Experience Portal</div>", unsafe_allow_html=True)

    # ── Tabs Navigation ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Overview", 
        "📊 Analytics", 
        "🧮 Predictor", 
        "🔔 Alerts & History"
    ])

    # ── TAB 1: Overview ──
    with tab1:
        st.markdown("<div class='glass-card' style='margin-bottom:24px;'>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-family:"Outfit"; font-size:1.3rem; font-weight:700; color:#0f172a; margin-bottom:4px;'>
            Today's Schedule
        </div>
        <div style='color:#64748b; font-size:0.9rem; margin-bottom:24px;'>
            Your timeline for today based on real-time tracking.
        </div>
        """, unsafe_allow_html=True)
        
        for item in schedule:
            dot_class = "past" if item["status"] == "past" else "future"
            st.markdown(f"""
            <div class='timeline-item'>
                <div class='timeline-dot {dot_class}'></div>
                <div class='timeline-content'>
                    <div class='timeline-time'>{item["time"]}</div>
                    <div class='timeline-title'>{item["title"]}</div>
                    <div style='color:#64748b; font-size:0.85rem; margin-top:8px;'>📍 {item["room"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 2: Analytics ──
    with tab2:
        st.markdown("<div class='glass-card' style='margin-bottom:24px;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:\"Outfit\"; font-size:1.3rem; font-weight:700; color:#0f172a; margin-bottom:16px;'>Subject-wise Breakup</div>", unsafe_allow_html=True)
        
        st.plotly_chart(_render_attendance_chart(subjects), use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Subject Metric Mini Cards
        cols = st.columns(len(subjects))
        for i, sub in enumerate(subjects):
            with cols[i]:
                color = sub["color"]
                st.markdown(f"""
                <div class='metric-mini'>
                    <div style='color:#475569; font-size:0.75rem; font-family:"Outfit"; font-weight:700; height:36px; display:flex; align-items:center; justify-content:center;'>{sub["name"]}</div>
                    <div style='font-size:1.8rem; font-weight:800; color:{color}; margin:8px 0;'>{sub["percentage"]}%</div>
                    <div style='color:#94a3b8; font-size:0.75rem; font-weight:500;'>{sub["attended"]} / {sub["conducted"]} Classes</div>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 3: Calculator ──
    with tab3:
        st.markdown("""
        <div style='text-align:center; margin-bottom:32px;'>
            <div style='font-family:"Outfit"; font-size:1.8rem; font-weight:800; color:#0f172a;'>75% Mandate Predictor</div>
            <div style='color:#64748b; font-size:1rem;'>Calculate exactly how many classes you can miss or need to attend.</div>
        </div>
        <style>
            /* Style the columns directly to act as glass cards */
            [data-testid="column"] {
                background: rgba(255, 255, 255, 0.7);
                border: 1px solid rgba(255, 255, 255, 0.5);
                border-radius: 24px;
                padding: 32px !important;
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                box-shadow: 0 10px 30px -10px rgba(0,0,0,0.05), 0 4px 6px -4px rgba(0,0,0,0.02), inset 0 1px 0 rgba(255,255,255,0.8);
            }
        </style>
        """, unsafe_allow_html=True)
        
        calc_col1, calc_col2 = st.columns(2)
        with calc_col1:
            st.markdown("<div style='font-family:\"Outfit\"; font-weight:700; color:#0f172a; margin-bottom:16px; font-size:1.1rem;'>Input Parameters</div>", unsafe_allow_html=True)
            
            # Select subject to auto-fill
            selected_sub_name = st.selectbox("Select Subject (Auto-fill)", ["Custom"] + [s["name"] for s in subjects])
            if selected_sub_name != "Custom":
                sub_data = next((s for s in subjects if s["name"] == selected_sub_name), None)
                def_cond = sub_data["conducted"]
                def_att = sub_data["attended"]
            else:
                def_cond, def_att = 40, 30
                
            classes_conducted = st.number_input("Total Classes Conducted", min_value=1, value=def_cond, step=1)
            classes_attended = st.number_input("Total Classes Attended", min_value=0, max_value=classes_conducted, value=def_att, step=1)
            
        with calc_col2:
            st.markdown("<div style='display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; height:100%;'>", unsafe_allow_html=True)
            curr_perc = (classes_attended / classes_conducted) * 100
            st.markdown("<div style='color:#64748b; font-weight:700; letter-spacing:1px; text-transform:uppercase; font-size:0.85rem; margin-bottom:8px;'>Current Status</div>", unsafe_allow_html=True)
            color = "#10b981" if curr_perc >= 75 else "#ef4444"
            st.markdown(f"<div style='font-size:4rem; font-weight:800; color:{color}; font-family:\"Outfit\"; line-height:1; margin-bottom:24px;'>{curr_perc:.1f}%</div>", unsafe_allow_html=True)
            
            if curr_perc >= 75:
                # Can bunk logic
                allowed_absences = int((classes_attended / 0.75) - classes_conducted)
                if allowed_absences > 0:
                    st.markdown(f"<div class='pill-safe' style='font-size:1rem; padding:12px 24px; display:inline-block;'>✅ You can safely miss <b>{allowed_absences}</b> more classes</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='pill-warning' style='font-size:1rem; padding:12px 24px; display:inline-block;'>⚠️ You are on the edge. Missing the next class is risky.</div>", unsafe_allow_html=True)
            else:
                # Need to attend logic
                needed = int(round((0.75 * classes_conducted - classes_attended) / 0.25))
                if needed <= 0: needed = 1
                st.markdown(f"<div class='pill-danger' style='font-size:1rem; padding:12px 24px; display:inline-block;'>🚨 You must attend the next <b>{needed}</b> classes</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 4: Alerts & History ──
    with tab4:
        st.markdown("<div class='glass-card' style='margin-bottom:24px;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:\"Outfit\"; font-size:1.3rem; font-weight:700; color:#0f172a; margin-bottom:16px;'>Active Risk Alerts</div>", unsafe_allow_html=True)
        
        at_risk = [s for s in subjects if s["percentage"] < 75.0]
        if not at_risk:
            st.markdown("""
            <div style='background:rgba(16, 185, 129, 0.05); border:1px solid rgba(16, 185, 129, 0.2); border-radius:16px; padding:24px; text-align:center;'>
                <div style='font-size:2.5rem; margin-bottom:8px;'>🎉</div>
                <div style='font-family:"Outfit"; font-size:1.2rem; font-weight:700; color:#059669;'>All Clear!</div>
                <div style='color:#10b981; margin-top:4px;'>You are safe in all subjects. Great job!</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for sub in at_risk:
                needed = int(round((0.75 * sub["conducted"] - sub["attended"]) / 0.25))
                st.markdown(f"""
                <div style='background:rgba(239, 68, 68, 0.05); border:1px solid rgba(239, 68, 68, 0.2); border-radius:16px; padding:20px; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                        <div style='display:flex; align-items:center; gap:8px; margin-bottom:4px;'>
                            <span style='font-size:1.2rem'>⚠️</span>
                            <span style='font-family:"Outfit"; font-size:1.1rem; font-weight:700; color:#b91c1c;'>{sub["name"]}</span>
                        </div>
                        <div style='color:#ef4444; font-size:0.9rem;'>
                            Attendance: <b>{sub["percentage"]}%</b> ({sub["attended"]}/{sub["conducted"]})
                        </div>
                    </div>
                    <div style='background:#fef2f2; border:1px solid #fecaca; color:#b91c1c; padding:8px 16px; border-radius:12px; font-weight:600; font-size:0.9rem;'>
                        Attend next {needed}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:\"Outfit\"; font-size:1.3rem; font-weight:700; color:#0f172a; margin-bottom:16px;'>Recent Session Log</div>", unsafe_allow_html=True)
        
        table_html = """
        <div style='overflow-x:auto;'>
            <table class='premium-table'>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Subject</th>
                        <th>Time</th>
                        <th style='text-align:right;'>Status</th>
                    </tr>
                </thead>
                <tbody>
        """
        for row in history[:10]: # take top 10 for clean look
            pill_class = "pill-safe" if row["Status"] == "Present" else "pill-danger"
            table_html += f"""
            <tr>
                <td>{row["Date"]}</td>
                <td>{row["Subject"]}</td>
                <td>{row["Time"]}</td>
                <td style='text-align:right;'><span class='{pill_class}'>{row["Status"]}</span></td>
            </tr>
            """
        table_html += "</tbody></table></div>"
        
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
