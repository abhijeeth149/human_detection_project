"""
main.py — Vidyaksha AI Observer: Role-Based Login Entry Point.

Run with:  streamlit run main.py
"""

import streamlit as st
import os
import cv2
import time
import numpy as np
import subprocess
from styles import LOGIN_CSS
from auth   import verify_faculty, verify_student, register_student
import faculty_dashboard
import student_dashboard

# ── Page Config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Vidyaksha — AI Observer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state bootstrap ───────────────────────────────────────────────────
if "is_logged_in" not in st.session_state:
    st.session_state.is_logged_in = False
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "is_registering" not in st.session_state:
    st.session_state.is_registering = False

# ── ROUTER ────────────────────────────────────────────────────────────────────
if st.session_state.is_logged_in:
    if st.session_state.user_role == "faculty":
        faculty_dashboard.show(st.session_state.user_id)
    elif st.session_state.user_role == "student":
        student_dashboard.show(st.session_state.user_id)
    st.stop()   # Nothing below should render when logged in

# ── LOGIN SCREEN ──────────────────────────────────────────────────────────────
st.markdown(LOGIN_CSS, unsafe_allow_html=True)
st.markdown("<div class='cyber-grid'></div>", unsafe_allow_html=True)

# Use a placeholder so the entire login UI is replaced upon successful auth
login_placeholder = st.empty()

with login_placeholder.container():
    # Vertical spacer
    st.markdown("<div style='height:6vh'></div>", unsafe_allow_html=True)

    # Centre the card using columns
    _, card_col, _ = st.columns([1, 1.4, 1])
    with card_col:
        st.markdown("""
        <div class='login-card'>
            <div class='login-title'>VIDYAKSHA</div>
            <div class='login-subtitle'>AI Observer · Secure Portal</div>
        </div>
        """, unsafe_allow_html=True)

        # Role selection tabs
        faculty_tab, student_tab = st.tabs(["🏫  FACULTY", "🎓  STUDENT"])

        # ── Faculty Login ──────────────────────────────────────────────────────
        with faculty_tab:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            emp_id   = st.text_input("Employee ID",   key="faculty_emp_id",
                                     placeholder="e.g. admin")
            fac_pass = st.text_input("Password",      key="faculty_password",
                                     type="password", placeholder="••••••••")

            if st.button("ACCESS COMMAND CENTER", key="faculty_login_btn"):
                if not emp_id or not fac_pass:
                    st.error("⚠️ Please fill in all fields.")
                elif verify_faculty(emp_id, fac_pass):
                    st.session_state.is_logged_in = True
                    st.session_state.user_role    = "faculty"
                    st.session_state.user_id      = emp_id
                    login_placeholder.empty()      # wipe login screen
                    st.rerun()
                else:
                    st.error("❌ Invalid Employee ID or Password.")

            st.markdown("""
            <div style='text-align:center; color:#334155; font-size:0.75rem;
                        margin-top:16px; letter-spacing:1px;'>
                Demo: <code style='color:#00b8d4;'>admin</code> / 
                <code style='color:#00b8d4;'>vidya123</code>
            </div>
            """, unsafe_allow_html=True)

        # ── Student Login ──────────────────────────────────────────────────────
        with student_tab:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            
            if st.session_state.is_registering:
                st.markdown("### 📝 Register New Student")
                st.info("Please capture 3 clear photos of your face: Front, Left, and Right.")
                new_reg_no = st.text_input("New Registration Number", placeholder="e.g. std003", key="reg_reg_no")
                new_pass = st.text_input("New Password", type="password", key="reg_pass")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("**1. Front Face**")
                    up1 = st.file_uploader("Upload Front", type=['jpg','png','jpeg'], key="up1")
                with col2:
                    st.write("**2. Look Slightly Left**")
                    up2 = st.file_uploader("Upload Left", type=['jpg','png','jpeg'], key="up2")
                with col3:
                    st.write("**3. Look Slightly Right**")
                    up3 = st.file_uploader("Upload Right", type=['jpg','png','jpeg'], key="up3")
                
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("Cancel", use_container_width=True):
                        st.session_state.is_registering = False
                        st.rerun()
                with btn_col2:
                    if st.button("Complete Registration", use_container_width=True, type="primary"):
                        f1, f2, f3 = up1, up2, up3
                        if not new_reg_no or not new_pass or not f1 or not f2 or not f3:
                            st.error("⚠️ All fields and 3 uploaded photos are required.")
                        else:
                            # 1. Register Credentials
                            if not register_student(new_reg_no, new_pass):
                                st.error(f"❌ Roll number '{new_reg_no}' already exists!")
                            else:
                                # 2. Save Photos
                                student_dir = os.path.join("dataset", "Cls photos", new_reg_no.strip())
                                os.makedirs(student_dir, exist_ok=True)
                                
                                # Helper to save camera/upload image
                                def save_img(img_file, filename):
                                    bytes_data = img_file.getvalue()
                                    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                                    cv2.imwrite(os.path.join(student_dir, filename), cv2_img)
                                
                                save_img(f1, "front.jpg")
                                save_img(f2, "left.jpg")
                                save_img(f3, "right.jpg")
                                
                                st.success("✅ Photos saved! Training Face AI... please wait.")
                                
                                # 3. Trigger Retraining
                                try:
                                    # Call the training script as a subprocess synchronously so user waits
                                    subprocess.run(["python", "train_from_cls_photos.py"], check=True)
                                    st.session_state.is_registering = False
                                    st.success("🎉 Registration and AI Training Complete! You can now log in.")
                                    time.sleep(2)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Training Failed: {e}")

            else:
                reg_no   = st.text_input("Registration Number", key="student_reg_no",
                                         placeholder="e.g. std001")
                std_pass = st.text_input("Password",            key="student_password",
                                         type="password",       placeholder="••••••••")

                if st.button("ACCESS STUDENT PORTAL", key="student_login_btn", type="primary"):
                    if not reg_no or not std_pass:
                        st.error("⚠️ Please fill in all fields.")
                    elif verify_student(reg_no, std_pass):
                        st.session_state.is_logged_in = True
                        st.session_state.user_role    = "student"
                        st.session_state.user_id      = reg_no
                        login_placeholder.empty()      # wipe login screen
                        st.rerun()
                    else:
                        st.error("❌ Invalid Registration Number or Password.")
                        
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("➕ Register New Student", key="register_student_btn", use_container_width=True):
                    st.session_state.is_registering = True
                    st.rerun()

                st.markdown("""
                <div style='text-align:center; color:#334155; font-size:0.75rem;
                            margin-top:16px; letter-spacing:1px;'>
                    Demo: <code style='color:#00b8d4;'>std001</code> / 
                    <code style='color:#00b8d4;'>pass123</code>
                </div>
                """, unsafe_allow_html=True)

    # Bottom tagline
    st.markdown("""
    <div style='text-align:center; color:#1e293b; font-size:0.72rem;
                letter-spacing:2px; margin-top:48px; text-transform:uppercase;'>
        Powered by Vidyaksha &nbsp;·&nbsp; Vidyaksha Observer System
    </div>
    """, unsafe_allow_html=True)
