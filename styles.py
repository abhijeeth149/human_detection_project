"""
styles.py — Vidyaksha UI: Premium Hackathon Edition.
Features: Ultra-modern glassmorphism, animated mesh gradients, Outfit & Inter typography.
"""

LOGIN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

h1, h2, h3, h4, h5, h6, .login-title, .student-header, .faculty-header {
    font-family: 'Outfit', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
header    { visibility: hidden; }
footer    { visibility: hidden; }

/* ── Background ── */
.stApp {
    background: #0f172a;
    background-image: 
        radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
        radial-gradient(at 50% 0%, hsla(225,39%,30%,0.2) 0, transparent 50%), 
        radial-gradient(at 100% 0%, hsla(339,49%,30%,0.2) 0, transparent 50%);
    background-attachment: fixed;
    color: #e2e8f0;
    min-height: 100vh;
}

/* ── Login card container ── */
.login-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 80vh;
}
.login-card {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 28px;
    padding: 56px 48px;
    max-width: 480px;
    width: 100%;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    box-shadow:
        0 4px 6px -1px rgba(0, 0, 0, 0.1),
        0 24px 48px -12px rgba(0, 0, 0, 0.5),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
    transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
.login-card:hover {
    transform: translateY(-4px);
}

/* ── Logo / Title ── */
.login-title {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
    letter-spacing: -0.02em;
}
.login-subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 0.95rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 40px;
}

/* ── Tab / Role Toggle ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15, 23, 42, 0.6);
    border-radius: 16px;
    padding: 6px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 12px;
    color: #64748b !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 12px 24px;
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background: rgba(56, 189, 248, 0.1) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"]    { display: none; }

/* ── Form inputs ── */
.stTextInput > div > div > input {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important;
    color: #f8fafc !important;
    padding: 16px 20px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
}
.stTextInput > div > div > input:focus {
    border-color: #38bdf8 !important;
    background: rgba(15, 23, 42, 0.8) !important;
    box-shadow: 0 0 0 4px rgba(56, 189, 248, 0.15) !important;
    outline: none !important;
}
.stTextInput > label {
    color: #cbd5e1 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    margin-bottom: 6px !important;
}

/* ── Login Button ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%) !important;
    color: #ffffff !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 16px 0 !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 10px 25px -5px rgba(56, 189, 248, 0.4), 0 8px 10px -6px rgba(56, 189, 248, 0.1) !important;
    margin-top: 16px !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 20px 25px -5px rgba(56, 189, 248, 0.5), 0 8px 10px -6px rgba(56, 189, 248, 0.1) !important;
}
.stButton > button:active {
    transform: translateY(1px) !important;
}

/* ── Error/info messages ── */
.stAlert {
    border-radius: 14px !important;
    font-family: 'Inter', sans-serif !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px);
}

/* ── Cyber grid decoration ── */
.cyber-grid {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
    mask-image: radial-gradient(circle at center, black 40%, transparent 100%);
    -webkit-mask-image: radial-gradient(circle at center, black 40%, transparent 100%);
}
</style>
"""

FACULTY_DASHBOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
h1, h2, h3, h4, h5, h6 { font-family: 'Outfit', sans-serif !important; font-weight: 700 !important; }

/* Light Dashboard Background with subtle modern aesthetic */
.stApp {
    background: #f8fafc;
    background-image: 
        radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.08) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(129, 140, 248, 0.08) 0px, transparent 50%);
    background-attachment: fixed;
    color: #0f172a;
}

#MainMenu { visibility: hidden; }
header    { visibility: hidden; }
footer    { visibility: hidden; }

/* Faculty header banner */
.faculty-header {
    font-family: 'Outfit', sans-serif !important;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #0ea5e9, #4f46e5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
    margin-bottom: 2px;
}

/* Sidebar */
[data-testid="stSidebar"] { 
    background: #ffffff !important; 
    border-right: 1px solid #e2e8f0; 
    box-shadow: 4px 0 24px rgba(0,0,0,0.02);
}

/* Sidebar button */
[data-testid="stSidebar"] .stButton > button {
    background: #fee2e2 !important;
    color: #ef4444 !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    border-radius: 12px !important;
    border: 1px solid #fca5a5 !important;
    padding: 12px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #ef4444 !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.2) !important;
    transform: translateY(-1px) !important;
}

/* Main content buttons */
.main-content .stButton > button, .stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-family: 'Outfit', sans-serif !important;
    transition: all 0.3s ease !important;
}

.stButton > button[kind="primary"] { 
    background: linear-gradient(135deg, #0ea5e9 0%, #4f46e5 100%) !important; 
    color: #ffffff !important; 
    border: none !important;
    box-shadow: 0 4px 12px rgba(14, 165, 233, 0.25) !important; 
}
.stButton > button[kind="primary"]:hover { 
    transform: translateY(-2px); 
    box-shadow: 0 8px 20px rgba(14, 165, 233, 0.4) !important; 
}

/* Base Cards */
.metric-card, .status-banner, .grid-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
    transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease;
}

/* Status banners */
.status-banner { padding: 20px 24px; text-align: center; font-weight: 700; font-size: 1.2rem; margin-top: 16px; font-family: 'Outfit'; }
.status-safe    { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; box-shadow: 0 4px 12px rgba(22,101,52,0.1); }
.status-warning { background: #fffbeb; color: #b45309; border: 1px solid #fde68a; box-shadow: 0 4px 12px rgba(180,83,9,0.1); }
.status-danger  { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; box-shadow: 0 4px 12px rgba(239,68,68,0.1); }

/* Custom Sliders */
.stSlider > div > div > div > div { background-color: #0ea5e9 !important; }
.stRadio [role="radiogroup"] > label > div:first-of-type { background-color: #0ea5e9 !important; border-color: #0ea5e9 !important; }

/* Custom Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border-radius: 12px;
    padding: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    border: 1px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600;
    padding: 10px 20px;
    border-radius: 8px;
    color: #64748b !important;
}
.stTabs [aria-selected="true"] {
    background: #f1f5f9 !important;
    color: #0ea5e9 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
</style>
"""

FACULTY_ENHANCED_CSS = """
<style>
/* ── Panel glow cards (Premium Glass/Card variant for Light Mode) ── */
.panel-glow {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 24px; padding: 24px;
    box-shadow: 0 10px 25px -5px rgba(0,0,0,0.05), 0 8px 10px -6px rgba(0,0,0,0.01);
    transition: box-shadow 0.3s ease, transform 0.3s ease;
}
.panel-glow:hover { 
    box-shadow: 0 20px 25px -5px rgba(0,0,0,0.08), 0 8px 10px -6px rgba(0,0,0,0.02); 
    transform: translateY(-2px);
}

/* ── Verification queue ── */
.vq-card {
    background: #fcf8f8;
    border: 1px solid #fecaca;
    border-radius: 16px; padding: 16px 20px; margin-bottom: 12px;
    display: flex; align-items: center; gap: 16px;
    transition: all 0.2s ease;
}
.vq-card:hover { transform: translateX(4px); border-color: #f87171; }
.vq-card-ok {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 16px; padding: 16px 20px; margin-bottom: 12px;
    color: #166534;
}

/* ── Heatmap cells ── */
.hm-green { background: #dcfce7; color: #166534; border-radius: 8px; padding: 10px 6px; text-align: center; font-weight: 700; font-size: 0.9rem; }
.hm-red   { background: #fee2e2; color: #b91c1c; border-radius: 8px; padding: 10px 6px; text-align: center; font-weight: 700; font-size: 0.9rem; }
.hm-label { color: #475569; font-size: 0.85rem; font-weight: 600; text-align: center; padding: 8px; font-family: 'Outfit'; }

/* ── Report card ── */
.report-locked {
    background: #f8fafc; border: 2px dashed #cbd5e1;
    border-radius: 20px; padding: 48px; text-align: center; color: #64748b;
}

/* Category Titles */
.section-title {
    font-family: 'Outfit', sans-serif; font-size: 1.1rem; color: #0f172a;
    font-weight: 700; margin-bottom: 20px;
    display: flex; align-items: center; gap: 8px;
}

/* Sidebar Stats */
.sidebar-stat {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 12px 16px; margin-bottom: 10px;
}
.sidebar-stat-label { color: #64748b; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }
.sidebar-stat-value { color: #0ea5e9; font-size: 1.6rem; font-weight: 800; font-family: 'Outfit', sans-serif; }
</style>
"""

STUDENT_DASHBOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── Premium Light Mode Dashboard Background ── */
.stApp {
    background: #f8fafc;
    background-image: 
        radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.12) 0px, transparent 40%),
        radial-gradient(at 100% 0%, rgba(129, 140, 248, 0.12) 0px, transparent 40%),
        radial-gradient(at 50% 100%, rgba(244, 63, 94, 0.05) 0px, transparent 50%);
    background-attachment: fixed;
    color: #0f172a;
    min-height: 100vh;
}

#MainMenu { visibility: hidden; }
header    { visibility: hidden; }
footer    { visibility: hidden; }

/* ── Typography & Headers ── */
.student-header {
    font-family: 'Outfit', sans-serif !important;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
    margin-bottom: 2px;
}
.student-sub {
    color: #64748b;
    font-size: 1rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 32px;
    font-family: 'Outfit', sans-serif;
}

/* ── Sidebar Styling ── */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.7) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border-right: 1px solid rgba(0, 0, 0, 0.05) !important;
}

/* Sidebar Logout Button */
[data-testid="stSidebar"] .stButton > button {
    background: rgba(239, 68, 68, 0.05) !important;
    color: #ef4444 !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    width: 100% !important;
    border-radius: 14px !important;
    border: 1px solid rgba(239, 68, 68, 0.2) !important;
    padding: 12px !important;
    transition: all 0.3s ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #ef4444 !important;
    color: white !important;
    box-shadow: 0 8px 20px rgba(239, 68, 68, 0.25) !important;
    transform: translateY(-2px);
}

/* ── Tabs Styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 6px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02);
    border: 1px solid rgba(0,0,0,0.04);
    gap: 8px;
    margin-bottom: 24px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 12px 24px;
    border-radius: 12px;
    color: #64748b !important;
    background: transparent;
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #0ea5e9 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    border: 1px solid rgba(0,0,0,0.02) !important;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* ── Premium Glass Cards ── */
.glass-card {
    background: rgba(255, 255, 255, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.5);
    border-radius: 24px;
    padding: 32px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 
        0 10px 30px -10px rgba(0,0,0,0.05),
        0 4px 6px -4px rgba(0,0,0,0.02),
        inset 0 1px 0 rgba(255,255,255,0.8);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 
        0 20px 40px -10px rgba(0,0,0,0.08),
        0 8px 12px -4px rgba(0,0,0,0.03),
        inset 0 1px 0 rgba(255,255,255,0.8);
}

/* ── Metric Mini Cards ── */
.metric-mini {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.04);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.02);
    text-align: center;
    transition: all 0.3s ease;
}
.metric-mini:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 24px rgba(0,0,0,0.06);
    border-color: rgba(14, 165, 233, 0.2);
}

/* ── Status Pills & Banners ── */
.pill-safe { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; padding: 6px 14px; border-radius: 50px; font-weight: 600; font-size: 0.85rem; }
.pill-warning { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; padding: 6px 14px; border-radius: 50px; font-weight: 600; font-size: 0.85rem; }
.pill-danger { background: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; padding: 6px 14px; border-radius: 50px; font-weight: 600; font-size: 0.85rem; }

/* ── History Table ── */
.premium-table {
    width: 100%; border-collapse: separate; border-spacing: 0;
}
.premium-table th {
    color: #64748b; font-family: 'Outfit'; font-weight: 600; font-size: 0.85rem; text-transform: uppercase;
    padding: 16px; border-bottom: 2px solid #e2e8f0; text-align: left;
}
.premium-table td {
    padding: 16px; color: #334155; font-size: 0.95rem; font-weight: 500;
    border-bottom: 1px solid #f1f5f9;
}
.premium-table tr:hover td { background: rgba(241, 245, 249, 0.5); }
.premium-table tr:last-child td { border-bottom: none; }

/* ── Number Inputs Override ── */
.stNumberInput > div > div > input {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 12px !important;
    color: #0f172a !important;
    font-weight: 600 !important;
}
.stNumberInput > div > div > input:focus {
    border-color: #0ea5e9 !important;
    box-shadow: 0 0 0 4px rgba(14, 165, 233, 0.1) !important;
}

/* ── Timeline ── */
.timeline-item {
    display: flex; gap: 20px; align-items: flex-start; margin-bottom: 24px; position: relative;
}
.timeline-item::before {
    content: ''; position: absolute; left: 6px; top: 24px; bottom: -24px; width: 2px;
    background: #e2e8f0; z-index: 0;
}
.timeline-item:last-child::before { display: none; }
.timeline-dot {
    width: 14px; height: 14px; border-radius: 50%; background: #0ea5e9;
    box-shadow: 0 0 0 4px #e0f2fe; margin-top: 4px; z-index: 1; position: relative;
}
.timeline-dot.past { background: #10b981; box-shadow: 0 0 0 4px #dcfce7; }
.timeline-dot.future { background: #cbd5e1; box-shadow: 0 0 0 4px #f1f5f9; }
.timeline-content {
    background: #ffffff; border: 1px solid #f1f5f9; border-radius: 16px; padding: 16px 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.02); flex-grow: 1;
}
.timeline-time { color: #64748b; font-size: 0.85rem; font-family: 'Outfit'; font-weight: 600; margin-bottom: 4px; }
.timeline-title { color: #0f172a; font-weight: 700; font-size: 1.05rem; }

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #ffffff !important;
    border-radius: 12px !important;
    border: 1px solid #cbd5e1 !important;
    color: #0f172a !important;
}
</style>
"""
