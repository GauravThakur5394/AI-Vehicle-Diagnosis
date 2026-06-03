import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import json
import streamlit.components.v1 as components

# --- PAGE CONFIGURATION & CUSTOM CSS ---
st.set_page_config(page_title="AI Vehicle Diagnostic Assistant", layout="wide", page_icon="🚗")

# --- THEME TOGGLE (Dark/Light Mode) ---
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"

with st.sidebar:
    st.markdown("---")
    st.write("**🎨 Theme Settings**")
    col_theme1, col_theme2 = st.columns(2)
    with col_theme1:
        if st.button("🌙 Dark", use_container_width=True, key="dark_btn"):
            st.session_state.theme_mode = "dark"
            st.rerun()
    with col_theme2:
        if st.button("☀️ Light", use_container_width=True, key="light_btn"):
            st.session_state.theme_mode = "light"
            st.rerun()
    st.markdown("---")

# --- DYNAMIC CSS BASED ON THEME ---
if st.session_state.theme_mode == "dark":
    THEME_CSS = """
    .stApp {
        background-image: linear-gradient(rgba(10, 14, 23, 0.85), rgba(10, 14, 23, 0.85)), url('https://images.unsplash.com/photo-1581092160562-40aa08e78837?q=80&w=2070');
        background-size: cover;
        background-attachment: fixed;
        color: #e0e0e0;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(15, 20, 30, 0.95) !important;
    }
    .stRadio > div > label > div {font-size: 24px !important; font-weight: bold; margin-bottom: 10px; color: #00ffff;}
    div.stButton > button:first-child {
        background-color: transparent;
        color: #00ffff;
        height: 60px;
        width: 100%;
        border-radius: 8px;
        font-size: 22px;
        font-weight: bold;
        border: 2px solid #00ffff;
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.2);
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: rgba(0, 255, 255, 0.1);
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.6);
    }
    @keyframes pulse-critical {
        0% { box-shadow: 0 0 0 0 rgba(255, 50, 50, 0.7); }
        70% { box-shadow: 0 0 20px 10px rgba(255, 50, 50, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 50, 50, 0); }
    }
    .severity-critical {
        border-left: 10px solid #ff3232 !important;
        animation: pulse-critical 2s infinite;
    }
    .severity-moderate {
        border-left: 10px solid #ffcc00 !important;
    }
    .severity-low {
        border-left: 10px solid #00ffcc !important;
    }
    """
else:  # LIGHT MODE
    THEME_CSS = """
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf0 100%);
        color: #333333;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #333333 !important;
    }
    .stRadio > div > label > div {font-size: 24px !important; font-weight: bold; margin-bottom: 10px; color: #0066cc;}
    div.stButton > button:first-child {
        background-color: transparent;
        color: #0066cc;
        height: 60px;
        width: 100%;
        border-radius: 8px;
        font-size: 22px;
        font-weight: bold;
        border: 2px solid #0066cc;
        box-shadow: 0 0 10px rgba(0, 102, 204, 0.2);
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: rgba(0, 102, 204, 0.1);
        box-shadow: 0 0 20px rgba(0, 102, 204, 0.6);
    }
    @keyframes pulse-critical {
        0% { box-shadow: 0 0 0 0 rgba(255, 50, 50, 0.7); }
        70% { box-shadow: 0 0 20px 10px rgba(255, 50, 50, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 50, 50, 0); }
    }
    .severity-critical {
        border-left: 10px solid #ff3232 !important;
        animation: pulse-critical 2s infinite;
    }
    .severity-moderate {
        border-left: 10px solid #ffaa00 !important;
    }
    .severity-low {
        border-left: 10px solid #00aa77 !important;
    }
    h1, h2, h3, h4, h5, h6 { color: #1a1a1a !important; }
    p, div, span, label { color: #333333 !important; }
    
    /* Selectbox and Dropdown Styling */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 2px solid #0066cc !important;
    }
    .stSelectbox [data-baseweb="select"] div {
        color: #333333 !important;
    }
    [data-baseweb="select"] div:first-child {
        color: #333333 !important;
    }
    [role="listbox"] {
        background-color: #ffffff !important;
        color: #333333 !important;
    }
    [role="option"] {
        color: #333333 !important;
        background-color: #ffffff !important;
    }
    [role="option"]:hover {
        background-color: #e0e8f8 !important;
    }
    
    /* Input fields */
    .stTextInput input,
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #333333 !important;
        border: 2px solid #0066cc !important;
    }
    
    /* Placeholder text */
    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #999999 !important;
    }
    """

st.markdown(f"<style>{THEME_CSS}</style>", unsafe_allow_html=True)


# --- API KEY CONFIGURATION (CRASH-PROOF) ---
api_key = os.environ.get("API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets.get("API_KEY") or st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass

if api_key:
    genai.configure(api_key=api_key)
else:
    st.sidebar.error("API Key not found! Please check your settings.")

# --- DATASET LOADING ---
FILE_NAME = "expanded_vehicle_fault_dataset.csv"

@st.cache_data
def load_data():
    try:
        return pd.read_csv(FILE_NAME)
    except FileNotFoundError:
        st.error(f"Dataset {FILE_NAME} not found. Please ensure it is uploaded.")
        return pd.DataFrame()

df = load_data()

# --- UNIFIED UI RENDERER ---
def render_diagnostic_ui(data):
    """Renders the completely styled UI output for both Tabs with Severity-based color coding"""
    if data.get('is_fallback'):
         st.warning("⚠️ **SYSTEM NOTICE: AI API Limit Reached. Currently running in offline database mode.**", icon="⏳")

    sev = str(data.get('severity', 'Moderate')).lower()
    if sev in ["critical", "high"]:
        sev_class = "severity-critical"
    elif sev in ["low", "minor"]:
        sev_class = "severity-low"
    else:
        sev_class = "severity-moderate"

    st.markdown(f'<div class="{sev_class}" style="background-color: rgba(20, 25, 35, 0.9); padding: 30px; border-radius: 12px; margin-top: 20px; border: 1px solid #333;">', unsafe_allow_html=True)

    # 1. SAFETY BOX
    st.markdown(f"""
    <div style="background-color: rgba(255, 50, 50, 0.1); padding: 20px; border-radius: 8px; margin-bottom: 25px;">
        <h2 style="color: #ff4d4d; margin-top: 0; font-size: 28px;">🚨 SAFETY FIRST</h2>
        <p style="color: #ffcccc; font-size: 18px; font-weight: bold; margin-bottom: 0;">{data['safety']}</p>
    </div>
    """, unsafe_allow_html=True)

    # 2. MECHANIC BOX
    mechanic_text = str(data.get('mechanic', '')).strip()
    if mechanic_text and mechanic_text.lower() not in ["no", "none", "n/a", ""]:
        st.markdown(f"""
        <div style="background-color: rgba(255, 204, 0, 0.1); padding: 20px; border-radius: 8px; margin-bottom: 25px;">
            <h2 style="color: #ffcc00; margin-top: 0; font-size: 28px;">🛠️ MECHANIC ADVICE</h2>
            <p style="color: #fff4cc; font-size: 18px; font-weight: bold; margin-bottom: 0;">{mechanic_text}</p>
        </div>
        """, unsafe_allow_html=True)

    # 3. DETAILS
    st.markdown(f"<h3 style='color: #00ffff;'>🔍 Diagnosis:</h3><p style='font-size: 16px; color: #e0e0e0;'>{data['diagnosis']}</p>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: #00ffff;'>🧰 Tools Needed:</h3><p style='font-size: 16px; color: #e0e0e0;'>{data['tools']}</p>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #00ffff;'>📋 Step-by-Step Fix:</h3>", unsafe_allow_html=True)
    
    # Force breaks on punctuation and respect newline formatting with CSS pre-wrap
    steps_cleaned = str(data['steps']).replace(". ", ".\n")
    st.markdown(f"<div style='font-size: 16px; color: #e0e0e0; line-height: 1.8; white-space: pre-wrap;'>{steps_cleaned}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# --- CORE LOGIC FUNCTIONS ---
def get_simplified_fix(row):
    model = genai.GenerativeModel('gemini-3.5-flash')
    prompt = f"""
    Rewrite the provided technical vehicle repair instructions into simple, 5th-grade level English.
    DATASET: Fault: {row['possible_fault']} | Tools: {row['tools_needed']} | Steps: {row['beginner_steps']} | Safety: {row['safety_warning']} | Mechanic: {row['mechanic_advice']}
    
    Return ONLY a valid JSON object exactly like this:
    {{
        "safety": "[Simplified safety warning]",
        "mechanic": "[Simplified mechanic advice]",
        "diagnosis": "[Simplified explanation of the fault]",
        "tools": "[List tools simply]",
        "steps": "[Numbered list of simplified steps]"
    }}
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        data['is_fallback'] = False
        data['severity'] = row.get('severity', 'Moderate')
        return data
    except Exception:
        return {
            "safety": row['safety_warning'],
            "mechanic": row['mechanic_advice'],
            "diagnosis": row['possible_fault'],
            "tools": row['tools_needed'],
            "steps": row['beginner_steps'],
            "severity": row.get('severity', 'Moderate'),
            "is_fallback": True
        }

def get_ai_diagnosis(user_input):
    model = genai.GenerativeModel('gemini-3.5-flash')
    prompt = f"""
    The user reports this vehicle issue: "{user_input}"
    Provide a diagnostic guide in simple, 5th-grade level English.
    
    Return ONLY a valid JSON object exactly like this:
    {{
        "safety": "[Strong safety warning]",
        "mechanic": "[When to call a mechanic]",
        "diagnosis": "[Explanation of the potential fault]",
        "tools": "[List tools simply]",
        "steps": "[Numbered list of diagnostic steps]",
        "category": "[Pick closest: Engine, Cooling System, Fuel System, Exhaust, Transmission, Drivetrain, Brakes, Wheels & Tires, Suspension, Steering, Electrical, Lighting & Signals, Infotainment]",
        "csv_data": {{
            "vehicle_type": "Car", "category": "General", "subcategory": "AI", 
            "symptom": "{user_input[:50]}", "possible_fault": "AI Generated", 
            "severity": "Medium", "user_fixable": "Yes", "tools_needed": "Basic tools", 
            "beginner_steps": "Follow AI steps", "mechanic_advice": "Consult professional", 
            "safety_warning": "Exercise caution"
        }}
    }}
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        
        try:
            pd.DataFrame([data['csv_data']]).to_csv(FILE_NAME, mode='a', header=False, index=False)
        except Exception:
            pass
            
        data['is_fallback'] = False
        data['severity'] = data.get('csv_data', {}).get('severity', 'Medium')
        return data
    except Exception:
        return {
            "safety": "Always park on a flat surface, engage the parking brake, and let the engine cool completely before inspecting.",
            "mechanic": "If you notice smoke, active fluid leaks, or require specialized diagnostic equipment.",
            "diagnosis": f"Potential issue related to: '{user_input}'. Please check your dashboard indicators.",
            "tools": "Standard hand tools, protective gloves, flashlight.",
            "steps": "1. Wait for the free-tier API window to clear.\n2. Visually inspect the area for obvious loose wires or leaks.\n3. Try browsing the 'Predefined Categories' tab for offline solutions.",
            "category": "DEFAULT",
            "severity": "Medium",
            "is_fallback": True
        }


# --- HIGH-PRECISION GEOMETRIC TRANSITION VISUALIZER ---
def render_3d_hologram(category):
    # Highly specific coordinate mapping for every system group inside the Jeep model meshes
    CATEGORY_CAMERAS = {
        "Engine":               {"cam": (-3.5, 2.2, 2.0),   "look": (-1.8, 1.0, 0.0),   "zone": "ENGINE_BAY"},
        "Engine Compartment":     {"cam": (-3.5, 2.2, 2.0),   "look": (-1.8, 1.0, 0.0),   "zone": "ENGINE_BAY"},
        "Engine Components":      {"cam": (-3.5, 2.2, 2.0),   "look": (-1.8, 1.0, 0.0),   "zone": "ENGINE_BAY"},
        "Cooling System":         {"cam": (-4.0, 1.8, 1.5),   "look": (-2.3, 0.9, 0.0),   "zone": "ENGINE_BAY"},
        "Fuel System":            {"cam": (3.0, 1.2, 2.5),    "look": (0.88, 0.3, 0.0),   "zone": "ENGINE_BAY"},
        "Liquid Systems":         {"cam": (-3.8, 1.5, 2.0),   "look": (-1.8, 0.8, 0.2),   "zone": "ENGINE_BAY"},
        "Emissions System":       {"cam": (-3.5, 1.2, -2.0),  "look": (-1.5, 0.6, -0.4),  "zone": "ENGINE_BAY"},
        "Exhaust":                {"cam": (3.5, 0.8, -2.5),   "look": (1.8, 0.3, -0.4),   "zone": "ENGINE_BAY"},
        "Air Conditioning System":{"cam": (-3.6, 2.0, 1.8),   "look": (-2.2, 1.1, 0.5),   "zone": "ENGINE_BAY"},
        "Heating & AC":           {"cam": (-3.6, 2.0, 1.8),   "look": (-2.2, 1.1, 0.5),   "zone": "ENGINE_BAY"},
        
        "Transmission":         {"cam": (-1.5, 0.8, 3.0),   "look": (-0.6, 0.5, 0.0),   "zone": "TRANSMISSION"},
        "Drivetrain":           {"cam": (1.5, 0.6, 3.0),    "look": (0.82, 0.4, 0.0),   "zone": "TRANSMISSION"},
        "Drive Chain":          {"cam": (1.5, 0.6, 3.0),    "look": (0.82, 0.4, 0.0),   "zone": "TRANSMISSION"},
        
        "Brakes":               {"cam": (-3.0, 0.8, 2.8),   "look": (-1.64, 0.46, 1.0), "zone": "FRONT_WHEEL"},
        "ABS System":           {"cam": (-3.0, 0.8, 2.8),   "look": (-1.64, 0.46, 1.0), "zone": "FRONT_WHEEL"},
        "Wheels & Tires":       {"cam": (3.2, 1.0, 3.2),    "look": (1.58, 0.38, 1.0),  "zone": "FRONT_WHEEL"},
        "Tyres":                {"cam": (3.2, 1.0, 3.2),    "look": (1.58, 0.38, 1.0),  "zone": "FRONT_WHEEL"},
        "Suspension":           {"cam": (-2.8, 0.6, -3.0),  "look": (-1.64, 0.3, -0.76),"zone": "FRONT_WHEEL"},
        
        "Steering":             {"cam": (-1.8, 2.0, 1.8),   "look": (-0.78, 1.5, -0.18),"zone": "STEERING"},
        
        "Electrical":           {"cam": (-2.5, 2.2, -2.5),  "look": (-1.62, 1.0, -0.72),"zone": "ELECTRICAL"},
        "Electrical System":    {"cam": (-2.5, 2.2, -2.5),  "look": (-1.62, 1.0, -0.72),"zone": "ELECTRICAL"},
        "Sensors & ADAS":       {"cam": (-2.0, 2.0, 2.5),   "look": (-0.72, 1.3, -0.5), "zone": "ELECTRICAL"},
        "EV & Hybrid":          {"cam": (-2.0, 2.0, 2.5),   "look": (-0.72, 1.3, -0.5), "zone": "ELECTRICAL"},
        
        "Lighting & Signals":   {"cam": (-4.5, 1.5, 0.0),   "look": (-2.58, 0.88, 0.0), "zone": "FRONT_FACE"},
        "Wipers":               {"cam": (-3.2, 1.8, 0.0),   "look": (-1.5, 1.4, 0.0),   "zone": "FRONT_FACE"},
        "Windows":              {"cam": (0.0, 2.2, 3.2),    "look": (0.5, 1.5, 0.0),    "zone": "FRONT_FACE"},
        
        "Infotainment":         {"cam": (0.2, 2.2, 2.5),    "look": (0.0, 1.4, 0.0),    "zone": "CABIN"},
        "DEFAULT":              {"cam": (7.0, 3.5, 7.0),    "look": (0.0, 1.0, 0.0),    "zone": "DEFAULT"}
    }

    if "current_cam" not in st.session_state:
        st.session_state.current_cam = CATEGORY_CAMERAS["DEFAULT"]
    if "prev_cam" not in st.session_state:
        st.session_state.prev_cam = CATEGORY_CAMERAS["DEFAULT"]

    target_cfg = CATEGORY_CAMERAS.get(category, CATEGORY_CAMERAS["DEFAULT"])

    if target_cfg["cam"] != st.session_state.current_cam["cam"] or target_cfg["look"] != st.session_state.current_cam["look"]:
        st.session_state.prev_cam = st.session_state.current_cam
        st.session_state.current_cam = target_cfg

    px, py, pz = st.session_state.prev_cam["cam"]
    plx, ply, plz = st.session_state.prev_cam["look"]
    cx, cy, cz = st.session_state.current_cam["cam"]
    lx, ly, lz = st.session_state.current_cam["look"]
    zone = st.session_state.current_cam["zone"]

    html_code = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#010d0d; overflow:hidden; }}
  #wrap {{ position:relative; width:100%; height:500px; }}
  canvas {{ display:block; width:100%; height:500px; }}
  #lbl {{
    position:absolute; bottom:8px; left:0; right:0;
    text-align:center; color:#00ffcc; font-size:10px;
    font-family:monospace; letter-spacing:2px; opacity:0.75;
    pointer-events:none;
  }}
  #zone {{
    position:absolute; top:8px; right:10px;
    color:#00ffcc; font-size:9px; font-family:monospace;
    letter-spacing:1px; opacity:0.6; pointer-events:none;
  }}
</style>
</head>
<body>
<div id="wrap">
  <canvas id="hc"></canvas>
  <div id="lbl">◈ JEEP 4×4 DIAGNOSTIC HOLOGRAM ◈</div>
  <div id="zone">ZONE: {zone}</div>
</div>
 
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>
<script>
const START_CX = {px}; const START_CY = {py}; const START_CZ = {pz};
const START_LX = {plx}; const START_LY = {ply}; const START_LZ = {plz};

const TARGET_CX = {cx}; const TARGET_CY = {cy}; const TARGET_CZ = {cz};
const TARGET_LX = {lx}; const TARGET_LY = {ly}; const TARGET_LZ = {lz};
const ZONE = "{zone}";
 
const cnv = document.getElementById('hc');
const W = cnv.parentElement.clientWidth, H = 500;
cnv.width  = W * devicePixelRatio;
cnv.height = H * devicePixelRatio;
 
const scene    = new THREE.Scene();
const cam      = new THREE.PerspectiveCamera(40, W/H, 0.05, 400);
const renderer = new THREE.WebGLRenderer({{ canvas:cnv, alpha:true, antialias:true }});
renderer.setSize(W, H);
renderer.setPixelRatio(devicePixelRatio);
renderer.setClearColor(0, 0);
 
const carGroup = new THREE.Group();
scene.add(carGroup);
 
function ln(pts, mat, g) {{
  const geo = new THREE.BufferGeometry().setFromPoints(pts.map(p => new THREE.Vector3(p[0],p[1],p[2])));
  const l = new THREE.Line(geo, mat); (g||carGroup).add(l); return l;
}}
function eg(geo, mat, g) {{
  const e = new THREE.EdgesGeometry(geo); const m = new THREE.LineSegments(e, mat); (g||carGroup).add(m); return m;
}}
function bx(w,h,d,mat,g) {{ return eg(new THREE.BoxGeometry(w,h,d), mat, g); }}
function cy(rt,rb,h,seg,mat,g) {{ return eg(new THREE.CylinderGeometry(rt,rb,h,seg), mat, g); }}
function pos(m,x,y,z) {{ m.position.set(x,y,z); return m; }}
function rot(m,x,y,z) {{ m.rotation.set(x,y,z); return m; }}

function getMat(compZone, baseColor, defaultOpacity = 0.8) {{
  if (ZONE === "DEFAULT" || ZONE === compZone) {{
    return new THREE.LineBasicMaterial({{ color: baseColor, transparent: true, opacity: 1.0 }});
  }}
  return new THREE.LineBasicMaterial({{ color: baseColor, transparent: true, opacity: 0.15 }});
}}
 
const mFrame = getMat("FRAME", 0x00ffcc, 0.22);
for (const z of [-0.38, 0.38]) {{ 
  pos(bx(5.2, 0.08, 0.06, mFrame), 0, 0.26, z); 
}}
pos(bx(0.06, 0.06, 0.76, mFrame), -2.5, 0.26, 0);
pos(bx(0.06, 0.06, 0.76, mFrame), -1.1, 0.26, 0);
pos(bx(0.12, 0.04, 0.76, mFrame), -0.2, 0.22, 0);
pos(bx(0.06, 0.06, 0.76, mFrame),  1.0, 0.26, 0);
pos(bx(0.06, 0.06, 0.76, mFrame),  2.2, 0.26, 0);
 
const mBody = getMat("BODY", 0x00ffcc, 0.6);
pos(bx(4.6, 0.12, 1.88, mBody), 0, 0.36, 0);
for (const z of [-0.96, 0.96]) {{ pos(bx(4.4, 0.18, 0.12, mBody), 0, 0.5, z); }}
for (const z of [-0.92, 0.92]) {{ ln([[-0.88, 0.36, z], [-0.88, 1.72, z]], mBody); }}
ln([[-0.88, 1.72, -0.92], [-0.88, 1.72, 0.92]], mBody);
ln([[-0.88, 0.36, -0.92], [-0.88, 0.36, 0.92]], mBody);

pos(bx(0.15, 0.22, 2.1, mBody), -2.72, 0.36, 0);
for (const z of [-0.96, 0.96]) {{
  ln([[-0.84, 1.28, z], [-0.74, 1.28, z * 1.15], [-0.74, 1.12, z * 1.15], [-0.84, 1.12, z]], mBody);
}}
 
for (const z of [-0.94, 0.94]) {{ ln([[-0.88, 1.12, z], [2.32, 1.12, z]], mBody); }}
for (const z of [-0.94, 0.94]) {{
  ln([[-0.88, 1.12, z], [-0.7, 1.98, z]], mBody); ln([[0.62, 1.12, z], [0.62, 1.98, z]], mBody);
  ln([[2.0, 1.12, z], [2.14, 1.98, z]], mBody); ln([[2.32, 1.12, z], [2.32, 1.98, z]], mBody);
}}
ln([[-0.7, 1.98, -0.94], [2.14, 1.98, -0.94]], mBody); ln([[-0.7, 1.98,  0.94], [2.14, 1.98,  0.94]], mBody);
ln([[-0.7, 1.98, -0.94], [-0.7, 1.98,  0.94]], mBody); ln([[2.14, 1.98, -0.94], [2.14, 1.98,  0.94]], mBody);
ln([[-0.7, 2.0, 0], [2.14, 2.0, 0]], mBody);
 
ln([[-0.88, 1.56, -0.94], [-0.88, 1.56, 0.94]], mBody);
ln([[-0.88, 1.56, -0.94], [-2.58, 1.18, -0.94]], mBody); ln([[-0.88, 1.56,  0.94], [-2.58, 1.18,  0.94]], mBody);
ln([[-2.58, 1.18, -0.94], [-2.58, 1.18, 0.94]], mBody); ln([[-0.88, 1.6, 0], [-2.58, 1.22, 0]], mBody);
for (let i = 0; i < 5; i++) {{ const xv = -1.2 - i * 0.25; ln([[xv, 1.44, -0.4], [xv, 1.44, 0.4]], mBody); }}
 
ln([[-2.58, 1.18, -0.94], [-2.58, 0.46, -0.94]], mBody); ln([[-2.58, 1.18,  0.94], [-2.58, 0.46,  0.94]], mBody);
ln([[-2.58, 0.46, -0.94], [-2.58, 0.46,  0.94]], mBody);
for (let i = 0; i < 5; i++) {{ ln([[-2.585, 0.54 + i * 0.12, -0.7], [-2.585, 0.54 + i * 0.12, 0.7]], mBody); }}
for (const z of [-0.5, -0.17, 0.17, 0.5]) {{ ln([[-2.585, 0.54, z], [-2.585, 1.14, z]], mBody); }}
pos(bx(0.1, 0.22, 1.94, mBody), -2.64, 0.38, 0);
for (const z of [-0.7, 0.7]) {{
  ln([[-2.64, 0.5, z], [-2.82, 0.5, z], [-2.82, 0.28, z], [-2.64, 0.28, z]], mBody);
}}
 
const mFace = getMat("FRONT_FACE", 0xffffff, 0.95);
for (const z of [-0.62, 0.62]) {{
  const c1 = new THREE.EllipseCurve(0, 0, 0.18, 0.18, 0, Math.PI * 2); ln(c1.getPoints(20).map(p => [-2.585, p.y + 0.88, z + p.x * 0.04]), mFace);
  const c2 = new THREE.EllipseCurve(0, 0, 0.1, 0.1, 0, Math.PI * 2); ln(c2.getPoints(12).map(p => [-2.585, p.y + 0.88, z + p.x * 0.04]), mFace);
}}
 
const mEng  = getMat("ENGINE_BAY", 0xff6600, 0.9);
const mCool = getMat("ENGINE_BAY", 0x00ccff, 0.8);
const engG = new THREE.Group(); carGroup.add(engG);
pos(bx(0.85, 0.52, 0.55, mEng, engG), -1.82, 0.82, 0);
pos(bx(0.82, 0.12, 0.48, mEng, engG), -1.82, 1.14, 0);
pos(bx(0.55, 0.15, 0.42, mEng, engG), -1.75, 1.25, 0);
pos(bx(0.2, 0.2, 0.2, mEng, engG), -1.35, 1.05, -0.4);

for(let k=0; k<3; k++) {{
  const pPts = new THREE.EllipseCurve(0, 0, 0.08 + k*0.02, 0.08 + k*0.02, 0, Math.PI*2).getPoints(8);
  ln(pPts.map(p => [-2.26, 0.82 + p.y, p.x]), mEng, engG);
}}

pos(bx(0.08, 0.58, 1.32, mCool, engG), -2.42, 0.88, 0);
const fanC = cy(0.22, 0.22, 0.04, 12, mCool, engG); rot(fanC, 0, 0, Math.PI / 2); pos(fanC, -2.36, 0.88, 0);
ln([[-2.34, 1.12, 0.35], [-2.05, 1.12, 0.35], [-1.9, 1.05, 0.26]], mCool, engG); 
ln([[-2.34, 0.62, -0.35], [-1.95, 0.62, -0.24]], mCool, engG);                     
 
ln([[-1.62, 0.78, -0.28], [-1.62, 0.44, -0.34], [-0.95, 0.32, -0.34], [0.85, 0.32, -0.34], [1.5, 0.35, -0.22], [2.2, 0.35, -0.42], [2.44, 0.38, -0.42]], mEng);
const muffler = cy(0.09, 0.09, 0.46, 8, mEng); rot(muffler, 0, 0, Math.PI / 2); pos(muffler, 1.84, 0.33, -0.34); carGroup.add(muffler);
 
const mTrans = getMat("TRANSMISSION", 0x33ffaa, 0.85);
pos(bx(0.32, 0.38, 0.36, mTrans), -1.24, 0.68, 0);
pos(bx(0.48, 0.28, 0.26, mTrans), -0.84, 0.54, 0);
pos(bx(0.22, 0.26, 0.28, mTrans), -0.48, 0.50, -0.08);
const fShaft = cy(0.02, 0.02, 1.1, 6, mTrans); rot(fShaft, 0, 0, 1.45); pos(fShaft, -1.06, 0.47, -0.1); carGroup.add(fShaft);
const rShaft = cy(0.025, 0.025, 1.9, 6, mTrans); rot(rShaft, 0, 0, -1.52); pos(rShaft, 0.48, 0.44, -0.05); carGroup.add(rShaft);
 
pos(bx(0.08, 0.12, 1.84, mTrans), -1.64, 0.46, 0);
const fDiffP = cy(0.18, 0.18, 0.22, 12, mTrans); rot(fDiffP, Math.PI / 2, 0, 0); pos(fDiffP, -1.64, 0.46, -0.16); carGroup.add(fDiffP);
pos(bx(0.09, 0.14, 1.84, mTrans), 1.58, 0.38, 0);
const rDiffP = cy(0.20, 0.20, 0.24, 12, mTrans); rot(rDiffP, Math.PI / 2, 0, 0); pos(rDiffP, 1.58, 0.38, 0); carGroup.add(rDiffP);
 
const mSusp = getMat("FRONT_WHEEL", 0xffdd00, 0.9);
function makeDetailedSuspension(ax, ay, az) {{
  const sg = new THREE.Group(); carGroup.add(sg);
  for (let l = 0; l < 4; l++) {{ 
    ln([[ax - (0.84 - l * 0.08) / 2, ay - l * 0.025, az], [ax + (0.84 - l * 0.08) / 2, ay - l * 0.025, az]], mSusp, sg); 
  }}
  const shk = cy(0.025, 0.02, 0.44, 6, mSusp, sg); pos(shk, ax, ay + 0.22, az);
}}
makeDetailedSuspension(-1.64, 0.30, -0.74); makeDetailedSuspension(-1.64, 0.30, 0.74);
makeDetailedSuspension(1.58, 0.24, -0.74); makeDetailedSuspension(1.58, 0.24, 0.74);
 
const mSteer = getMat("STEERING", 0xcc44ff, 0.9);

const swGroup = new THREE.Group();
swGroup.position.set(-0.78, 1.46, -0.22);
swGroup.rotation.x = -Math.PI / 6;
swGroup.rotation.y = Math.PI / 2;

carGroup.add(swGroup);

const rimCirc = new THREE.EllipseCurve(0, 0, 0.18, 0.18, 0, Math.PI * 2).getPoints(24);
ln(rimCirc.map(p => [p.x, p.y, 0]), mSteer, swGroup);

ln([[-0.18, 0, 0], [0.18, 0, 0]], mSteer, swGroup);
ln([[0, 0, 0], [0, -0.18, 0]], mSteer, swGroup);

ln([[-0.78, 1.46, -0.22], [-1.34, 0.76, -0.38]], mSteer);
pos(bx(0.12, 0.12, 0.12, mSteer), -1.34, 0.76, -0.38);
ln([[-1.34, 0.76, -0.38], [-1.42, 0.52, -0.38]], mSteer);
ln([[-1.42, 0.52, -0.38], [-1.64, 0.46, 0.88]], mSteer);
ln([[-1.64, 0.44, -0.88], [-1.64, 0.44, 0.88]], mSteer);
 
const mBrk = getMat("FRONT_WHEEL", 0xff2233, 0.95);
function makeBrake(x, z, front) {{
  const ay = front ? 0.46 : 0.38;
  const rd = cy(0.31, 0.31, 0.04, 24, mBrk); rot(rd, Math.PI / 2, 0, 0); pos(rd, x, ay, z); carGroup.add(rd);
  
  for(let s=0; s<8; s++) {{
    const an = (s / 8) * Math.PI * 2;
    ln([[x, ay, z], [x, ay + Math.sin(an)*0.3, z + Math.cos(an)*0.3]], mBrk);
  }}
  const caliper = pos(bx(0.16, 0.14, 0.08, mBrk), x + 0.1, ay + 0.16, z + (z < 0 ? 0.04 : -0.04));
  carGroup.add(caliper);
}}
makeBrake(-1.64, -0.88, true);  makeBrake(-1.64, 0.88, true);
makeBrake(1.58, -0.88, false);  makeBrake(1.58, 0.88, false);
 
const mElec = getMat("ELECTRICAL", 0x44aaff, 0.7);
pos(bx(0.28, 0.24, 0.26, mElec), -1.58, 0.98, -0.68);
pos(bx(0.22, 0.12, 0.28, mElec), -1.22, 1.05, -0.68);
ln([[-1.58, 1.05, -0.55], [-1.22, 1.05, -0.55]], mElec);
ln([[-1.22, 1.05, -0.68], [-0.82, 1.1, 0.0]], mElec);
 
const mFuel = getMat("ENGINE_BAY", 0x0088ff, 0.4);
pos(bx(1.0, 0.24, 1.32, mFuel), 0.88, 0.28, 0);
 
const mRim  = getMat("FRONT_WHEEL", 0x00ffcc, 0.85);
const mTyre = getMat("FRONT_WHEEL", 0x00ffcc, 0.32);
function makeWheel(ax, az, front) {{
  const wg = new THREE.Group(); carGroup.add(wg);
  wg.position.set(ax, front ? 0.46 : 0.38, az);
  cy(0.52, 0.52, 0.28, 28, mTyre, wg).rotation.x = Math.PI / 2;
  cy(0.36, 0.36, 0.24, 18, mRim,  wg).rotation.x = Math.PI / 2;
  for (let i = 0; i < 20; i++) {{
    const a = i / 20 * Math.PI * 2; ln([[0.16, Math.cos(a) * 0.52, Math.sin(a) * 0.52], [-0.16, Math.cos(a) * 0.52, Math.sin(a) * 0.52]], mTyre, wg);
  }}
}}
makeWheel(-1.64, -1.02, true);  makeWheel(-1.64, 1.02, true);
makeWheel(1.58, -1.02, false); makeWheel(1.58, 1.02, false);
 
const grid = new THREE.GridHelper(18, 24, 0x002e1a, 0x001a0e); grid.position.y = -0.14; scene.add(grid);
 
const currentLook = {{ x: START_LX, y: START_LY, z: START_LZ }};
const currentCamOffset = {{ 
  x: START_CX - START_LX, 
  y: START_CY - START_LY, 
  z: START_CZ - START_LZ 
}};
 
const tl = gsap.timeline();
tl.to(currentLook, {{ x: 0.0, y: 1.0, z: 0.0, duration: 0.6, ease: 'power2.inOut' }})
  .to(currentCamOffset, {{ x: 7.0, y: 2.5, z: 7.0, duration: 0.6, ease: 'power2.inOut' }}, "<");
 
tl.to(currentLook, {{ x: TARGET_LX, y: TARGET_LY, z: TARGET_LZ, duration: 1.0, ease: 'power3.out' }}, "+=0.1")
  .to(currentCamOffset, {{ 
    x: TARGET_CX - TARGET_LX, 
    y: TARGET_CY - TARGET_LY, 
    z: TARGET_CZ - TARGET_LZ, 
    duration: 1.0, 
    ease: 'power3.out'  
  }}, "<");
 
let orbitAngle = 0;

function animate() {{
  requestAnimationFrame(animate);
  
  orbitAngle += 0.004;
  
  const radius = Math.sqrt(currentCamOffset.x * currentCamOffset.x + currentCamOffset.z * currentCamOffset.z);
  const baseAngle = Math.atan2(currentCamOffset.z, currentCamOffset.x);
  const finalAngle = baseAngle + orbitAngle;
  
  cam.position.x = currentLook.x + radius * Math.cos(finalAngle);
  cam.position.y = currentLook.y + currentCamOffset.y;
  cam.position.z = currentLook.z + radius * Math.sin(finalAngle);
  
  cam.lookAt(currentLook.x, currentLook.y, currentLook.z);
  
  renderer.render(scene, cam);
}}
animate();
</script>
</body>
</html>
"""
    components.html(html_code, height=520)


# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🧭 Navigation Menu")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio(
    "Select an Option:",
    ["📚 Predefined Categories", "🤖 AI Assistant"]
)
st.sidebar.markdown("---")

# --- TAB 1: PREDEFINED CATEGORIES ---
if app_mode == "📚 Predefined Categories":
    st.title("📚 Browse Known Faults")
    st.write("Select your vehicle's symptoms from the menus below.")
    
    if "tab1_memory" not in st.session_state:
        st.session_state.tab1_memory = None

    if not df.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            categories = df['category'].dropna().unique()
            selected_category = st.selectbox("1️⃣ Select Category", categories)
            
        with col2:
            subcats = df[df['category'] == selected_category]['subcategory'].dropna().unique()
            selected_subcategory = st.selectbox("2️⃣ Select Subcategory", subcats)
            
        with col3:
            symptoms = df[(df['category'] == selected_category) & (df['subcategory'] == selected_subcategory)]['symptom'].dropna().unique()
            selected_symptom = st.selectbox("3️⃣ Select Specific Issue", symptoms)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # FIX: Define separate UI layout slots so visual order matches user intent, 
        # but execution order forces the model to render before the blocking API code.
        btn_slot = st.empty()
        model_slot = st.empty()
        
        # Force the model to render immediately in its designated slot
        with model_slot:
            render_3d_hologram(selected_category)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Render the button inside its designated slot above the model
        if btn_slot.button("🛠️ Diagnose Issue"):
            loading_container = st.empty()
            loading_container.markdown("""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 150px;">
                    <div style="color: #00ffff; font-size: 18px; margin-bottom: 10px; font-weight: bold;">AI Systems Analyzing Telemetry...</div>
                    <div style="width: 200px; height: 4px; background: rgba(0,255,255,0.2); overflow: hidden; border-radius: 2px;">
                        <div style="width: 40px; height: 100%; background: #00ffff; box-shadow: 0 0 10px #00ffff; animation: drive 1s linear infinite;"></div>
                    </div>
                </div>
                <style>
                    @keyframes drive { 0% { transform: translateX(-40px); } 100% { transform: translateX(200px); } }
                </style>
            """, unsafe_allow_html=True)
            
            selected_row = df[
                (df['category'] == selected_category) & 
                (df['subcategory'] == selected_subcategory) & 
                (df['symptom'] == selected_symptom)
            ].iloc[0]
            
            st.session_state.tab1_memory = get_simplified_fix(selected_row)
            loading_container.empty()
        
        if st.session_state.tab1_memory:
            render_diagnostic_ui(st.session_state.tab1_memory)

# --- TAB 2: AI ASSISTANT ---
elif app_mode == "🤖 AI Assistant":
    st.title("🤖 AI Diagnostic Assistant")
    st.write("Describe your vehicle's problem in your own words.")
    
    if "tab2_memory" not in st.session_state:
        st.session_state.tab2_memory = None
        
    user_query = st.text_area("What is happening with your vehicle?", height=150, placeholder="Example: My car shakes violently when I brake at high speeds...")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # FIX: Define layout placeholders for the button and 3D model
    ai_btn_slot = st.empty()
    ai_model_slot = st.empty()
    
    # Force the model to load into its slot first so it stays on screen during the API call
    current_ai_cat = st.session_state.tab2_memory.get("category", "DEFAULT") if st.session_state.tab2_memory else "DEFAULT"
    with ai_model_slot:
        render_3d_hologram(current_ai_cat)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Render the AI button inside its designated slot above the model
    if ai_btn_slot.button("🧠 Diagnose with AI"):
        if user_query.strip():
            loading_container = st.empty()
            loading_container.markdown("""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 150px;">
                    <div style="color: #00ffff; font-size: 18px; margin-bottom: 10px; font-weight: bold;">AI Quantum Engine Computing Trajectory...</div>
                    <div style="width: 200px; height: 4px; background: rgba(0,255,255,0.2); overflow: hidden; border-radius: 2px;">
                        <div style="width: 40px; height: 100%; background: #00ffff; box-shadow: 0 0 10px #00ffff; animation: drive 1s linear infinite;"></div>
                    </div>
                </div>
                <style>
                    @keyframes drive { 0% { transform: translateX(-40px); } 100% { transform: translateX(200px); } }
                </style>
            """, unsafe_allow_html=True)
            
            ai_res = get_ai_diagnosis(user_query)
            st.session_state.tab2_memory = ai_res
            loading_container.empty()
            st.rerun()
        else:
            st.error("Please describe your issue in the text box before clicking Diagnose.")
            
    if st.session_state.tab2_memory:
        render_diagnostic_ui(st.session_state.tab2_memory)
