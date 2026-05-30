import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import json

# --- PAGE CONFIGURATION & CUSTOM CSS ---
st.set_page_config(page_title="AI Vehicle Diagnostic Assistant", layout="wide", page_icon="🚗")

st.markdown("""
    <style>
    /* Make the Sidebar Navigation Huge */
    .css-1544g2n {padding-top: 2rem;}
    .stRadio > div > label > div {font-size: 24px !important; font-weight: bold; margin-bottom: 10px;}
    
    /* Make the Diagnose Buttons Big and Eye-Catching */
    div.stButton > button:first-child {
        background-color: #28a745;
        color: white;
        height: 60px;
        width: 100%;
        border-radius: 12px;
        font-size: 22px;
        font-weight: bold;
        border: 2px solid #1e7e34;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #218838;
        border: 2px solid #1c7430;
    }
    </style>
""", unsafe_allow_html=True)

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
    """Renders the completely styled UI output for both Tabs so they look identical"""
    
    if data.get('is_fallback'):
         st.warning("⚠️ **SYSTEM NOTICE: AI API Limit Reached. Currently running in offline database mode.**", icon="⏳")

    # 1. HUGE SAFETY WARNING BOX
    st.markdown(f"""
    <div style="background-color: #ffe6e6; padding: 25px; border-radius: 12px; border-left: 10px solid #cc0000; margin-bottom: 25px;">
        <h1 style="color: #cc0000; margin-top: 0; font-size: 32px;">🚨 SAFETY FIRST</h1>
        <p style="color: #990000; font-size: 22px; font-weight: bold; margin-bottom: 0;">{data['safety']}</p>
    </div>
    """, unsafe_allow_html=True)

    # 2. HUGE MECHANIC ADVICE BOX (Only shows if mechanic is needed)
    mechanic_text = str(data.get('mechanic', '')).strip()
    if mechanic_text and mechanic_text.lower() not in ["no", "none", "n/a", ""]:
         st.markdown(f"""
        <div style="background-color: #fff4cc; padding: 25px; border-radius: 12px; border-left: 10px solid #ffcc00; margin-bottom: 25px;">
            <h1 style="color: #b38f00; margin-top: 0; font-size: 32px;">🛠️ MECHANIC ADVICE</h1>
            <p style="color: #806600; font-size: 22px; font-weight: bold; margin-bottom: 0;">{mechanic_text}</p>
        </div>
        """, unsafe_allow_html=True)

    # 3. DIAGNOSIS, TOOLS & STEPS
    st.markdown("---")
    st.markdown(f"### 🔍 **Diagnosis:**\n{data['diagnosis']}")
    st.markdown(f"### 🧰 **Tools Needed:**\n{data['tools']}")
    st.markdown("### 📋 **Step-by-Step Fix:**")
    st.markdown(data['steps'])


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
        return data
    except Exception:
        # FALLBACK: If API is blocked, format the RAW dataset cleanly!
        return {
            "safety": row['safety_warning'],
            "mechanic": row['mechanic_advice'],
            "diagnosis": row['possible_fault'],
            "tools": row['tools_needed'],
            "steps": row['beginner_steps'],
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
        
        # Save to CSV
        try:
            pd.DataFrame([data['csv_data']]).to_csv(FILE_NAME, mode='a', header=False, index=False)
        except Exception:
            pass
            
        data['is_fallback'] = False
        return data
    except Exception:
        # FALLBACK: Provide a safe generic response if API is blocked
        return {
            "safety": "Always park on a flat surface, engage the parking brake, and let the engine cool completely before inspecting.",
            "mechanic": "If you notice smoke, active fluid leaks, or require specialized diagnostic equipment.",
            "diagnosis": f"Potential issue related to: '{user_input}'. Please check your dashboard indicators.",
            "tools": "Standard hand tools, protective gloves, flashlight.",
            "steps": "1. Wait for the free-tier API window to clear.\n2. Visually inspect the area for obvious loose wires or leaks.\n3. Try browsing the 'Predefined Categories' tab for offline solutions.",
            "is_fallback": True
        }

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
    st.write("Select your vehicle's symptoms from the interactive menus below.")
    
    # Create a memory state for Tab 1
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
        
        # When clicked, save the result into memory
        if st.button("🛠️ Diagnose Issue"):
            with st.spinner("Analyzing database and simplifying instructions..."):
                selected_row = df[
                    (df['category'] == selected_category) & 
                    (df['subcategory'] == selected_subcategory) & 
                    (df['symptom'] == selected_symptom)
                ].iloc[0]
                
                # Fetch and SAVE to memory
                st.session_state.tab1_memory = get_simplified_fix(selected_row)
        
        # Always display whatever is in memory, preventing wasted API calls!
        if st.session_state.tab1_memory:
            render_diagnostic_ui(st.session_state.tab1_memory)

# --- TAB 2: AI ASSISTANT ---
elif app_mode == "🤖 AI Assistant":
    st.title("🤖 AI Diagnostic Assistant")
    st.write("Describe your vehicle's problem in your own words.")
    
    # Create a memory state for Tab 2
    if "tab2_memory" not in st.session_state:
        st.session_state.tab2_memory = None
        
    user_query = st.text_area("What is happening with your vehicle?", height=150, placeholder="Example: My car shakes violently when I brake at high speeds...")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # When clicked, save the result into memory
    if st.button("🧠 Diagnose with AI"):
        if user_query.strip():
            with st.spinner("Processing symptoms with Google Gemini..."):
                # Fetch and SAVE to memory
                st.session_state.tab2_memory = get_ai_diagnosis(user_query)
        else:
            st.error("Please describe your issue in the text box before clicking Diagnose.")
            
    # Always display whatever is in memory!
    if st.session_state.tab2_memory:
        render_diagnostic_ui(st.session_state.tab2_memory)
