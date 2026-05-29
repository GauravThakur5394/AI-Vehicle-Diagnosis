import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import google.api_core.exceptions

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Vehicle Diagnostic AI", layout="wide")
genai.configure(api_key=st.secrets["API_KEY"])

FILE_1 = "expanded_vehicle_fault_dataset.csv"
FILE_2 = "merged_vehicle_fault_dataset.csv"

# --- 2. DATA LOAD & INTELLIGENT MERGE ---
@st.cache_data
def load_data():
    df1 = pd.read_csv(FILE_1)
    df2 = pd.read_csv(FILE_2)
    
    cols_needed = ['vehicle_type', 'category', 'subcategory', 'symptom', 'possible_fault', 
                   'severity', 'user_fixable', 'tools_needed', 'beginner_steps', 
                   'mechanic_advice', 'safety_warning']
    
    for col in cols_needed:
        if col not in df1.columns: df1[col] = "N/A"
        if col not in df2.columns: df2[col] = "N/A"
    
    combined_df = pd.concat([df1[cols_needed], df2[cols_needed]], ignore_index=True)
    return combined_df.drop_duplicates()

def get_simplified_fix(row):
    model = genai.GenerativeModel('gemini-3.5-flash')
    
    prompt = f"""
    You are an AI Vehicle Fault Assistant for beginners.
    Rewrite the provided technical vehicle repair instructions into simple, 5th-grade level English.
    
    DATASET CONTEXT:
    - Fault: {row['possible_fault']}
    - Tools Needed: {row['tools_needed']}
    - Steps: {row['beginner_steps']}
    - Safety Warning: {row['safety_warning']}
    - Mechanic Advice: {row['mechanic_advice']}
    
    FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
    **Diagnosis:** [Simple explanation of the fault]
    **Tools Needed:** [List tools simply]
    **Safety First:** [Simplified safety warning in bold]
    **Step-by-Step Fix:** [Numbered list of simplified steps]
    **When to Call a Mechanic:** [Simplified mechanic advice]
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except google.api_core.exceptions.ResourceExhausted:
        # Fallback to displaying raw text so the app doesn't crash during evaluation
        fallback_text = f"""
        ⚠️ **Notice:** AI simplification limit reached. Showing raw database instructions.
        
        **Diagnosis:** {row['possible_fault']}
        **Tools Needed:** {row['tools_needed']}
        **Safety First:** {row['safety_warning']}
        **Step-by-Step Fix:** {row['beginner_steps']}
        **When to Call a Mechanic:** {row['mechanic_advice']}
        """
        return fallback_text
    except Exception as e:
        return f"Error simplifying text: {str(e)}"

# --- 3. AI ENGINE (Optimized for Detail) ---
def get_ai_diagnosis(user_input):
    model = genai.GenerativeModel('gemini-3.5-flash')
    
    prompt = f"""
    You are an AI Vehicle Fault Assistant for beginners with zero automotive knowledge.
    The user is reporting this issue: "{user_input}"
    
    Your job is to provide a diagnostic guide for this issue in simple, 5th-grade level English.
    
    STRICT RULES:
    1. DO NOT invent tools.
    2. Keep explanations simple.
    3. Emphasize the safety warning.
    
    Return ONLY a valid JSON object with this exact structure:
    {{
        "ui_response": "**Diagnosis:** [Explanation]\\n**Tools Needed:** [List tools simply]\\n**Safety First:** [Simplified safety warning in bold]\\n**Step-by-Step Fix:** [Numbered list of simplified steps]\\n**When to Call a Mechanic:** [Simplified mechanic advice]",
        "csv_data": {{
            "vehicle_type": "Car", 
            "category": "General", 
            "subcategory": "General", 
            "symptom": "{user_input[:50]}", 
            "possible_fault": "AI Generated Diagnosis", 
            "severity": "Medium", 
            "user_fixable": "Yes", 
            "tools_needed": "Basic tools", 
            "beginner_steps": "Follow generated steps", 
            "mechanic_advice": "Consult a professional", 
            "safety_warning": "Exercise caution"
        }}
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        # Clean response and parse JSON
        text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        
        # Save to CSV for the "Self-Healing" logic
        new_row = pd.DataFrame([data['csv_data']])
        new_row.to_csv(FILE_1, mode='a', header=False, index=False)
        
        return data['ui_response']
        
    except google.api_core.exceptions.ResourceExhausted:
        # Custom fallback message when the free tier limit drops out
        error_ui = """
        ⚠️ **System Status Notice:** The system is currently handling high diagnostic traffic. 
        
        **Diagnosis:** Temporary API connection rate-limit reached.
        **Tools Needed:** None required.
        **Safety First:** **Please check your dashboard indicators carefully.**
        **Step-by-Step Fix:** 1. Wait 60 seconds for the free-tier API window to clear.
        2. Click the **Diagnose** button again to retry your request.
        3. If the issue persists, browse the predefined categories in the **Browse Known Faults** tab.
        **When to Call a Mechanic:** If you require immediate roadside emergency assistance.
        """
        return error_ui
        
    except Exception as e:
        return f"Error diagnosing issue: {str(e)}"
# --- 4. UI INTERFACE ---
st.title("🚗 AI Vehicle Diagnostic Assistant")
tab1, tab2 = st.tabs(["🔍 Browse Known Faults", "🤖 AI Assistant"])

df = load_data()

with tab1:
    st.header("Search Knowledge Base")
    # Filters
    cat = st.selectbox("Category", df['category'].unique())
    sub = st.selectbox("Subcategory", df[df['category'] == cat]['subcategory'].unique())
    symp = st.selectbox("Symptom", df[(df['category'] == cat) & (df['subcategory'] == sub)]['symptom'].unique())
    
    if st.button("View Fix"):
        # Select the specific row from your dataframe
        row = df[(df['category'] == cat) & (df['subcategory'] == sub) & (df['symptom'] == symp)].iloc[0]
        
        # Use a spinner while Gemini processes the simplification
        with st.spinner("Simplifying solution for you..."):
            # Call your new function to get the structured, beginner-friendly format
            st.markdown(get_simplified_fix(row))

with tab2:
    st.header("Ask AI")
    user_q = st.text_area("Describe your issue:")
    if st.button("Diagnose"):
        with st.spinner("Expert mechanic is analyzing..."):
            st.markdown(get_ai_diagnosis(user_q))
