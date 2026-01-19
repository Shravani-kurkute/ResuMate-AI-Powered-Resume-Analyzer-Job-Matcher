import streamlit as st
import pandas as pd
import numpy as np
import pickle
from src.Job_Matcher_06 import get_top_matches
from src.Resume_Parser_07 import parse_resume
import tempfile
import os


# Page configuration
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Enhanced Custom CSS
st.markdown("""
    <style>
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #2ecc71;
        --danger-color: #e74c3c;
        --warning-color: #f39c12;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.2rem;
        font-weight: 500;
        margin-bottom: 2rem;
    }
    
    .prediction-box {
        padding: 2.5rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-size: 2.8rem;
        font-weight: 900;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
        animation: fadeIn 0.5s ease-in;
        border: 3px solid rgba(255,255,255,0.3);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        font-size: 1.3rem;
        font-weight: 700;
        padding: 1rem;
        border-radius: 15px;
        border: none;
        box-shadow: 0 8px 20px rgba(46, 204, 113, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 25px rgba(46, 204, 113, 0.6);
        background: linear-gradient(135deg, #27ae60 0%, #229954 100%);
    }
    
    .big-emoji {
        font-size: 4rem;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    </style>
""", unsafe_allow_html=True)


# Load model and scaler
@st.cache_resource
def load_models():
    try:
        with open('Models/Best_Placement_Model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('Models/Final_Scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found!")
        st.stop()


model, scaler = load_models()


# Header
st.markdown('<p class="main-header">🎓 Student Placement Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Career Placement Prediction System | 91.36% Accuracy</p>', unsafe_allow_html=True)
st.markdown("---")


# Sidebar
with st.sidebar:
    st.markdown("## 📊 Model Intelligence")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 1rem;'>
        <h3 style='margin: 0; color: white;'>⚡ Model Stats</h3>
        <p style='font-size: 1.1rem; margin: 0.5rem 0;'><b>Algorithm:</b> Logistic Regression</p>
        <p style='font-size: 1.1rem; margin: 0.5rem 0;'><b>Accuracy:</b> 91.36%</p>
        <p style='font-size: 1.1rem; margin: 0.5rem 0;'><b>F1-Score:</b> 0.9136</p>
        <p style='font-size: 1.1rem; margin: 0.5rem 0;'><b>NLP:</b> Resume Parser</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🎯 Placement Tiers")
    st.markdown("""
    <div style='padding: 1rem; background: #fff3cd; border-radius: 10px; margin: 0.5rem 0;'>
        <span style='font-size: 1.5rem;'>🟡</span> <b>Premium</b><br>
        <small>≥20 LPA</small>
    </div>
    <div style='padding: 1rem; background: #d1ecf1; border-radius: 10px; margin: 0.5rem 0;'>
        <span style='font-size: 1.5rem;'>🟢</span> <b>Standard</b><br>
        <small>15-19 LPA</small>
    </div>
    <div style='padding: 1rem; background: #d6d8db; border-radius: 10px; margin: 0.5rem 0;'>
        <span style='font-size: 1.5rem;'>🔵</span> <b>Basic</b><br>
        <small><15 LPA</small>
    </div>
    <div style='padding: 1rem; background: #f8d7da; border-radius: 10px; margin: 0.5rem 0;'>
        <span style='font-size: 1.5rem;'>🔴</span> <b>Not Placed</b><br>
        <small>Needs improvement</small>
    </div>
    """, unsafe_allow_html=True)


# Input Method Selection
st.markdown("## 🚀 Choose Input Method")
input_method = st.radio("Select input method", ["📝 Manual Entry", "📤 Upload Resume (NLP)"], horizontal=True, label_visibility="hidden")

# Initialize variables
marks_10 = 75.0
marks_12 = 75.0
cgpa = 7.0
internships = "No"
training = "No"
innovative_project = "No"
technical_course = "No"
comm_level = 3
tech_skills = 40


# Resume Upload Section
if input_method == "📤 Upload Resume (NLP)":
    st.markdown("### 📄 Upload Your Resume")
    uploaded_file = st.file_uploader("Choose PDF resume", type=['pdf'], help="Upload your resume in PDF format")
    
    if uploaded_file is not None:
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        # Parse resume
        with st.spinner("🔍 Parsing your resume with NLP..."):
            try:
                parsed_data = parse_resume(tmp_path)
                
                if parsed_data:
                    st.success("✅ Resume parsed successfully!")
                    
                    # Extract data
                    marks_10 = parsed_data['tenth_marks']
                    marks_12 = parsed_data['twelfth_marks']
                    cgpa = parsed_data['cgpa']
                    internships = "Yes" if parsed_data['internships'] else "No"
                    training = "Yes" if parsed_data['training'] else "No"
                    innovative_project = "Yes" if parsed_data['projects'] else "No"
                    technical_course = "Yes" if parsed_data['technical_course'] else "No"
                    comm_level = parsed_data['communication_level']
                    tech_skills = parsed_data['technical_skills_score']
                    
                    # Display extracted info
                    st.markdown("### 📊 Extracted Information")
                    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
                    
                    with col_info1:
                        st.metric("CGPA", cgpa)
                        st.metric("10th Marks", f"{marks_10}%")
                    with col_info2:
                        st.metric("12th Marks", f"{marks_12}%")
                        st.metric("Tech Skills", f"{tech_skills}/100")
                    with col_info3:
                        st.metric("Skills Found", len(parsed_data['skills']))
                        st.metric("Communication", f"{comm_level}/5")
                    with col_info4:
                        st.metric("Internships", internships)
                        st.metric("Projects", innovative_project)
                    
                    with st.expander("🔍 View All Extracted Skills"):
                        st.write(", ".join(parsed_data['skills']))
                    
                    st.info("✅ Data auto-filled below! Review and click PREDICT!")
                else:
                    st.error("❌ Failed to parse resume. Try manual entry.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Try manual entry instead")
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass


st.markdown("---")


# Main content
col1, col2 = st.columns([2, 1], gap="large")


with col1:
    st.markdown("## 📝 Student Profile")
    
    tab1, tab2 = st.tabs(["📚 Academic Performance", "💼 Skills & Experience"])
    
    with tab1:
        st.markdown("### 🎓 Academic Details")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            marks_10 = st.number_input("🔟 10th Marks (%)", min_value=0.0, max_value=100.0, value=float(marks_10), step=0.1)
        with col_b:
            marks_12 = st.number_input("📘 12th Marks (%)", min_value=0.0, max_value=100.0, value=float(marks_12), step=0.1)
        with col_c:
            cgpa = st.number_input("🎯 College CGPA", min_value=0.0, max_value=10.0, value=float(cgpa), step=0.1)
    
    with tab2:
        st.markdown("### 💪 Professional Skills")
        
        col_d, col_e = st.columns(2)
        
        with col_d:
            st.markdown("**📜 Certifications**")
            technical_course = st.selectbox("💻 Technical Course?", ["Yes", "No"], index=0 if technical_course=="Yes" else 1)
            training = st.selectbox("🏭 Industrial Training?", ["Yes", "No"], index=0 if training=="Yes" else 1)
            internships = st.selectbox("🎓 Internships Done?", ["Yes", "No"], index=0 if internships=="Yes" else 1)
        
        with col_e:
            st.markdown("**⭐ Skill Ratings**")
            comm_level = st.slider("🗣️ Communication", min_value=1, max_value=5, value=int(comm_level))
            tech_skills = st.slider("💻 Technical Skills", min_value=0, max_value=100, value=int(tech_skills))
            innovative_project = st.selectbox("🚀 Innovative Project?", ["Yes", "No"], index=0 if innovative_project=="Yes" else 1)


with col2:
    st.markdown("## 🎯 Prediction Zone")
    predict_clicked = st.button("🚀 PREDICT MY PLACEMENT", use_container_width=True)
    
    if predict_clicked:
        # Prepare input
        input_data = pd.DataFrame({
            '10th marks': [marks_10],
            '12th marks': [marks_12],
            'Cgpa': [cgpa],
            'Internships(Y/N)': [1 if internships == "Yes" else 0],
            'Training(Y/N)': [1 if training == "Yes" else 0],
            'Innovative Project(Y/N)': [1 if innovative_project == "Yes" else 0],
            'Communication level': [comm_level],
            'Technical Course(Y/N)': [1 if technical_course == "Yes" else 0],
            'Technical_Skills_Score': [tech_skills]
        })
        
        # Predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Map classes
        class_map = {0: 'Basic', 1: 'Not Placed', 2: 'Premium', 3: 'Standard'}
        predicted_class = class_map[prediction]
        emoji_map = {'Basic': '🔵', 'Standard': '🟢', 'Premium': '🟡', 'Not Placed': '🔴'}
        
        # Result
        st.markdown(f"""
        <div class="prediction-box">
            <div class="big-emoji">{emoji_map[predicted_class]}</div>
            {predicted_class}
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence
        st.markdown("### 📊 Prediction Confidence")
        for idx, class_name in class_map.items():
            confidence = prediction_proba[idx] * 100
            col_metric1, col_metric2 = st.columns([3, 1])
            with col_metric1:
                st.write(f"{emoji_map[class_name]} **{class_name}**")
                st.progress(confidence/100)
            with col_metric2:
                st.markdown(f"<h3 style='color: #667eea; margin: 0;'>{confidence:.1f}%</h3>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Job Matching
        st.markdown("### 🏢 Top Company Matches")
        st.markdown(f"Based on your **{predicted_class}** prediction")
        
        student_profile = {
            'cgpa': cgpa,
            'tenth_marks': marks_10,
            'twelfth_marks': marks_12,
            'skills': ['Python', 'Java', 'DSA', 'SQL', 'React'],
            'internships': 1 if internships == "Yes" else 0,
            'projects': 1 if innovative_project == "Yes" else 0,
            'training': 1 if training == "Yes" else 0,
            'technical_course': 1 if technical_course == "Yes" else 0
        }
        
        try:
            matches = get_top_matches(student_profile, predicted_category=predicted_class, top_n=3)
            
            if matches:
                for i, match in enumerate(matches, 1):
                    if match['match_score'] >= 80:
                        card_color = "#d4edda"
                        border_color = "#28a745"
                    elif match['match_score'] >= 60:
                        card_color = "#d1ecf1"
                        border_color = "#17a2b8"
                    else:
                        card_color = "#fff3cd"
                        border_color = "#ffc107"
                    
                    st.markdown(f"""
                    <div style='background: {card_color}; padding: 1.5rem; border-radius: 15px; 
                                margin: 1rem 0; border-left: 5px solid {border_color};
                                box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <h3 style='margin: 0; color: #333;'>#{i} {match['company']}</h3>
                            <div style='background: {border_color}; color: white; padding: 0.5rem 1rem; 
                                        border-radius: 25px; font-weight: bold; font-size: 1.2rem;'>
                                {match['match_score']}% Match
                            </div>
                        </div>
                        <p style='margin: 0.5rem 0; color: #555; font-size: 1.1rem;'>
                            <b>💼</b> {match['role']} | <b>📍</b> {match['location']} | <b>💰</b> {match['package']}
                        </p>
                        <p style='margin: 0.5rem 0; color: #555;'>
                            <b>🎯 CGPA:</b> {match['cgpa_required']} | 
                            {'✅ <span style="color: green;">ELIGIBLE</span>' if match['meets_cgpa'] else '❌ <span style="color: red;">BELOW CUTOFF</span>'}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander(f"📊 Skill Analysis - {match['company']}"):
                        col_s1, col_s2 = st.columns(2)
                        with col_s1:
                            st.markdown("**✅ Skills You Have:**")
                            if match['skills_matched']:
                                for skill in match['skills_matched']:
                                    st.markdown(f"- ✔️ {skill}")
                            else:
                                st.markdown("- ⚠️ No matches")
                        with col_s2:
                            st.markdown("**📚 Skills to Learn:**")
                            if match['skills_gap']:
                                for skill in match['skills_gap']:
                                    st.markdown(f"- 📖 {skill}")
                            else:
                                st.markdown("- 🎉 All skills covered!")
            else:
                st.warning("⚠️ No matching companies found.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px; color: white;'>
    <h2 style='margin: 0; color: white;'>🎓 Student Placement Predictor</h2>
    <p style='font-size: 1.1rem; margin: 0.5rem 0;'>
        Built with ❤️ using ML + NLP
    </p>
    <p style='font-size: 0.95rem; opacity: 0.9;'>
        📊 401 Students | 91.36% Accuracy | 20 VIT Companies | PCA Validated
    </p>
</div>
""", unsafe_allow_html=True)
