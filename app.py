import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="MBTI Classifier", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4B9CD3;'>🧠 MBTI Personality Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Analyze your writing and discover your personality type!</p>", unsafe_allow_html=True)
st.markdown("---")

# Load Models
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("model_dim_EI.pkl", "rb") as f:
    model_EI = pickle.load(f)
with open("model_dim_SN.pkl", "rb") as f:
    model_SN = pickle.load(f)
with open("model_dim_TF.pkl", "rb") as f:
    model_TF = pickle.load(f)
with open("model_dim_JP.pkl", "rb") as f:
    model_JP = pickle.load(f)

# Dictionaries
mbti_descriptions = {
    "ISTJ": "ISTJ - The Logistician: Responsible, serious, logical, values traditions and structure.",
    "ISFJ": "ISFJ - The Defender: Nurturing, reliable, detail-oriented, values loyalty and harmony.",
    "INFJ": "INFJ - The Advocate: Insightful, idealistic, deeply caring, driven by inner values.",
    "INTJ": "INTJ - The Architect: Strategic, independent, analytical, long-term planner.",
    "ISTP": "ISTP - The Virtuoso: Practical, hands-on, curious, thrives on action and problem-solving.",
    "ISFP": "ISFP - The Adventurer: Artistic, sensitive, open-minded, enjoys personal freedom.",
    "INFP": "INFP - The Mediator: Empathetic, imaginative, guided by strong inner values.",
    "INTP": "INTP - The Thinker: Analytical, abstract, independent, values logic and ideas.",
    "ESTP": "ESTP - The Entrepreneur: Energetic, spontaneous, loves challenges and social interaction.",
    "ESFP": "ESFP - The Entertainer: Outgoing, fun-loving, lives in the moment and enjoys life.",
    "ENFP": "ENFP - The Campaigner: Enthusiastic, creative, warm, values authenticity and connection.",
    "ENTP": "ENTP - The Debater: Energetic, curious, enjoys intellectual challenges and change.",
    "ESTJ": "ESTJ - The Executive: Organized, responsible, values rules and structure.",
    "ESFJ": "ESFJ - The Consul: Caring, sociable, puts others' needs first, values harmony.",
    "ENFJ": "ENFJ - The Protagonist: Inspiring, empathetic, natural leader, driven to help others.",
    "ENTJ": "ENTJ - The Commander: Bold, strategic, confident, thrives on leadership and vision."
}

career_suggestions = {
    "ISTJ": ["Accountant", "Auditor", "Military Officer", "Project Manager"],
    "ISFJ": ["Nurse", "Elementary School Teacher", "Social Worker", "Office Manager"],
    "INFJ": ["Psychologist", "Writer", "Counselor", "Non-profit Leader"],
    "INTJ": ["Scientist", "Engineer", "Strategist", "Data Analyst"],
    "ISTP": ["Mechanic", "Engineer", "Pilot", "Surgeon"],
    "ISFP": ["Artist", "Graphic Designer", "Chef", "Photographer"],
    "INFP": ["Writer", "Therapist", "Librarian", "Humanitarian Worker"],
    "INTP": ["Philosopher", "Programmer", "Inventor", "Researcher"],
    "ESTP": ["Entrepreneur", "Sales Representative", "Stockbroker", "Athlete"],
    "ESFP": ["Actor", "Musician", "Event Planner", "Public Relations Specialist"],
    "ENFP": ["Journalist", "Life Coach", "Marketer", "Teacher"],
    "ENTP": ["Startup Founder", "Consultant", "Lawyer", "Inventor"],
    "ESTJ": ["Police Officer", "Manager", "Judge", "Banker"],
    "ESFJ": ["Nurse", "Social Worker", "Real Estate Agent", "Customer Support"],
    "ENFJ": ["Teacher", "Counselor", "HR Manager", "Public Speaker"],
    "ENTJ": ["CEO", "Executive", "Lawyer", "Project Manager"]
}

dimension_explanations = {
    "E": "Extraversion: Outgoing, enjoys social interaction.",
    "I": "Introversion: Reflective, enjoys solitude.",
    "S": "Sensing: Focuses on concrete facts and details.",
    "N": "Intuition: Enjoys ideas, patterns, and possibilities.",
    "T": "Thinking: Makes decisions with logic and objectivity.",
    "F": "Feeling: Makes decisions based on empathy and values.",
    "J": "Judging: Prefers structure, planning, and organization.",
    "P": "Perceiving: Prefers flexibility and spontaneity."
}

# Predict MBTI
def predict_mbti(text):
    X = vectorizer.transform([text])
    prob_EI = model_EI.predict_proba(X)[0]
    prob_SN = model_SN.predict_proba(X)[0]
    prob_TF = model_TF.predict_proba(X)[0]
    prob_JP = model_JP.predict_proba(X)[0]

    mbti = ""
    confidences = {}

    mbti += "E" if prob_EI[0] >= 0.5 else "I"
    confidences["E-I"] = prob_EI

    mbti += "S" if prob_SN[0] >= 0.5 else "N"
    confidences["S-N"] = prob_SN

    mbti += "T" if prob_TF[0] >= 0.5 else "F"
    confidences["T-F"] = prob_TF

    mbti += "J" if prob_JP[0] >= 0.5 else "P"
    confidences["J-P"] = prob_JP

    return mbti, confidences

# Plot Horizontal Bar Chart
def plot_bar_chart(confidences):
    labels = ['E vs I', 'S vs N', 'T vs F', 'J vs P']
    confidence_values = [confidences[key][1] for key in ['E-I', 'S-N', 'T-F', 'J-P']]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, confidence_values, color='steelblue')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Confidence Score')
    ax.set_title("📊 MBTI Dimension Confidence")
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}', va='center', fontsize=10)
    st.pyplot(fig)

# UI
user_input = st.text_area("✏️ Enter text that reflects your personality:", height=200, placeholder="Tell us about yourself...")

if st.button("🎯 Predict MBTI"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        mbti_result, confs = predict_mbti(user_input)

        st.markdown(f"<h2 style='color:#0E76A8;'>🧠 Your MBTI type: <span style='color:#f63366'>{mbti_result}</span></h2>", unsafe_allow_html=True)

        st.markdown("### 📘 Personality Type Overview")
        st.info(mbti_descriptions.get(mbti_result, "No description available."))

        st.markdown("### 💼 Suggested Careers")
        careers = career_suggestions.get(mbti_result, [])
        if careers:
            st.markdown("<ul>" + "".join([f"<li>{job}</li>" for job in careers]) + "</ul>", unsafe_allow_html=True)
        else:
            st.write("No suggestions available.")

        st.markdown("### 📊 Confidence per MBTI Dimension")
        plot_bar_chart(confs)

        st.markdown("### 🔎 Explanation of MBTI Dimensions")
        for dim in mbti_result:
            st.markdown(f"- **{dim}**: {dimension_explanations[dim]}")
        
