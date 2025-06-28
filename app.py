import os
os.environ['STREAMLIT_SERVER_RUN_ON_SAVE'] = 'false'
import streamlit as st
from dotenv import load_dotenv
from triage_agent import process_doctor_conversation
from langchain_core.messages import HumanMessage, AIMessage
import json
import torch
torch.classes.__path__ = []

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDtkhXwbXJ4G34igeM8z44MJieWANsRpWM"

st.set_page_config(
    page_title="AI Doctor - Medical Triage",
    page_icon="👨‍⚕️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_complete" not in st.session_state:
        st.session_state.conversation_complete = False
    if "patient_info" not in st.session_state:
        st.session_state.patient_info = {}

def main():
    initialize_session_state()
    
    st.title("🏥 AI Medical Triage Assistant")
    st.markdown("Welcome! I'm here to help assess your symptoms and provide medical guidance.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Consultation")
        
        if st.session_state.messages:
            for message in st.session_state.messages:
                if isinstance(message, HumanMessage):
                    with st.chat_message("user"):
                        st.write(message.content)
                elif isinstance(message, AIMessage):
                    with st.chat_message("assistant"):
                        st.write(message.content)
        
        if st.session_state.conversation_complete and st.session_state.patient_info:
            st.markdown("---")
            st.subheader("📋 Patient Summary")
            
            patient_info = st.session_state.patient_info
            col_a, col_b = st.columns(2)
            
            with col_a:
                if patient_info.get("age"):
                    st.metric("Age", f"{patient_info['age']} years")
                if patient_info.get("gender"):
                    st.metric("Gender", patient_info["gender"].title())
                if patient_info.get("duration"):
                    st.metric("Duration", patient_info["duration"])
            
            with col_b:
                if patient_info.get("symptoms"):
                    st.metric("Symptoms", len(patient_info["symptoms"]))
                    st.write("**Identified Symptoms:**")
                    for symptom in patient_info["symptoms"]:
                        st.write(f"• {symptom}")
                if patient_info.get("severity"):
                    st.metric("Severity", patient_info["severity"].title())
        
        if not st.session_state.conversation_complete:
            st.markdown("---")
            
            if prompt := st.chat_input("Describe your symptoms or answer the doctor's question..."):
                st.session_state.messages.append(HumanMessage(content=prompt))
                
                with st.spinner("🤔 AI is analyzing your response..."):
                    try:
                        result = process_doctor_conversation(
                            user_input=prompt,
                            conversation_history=st.session_state.messages[:-1]
                        )
                        
                        if result["success"]:
                            st.session_state.messages = result["messages"]
                            st.session_state.conversation_complete = result["conversation_complete"]
                            st.session_state.patient_info = result["patient_info"]
                            
                            st.rerun()
                        else:
                            st.error(f"Error: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
        else:
            st.success("✅ Consultation complete! Review the medical assessment above.")
    
    with col2:
        st.header("📊 Consultation Status")
        
        if st.session_state.messages:
            user_messages = len([m for m in st.session_state.messages if isinstance(m, HumanMessage)])
            ai_messages = len([m for m in st.session_state.messages if isinstance(m, AIMessage)])
            
            st.metric("Questions Asked", ai_messages)
            st.metric("Your Responses", user_messages)
            st.metric("Status", "Complete" if st.session_state.conversation_complete else "In Progress")
            
            if not st.session_state.conversation_complete:
                progress = min(len(st.session_state.messages) / 10, 1.0)
                st.progress(progress)
                st.caption(f"Gathering information... ({int(progress * 100)}%)")
            else:
                st.progress(1.0)
                st.caption("Assessment complete!")
        
        st.header("🚀 Quick Start")
        
        if st.button("💬 Start with symptoms", use_container_width=True):
            if not st.session_state.messages:
                st.session_state.messages.append(HumanMessage(content="I'm not feeling well and would like to discuss my symptoms."))
                st.rerun()
        
        if st.button("🏥 Emergency symptoms", use_container_width=True):
            if not st.session_state.messages:
                st.session_state.messages.append(HumanMessage(content="I'm experiencing severe symptoms that I'm concerned about."))
                st.rerun()
        
        if st.button("🔄 Reset Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_complete = False
            st.session_state.patient_info = {}
            st.rerun()
        
        st.markdown("---")
        st.markdown("**⚠️ Medical Disclaimer:**")
        st.markdown("This AI system provides general guidance only and should not replace professional medical advice. Always consult healthcare professionals for medical decisions.")

if __name__ == "__main__":
    main() 