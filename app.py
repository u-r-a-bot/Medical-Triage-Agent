import os
os.environ['STREAMLIT_SERVER_RUN_ON_SAVE'] = 'false'
import streamlit as st
from dotenv import load_dotenv
from triage_agent import process_doctor_conversation
from langchain_core.messages import HumanMessage, AIMessage
import json
import torch
torch.classes.__path__ = []  # Neutralizes the path inspection

# Load environment variables
load_dotenv()

# Set default API key if not already set
if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDtkhXwbXJ4G34igeM8z44MJieWANsRpWM"

# Page configuration
st.set_page_config(
    page_title="AI Doctor - Medical Triage",
    page_icon="👨‍⚕️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_complete" not in st.session_state:
        st.session_state.conversation_complete = False
    if "patient_info" not in st.session_state:
        st.session_state.patient_info = {}

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("👨‍⚕️ AI Doctor - Medical Triage")
    st.markdown("**Your AI medical assistant is ready to help. Please describe your symptoms or concerns.**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        api_key = st.text_input(
            "Google API Key",
            type="password",
            value=os.getenv("GOOGLE_API_KEY", ""),
            help="Enter your Google Gemini API key"
        )
        
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            st.success("✅ API configured")
        else:
            st.warning("⚠️ Please enter your API key")
            return
        
        st.divider()
        
        # Reset button
        if st.button("🔄 Start New Consultation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_complete = False
            st.session_state.patient_info = {}
            st.rerun()
        
        # Patient info display
        if st.session_state.patient_info:
            st.header("📋 Patient Info")
            info = st.session_state.patient_info
            
            if info.get("age"):
                st.write(f"**Age:** {info['age']}")
            if info.get("gender"):
                st.write(f"**Gender:** {info['gender']}")
            if info.get("symptoms"):
                st.write("**Symptoms:**")
                for symptom in info["symptoms"]:
                    st.write(f"• {symptom}")
            if info.get("severity"):
                st.write(f"**Severity:** {info['severity']}")
            if info.get("duration"):
                st.write(f"**Duration:** {info['duration']}")
        
        st.divider()
        
        st.markdown("""
        **⚠️ Medical Disclaimer:**
        
        This AI assistant is for educational purposes only and should not replace professional medical advice, diagnosis, or treatment.
        
        Always consult with qualified healthcare professionals for medical decisions.
        
        In case of emergency, call emergency services immediately.
        """)
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat messages
            for message in st.session_state.messages:
                if isinstance(message, HumanMessage):
                    # User message
                    with st.chat_message("user"):
                        st.write(message.content)
                elif isinstance(message, AIMessage):
                    # AI message
                    with st.chat_message("assistant"):
                        st.write(message.content)
        
        # Display final recommendation if complete
        if st.session_state.conversation_complete and st.session_state.messages:
            last_message = st.session_state.messages[-1]
            if isinstance(last_message, AIMessage) and "Final Medical Assessment" in last_message.content:
                st.markdown("---")
                st.markdown("### 🏥 **Final Medical Assessment**")
                st.markdown(last_message.content)
        
        # Input area
        if not st.session_state.conversation_complete:
            st.markdown("---")
            
            # Chat input
            if prompt := st.chat_input("Describe your symptoms or answer the doctor's question..."):
                # Add user message to chat
                st.session_state.messages.append(HumanMessage(content=prompt))
                
                # Process the conversation
                with st.spinner("🤔 Doctor is thinking..."):
                    try:
                        result = process_doctor_conversation(
                            user_input=prompt,
                            conversation_history=st.session_state.messages[:-1]  # Exclude the message we just added
                        )
                        
                        if result["success"]:
                            # Update session state
                            st.session_state.messages = result["messages"]
                            st.session_state.conversation_complete = result["conversation_complete"]
                            st.session_state.patient_info = result["patient_info"]
                            
                            # Rerun to refresh the chat
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
            # Count exchanges
            user_messages = len([m for m in st.session_state.messages if isinstance(m, HumanMessage)])
            ai_messages = len([m for m in st.session_state.messages if isinstance(m, AIMessage)])
            
            st.metric("Questions Asked", ai_messages)
            st.metric("Your Responses", user_messages)
            st.metric("Status", "Complete" if st.session_state.conversation_complete else "In Progress")
            
            # Progress bar
            if not st.session_state.conversation_complete:
                # Estimate progress based on message count
                progress = min(len(st.session_state.messages) / 10, 1.0)
                st.progress(progress)
                st.caption(f"Gathering information... ({int(progress * 100)}%)")
            else:
                st.progress(1.0)
                st.caption("Assessment complete!")
        
        # Quick actions
        st.header("🚀 Quick Start")
        
        if st.button("💬 Start with symptoms", use_container_width=True):
            if not st.session_state.messages:
                st.session_state.messages.append(HumanMessage(content="I'm not feeling well and would like to discuss my symptoms."))
                st.rerun()
        
        if st.button("🏥 Emergency symptoms", use_container_width=True):
            if not st.session_state.messages:
                st.session_state.messages.append(HumanMessage(content="I'm experiencing severe symptoms that I'm concerned about."))
                st.rerun()

if __name__ == "__main__":
    main() 