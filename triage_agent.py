import os
from typing import Dict, List, Any, TypedDict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import json
import re
from rag import get_retriever

# Define the conversation state
class DoctorConversationState(TypedDict):
    messages: List[Any]
    patient_info: Dict[str, Any]
    symptoms_analyzed: bool
    conversation_complete: bool
    final_recommendation: str

def get_llm():
    """Get the LLM instance with proper API key."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,  # Higher temperature for more natural conversation
        google_api_key=api_key
    )

# Initialize the retriever
retriever = get_retriever()

def extract_patient_info(messages: List[Any]) -> Dict[str, Any]:
    """Extract patient information from conversation messages."""
    patient_info = {
        "age": None,
        "gender": None,
        "symptoms": [],
        "duration": None,
        "severity": None,
        "medical_history": [],
        "medications": [],
        "allergies": []
    }
    
    conversation_text = " ".join([msg.content for msg in messages if hasattr(msg, 'content')])
    conversation_lower = conversation_text.lower()
    
    # Extract age
    age_match = re.search(r'(\d+)\s*(?:years?\s*old|y\.?o\.?)', conversation_lower)
    if age_match:
        patient_info["age"] = int(age_match.group(1))
    
    # Extract gender
    if any(word in conversation_lower for word in ["male", "man", "boy"]):
        patient_info["gender"] = "male"
    elif any(word in conversation_lower for word in ["female", "woman", "girl"]):
        patient_info["gender"] = "female"
    
    # Extract symptoms
    symptom_keywords = [
        'pain', 'fever', 'headache', 'cough', 'nausea', 'vomiting', 'dizziness',
        'shortness of breath', 'chest pain', 'abdominal pain', 'fatigue',
        'sore throat', 'runny nose', 'congestion', 'diarrhea', 'constipation',
        'rash', 'swelling', 'bleeding', 'bruising', 'weakness', 'numbness',
        'sweating', 'chills', 'loss of appetite', 'weight loss', 'weight gain'
    ]
    
    for keyword in symptom_keywords:
        if keyword in conversation_lower:
            patient_info["symptoms"].append(keyword)
    
    # Extract duration
    duration_patterns = [
        r'(\d+)\s*(?:hours?|hrs?)',
        r'(\d+)\s*(?:days?|d)',
        r'(\d+)\s*(?:weeks?|wks?)',
        r'(\d+)\s*(?:months?|mos?)',
        r'(yesterday|today|this morning|this evening|last night)'
    ]
    
    for pattern in duration_patterns:
        match = re.search(pattern, conversation_lower)
        if match:
            patient_info["duration"] = match.group(0)
            break
    
    # Extract severity
    if any(word in conversation_lower for word in ['severe', 'very bad', 'terrible', 'awful']):
        patient_info["severity"] = "severe"
    elif any(word in conversation_lower for word in ['moderate', 'bad', 'uncomfortable']):
        patient_info["severity"] = "moderate"
    elif any(word in conversation_lower for word in ['mild', 'slight', 'okay', 'manageable']):
        patient_info["severity"] = "mild"
    
    return patient_info

def doctor_agent_ask_question(messages: List[Any]) -> str:
    """Doctor agent that asks intelligent follow-up questions based on conversation context."""
    
    # Get conversation context
    conversation_text = " ".join([msg.content for msg in messages if hasattr(msg, 'content')])
    
    # Create a system prompt for the doctor agent
    system_prompt = """You are a caring and professional doctor conducting a patient consultation. Your role is to ask natural, conversational follow-up questions to understand the patient's condition better.

Guidelines:
- Be empathetic and professional
- Ask one clear question at a time
- Base your questions on what the patient has already told you
- Focus on gathering essential information: symptoms, duration, severity, medical history
- Use natural, conversational language
- Don't ask questions the patient has already answered

Current conversation context:
{conversation_context}

Based on this conversation, what would you like to ask the patient next? Ask only one question."""

    llm = get_llm()
    
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "What should I ask the patient next?")
        ])
        
        response = llm.invoke(prompt.format(conversation_context=conversation_text))
        return str(response.content) if hasattr(response, 'content') else str(response)
        
    except Exception as e:
        # Fallback questions
        fallback_questions = [
            "Can you tell me more about your symptoms?",
            "How long have you been experiencing these symptoms?",
            "How severe would you rate these symptoms?",
            "Do you have any relevant medical history?",
            "Are you currently taking any medications?"
        ]
        
        # Choose appropriate fallback based on conversation length
        question_index = min(len(messages) // 2, len(fallback_questions) - 1)
        return fallback_questions[question_index]

def symptom_analysis_agent(messages: List[Any]) -> str:
    """Symptom analysis agent that uses RAG to analyze symptoms and provide medical context."""
    
    # Extract symptoms
    patient_info = extract_patient_info(messages)
    symptoms = patient_info["symptoms"]
    
    if not symptoms:
        return "I need to understand your symptoms better to provide a proper assessment."
    
    # Get medical context using RAG
    symptoms_text = " ".join(symptoms)
    
    try:
        context_docs = retriever.invoke(symptoms_text)
        medical_context = "\n\n".join([doc.page_content for doc in context_docs])
    except Exception as e:
        medical_context = "Unable to retrieve specific medical context at this time."
    
    # Analyze symptoms with medical context
    analysis_prompt = f"""As a medical professional, analyze the patient's symptoms in the context of available medical information.

Patient Symptoms: {', '.join(symptoms)}
Medical Context: {medical_context}

Provide a brief analysis of what these symptoms might indicate and any concerning patterns."""

    llm = get_llm()
    
    try:
        response = llm.invoke(analysis_prompt)
        return str(response.content) if hasattr(response, 'content') else str(response)
    except Exception as e:
        return f"Based on your symptoms ({', '.join(symptoms)}), I need to gather more information to provide a proper assessment."

def generate_final_recommendation(messages: List[Any]) -> str:
    """Generate comprehensive final triage recommendation."""
    
    conversation_text = " ".join([msg.content for msg in messages if hasattr(msg, 'content')])
    patient_info = extract_patient_info(messages)
    symptoms = patient_info["symptoms"]
    
    # Get medical context
    symptoms_text = " ".join(symptoms) if symptoms else "general symptoms"
    
    try:
        context_docs = retriever.invoke(symptoms_text)
        medical_context = "\n\n".join([doc.page_content for doc in context_docs])
    except Exception as e:
        medical_context = "Unable to retrieve specific medical context."
    
    # Generate comprehensive recommendation
    recommendation_prompt = f"""Based on the complete patient consultation and medical knowledge, provide a comprehensive triage recommendation.

Patient Consultation:
{conversation_text}

Patient Information:
- Age: {patient_info.get('age', 'Not specified')}
- Gender: {patient_info.get('gender', 'Not specified')}
- Symptoms: {', '.join(symptoms) if symptoms else 'Not specified'}
- Duration: {patient_info.get('duration', 'Not specified')}
- Severity: {patient_info.get('severity', 'Not specified')}

Medical Context:
{medical_context}

Please provide:
1. **Triage Recommendation**: (self-care, primary care, urgent care, or emergency)
2. **Detailed Reasoning**: Explain why this recommendation is appropriate
3. **Immediate Actions**: What the patient should do right now
4. **Red Flags**: Warning signs to watch for
5. **Follow-up Plan**: When and how to follow up
6. **Precautions**: Any specific precautions to take

Be thorough, professional, and prioritize patient safety."""

    llm = get_llm()
    
    try:
        response = llm.invoke(recommendation_prompt)
        return str(response.content) if hasattr(response, 'content') else str(response)
    except Exception as e:
        return f"Error generating recommendation: {str(e)}"

def should_continue_conversation(messages: List[Any]) -> bool:
    """Determine if the conversation should continue or conclude."""
    
    conversation_text = " ".join([msg.content for msg in messages if hasattr(msg, 'content')])
    conversation_lower = conversation_text.lower()
    
    # Check if we have enough information
    has_symptoms = any(word in conversation_lower for word in ['pain', 'fever', 'headache', 'cough', 'nausea', 'vomiting', 'dizziness', 'shortness of breath', 'chest pain', 'abdominal pain', 'fatigue', 'sore throat', 'runny nose', 'congestion', 'diarrhea', 'constipation', 'rash', 'swelling', 'bleeding', 'bruising', 'weakness', 'numbness', 'sweating', 'chills', 'loss of appetite', 'weight loss', 'weight gain'])
    has_duration = any(word in conversation_lower for word in ['hours', 'days', 'weeks', 'months', 'yesterday', 'today', 'morning', 'evening', 'last night', 'this morning', 'this evening'])
    has_severity = any(word in conversation_lower for word in ['severe', 'mild', 'moderate', 'bad', 'terrible', 'okay', 'manageable', 'uncomfortable', 'slight', 'awful'])
    
    # More lenient criteria - conclude if we have basic info and reasonable conversation length
    enough_info = has_symptoms  # Just need symptoms as minimum
    enough_exchanges = len(messages) >= 6  # At least 3 Q&A pairs (6 messages)
    
    # Additional check: if we have symptoms + duration + severity, conclude sooner
    if has_symptoms and has_duration and has_severity:
        enough_exchanges = len(messages) >= 4  # Only need 2 Q&A pairs if we have all key info
    
    # Force conclusion after too many exchanges (prevent infinite loops)
    too_many_exchanges = len(messages) >= 12  # Max 6 Q&A pairs
    
    return not (enough_info and enough_exchanges) and not too_many_exchanges

def process_doctor_conversation(user_input: str, conversation_history: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Process one step of the doctor-patient conversation."""
    
    # Initialize conversation history
    if conversation_history is None:
        conversation_history = []
    
    # Add user input to history
    conversation_history.append(HumanMessage(content=user_input))
    
    # Extract patient info
    patient_info = extract_patient_info(conversation_history)
    
    # Check if conversation should continue
    if should_continue_conversation(conversation_history):
        # Doctor agent asks next question
        next_question = doctor_agent_ask_question(conversation_history)
        conversation_history.append(AIMessage(content=next_question))
        
        return {
            "success": True,
            "messages": conversation_history,
            "patient_info": patient_info,
            "symptoms_analyzed": False,
            "conversation_complete": False,
            "final_recommendation": "",
            "next_question": next_question
        }
    else:
        # Generate final recommendation
        final_recommendation = generate_final_recommendation(conversation_history)
        conversation_history.append(AIMessage(content=f"**Final Medical Assessment:**\n{final_recommendation}"))
        
        return {
            "success": True,
            "messages": conversation_history,
            "patient_info": patient_info,
            "symptoms_analyzed": True,
            "conversation_complete": True,
            "final_recommendation": final_recommendation,
            "next_question": ""
        }

# Legacy functions for backward compatibility
def process_triage_request(patient_description: str) -> Dict[str, Any]:
    """Legacy function - now uses the new doctor conversation system."""
    return process_doctor_conversation(patient_description)

def process_chat_triage_request(patient_description: str, conversation_history: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Legacy function - now uses the new doctor conversation system."""
    return process_doctor_conversation(patient_description, conversation_history)

def process_conversation_step(user_input: str, conversation_history: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Legacy function - now uses the new doctor conversation system."""
    return process_doctor_conversation(user_input, conversation_history)

if __name__ == "__main__":
    # Test the triage agent
    test_case = "I have a severe headache, fever of 102°F, and neck stiffness"
    result = process_triage_request(test_case)
    print(json.dumps(result, indent=2)) 