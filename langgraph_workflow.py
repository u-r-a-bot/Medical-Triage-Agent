import os
from typing import Dict, List, Any, TypedDict, Optional, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import json
import re
from rag import get_retriever

class DoctorConversationState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    patient_info: Dict[str, Any]
    symptoms_analyzed: bool
    conversation_complete: bool
    final_recommendation: str
    current_agent: str

def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        google_api_key=api_key
    )

retriever = get_retriever()

def extract_patient_info(messages: List[Any]) -> Dict[str, Any]:
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
    
    age_match = re.search(r'(\d+)\s*(?:years?\s*old|y\.?o\.?)', conversation_lower)
    if age_match:
        patient_info["age"] = int(age_match.group(1))
    
    if any(word in conversation_lower for word in ["male", "man", "boy"]):
        patient_info["gender"] = "male"
    elif any(word in conversation_lower for word in ["female", "woman", "girl"]):
        patient_info["gender"] = "female"
    
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
    
    if any(word in conversation_lower for word in ['severe', 'very bad', 'terrible', 'awful']):
        patient_info["severity"] = "severe"
    elif any(word in conversation_lower for word in ['moderate', 'bad', 'uncomfortable']):
        patient_info["severity"] = "moderate"
    elif any(word in conversation_lower for word in ['mild', 'slight', 'okay', 'manageable']):
        patient_info["severity"] = "mild"
    
    return patient_info

def initial_assessment_agent(state: DoctorConversationState) -> DoctorConversationState:
    messages = state["messages"]
    
    if len(messages) == 1:
        patient_info = extract_patient_info(messages)
        
        assessment_prompt = """You are a medical assistant conducting an initial assessment. 
        Analyze the patient's first message and provide a welcoming, professional response.
        Ask one relevant follow-up question to gather more information about their symptoms."""
        
        llm = get_llm()
        response = llm.invoke(assessment_prompt + f"\n\nPatient message: {messages[-1].content}")
        
        messages.append(AIMessage(content=str(response.content)))
        
        return {
            **state,
            "messages": messages,
            "patient_info": patient_info,
            "current_agent": "conversation_agent"
        }
    
    return state

def conversation_agent(state: DoctorConversationState) -> DoctorConversationState:
    messages = state["messages"]
    
    patient_info = extract_patient_info(messages)
    
    if should_continue_conversation(messages):
        system_prompt = """You are a caring and professional medical assistant conducting a patient consultation. 
        Ask one natural, conversational follow-up question to understand the patient's condition better."""
        
        llm = get_llm()
        conversation_text = " ".join([msg.content for msg in messages if hasattr(msg, 'content')])
        
        response = llm.invoke(system_prompt + f"\n\nConversation context: {conversation_text}")
        
        messages.append(AIMessage(content=str(response.content)))
        
        return {
            **state,
            "messages": messages,
            "patient_info": patient_info,
            "current_agent": "conversation_agent"
        }
    else:
        return {
            **state,
            "messages": messages,
            "patient_info": patient_info,
            "current_agent": "symptom_analysis_agent"
        }

def symptom_analysis_agent(state: DoctorConversationState) -> DoctorConversationState:
    messages = state["messages"]
    patient_info = state["patient_info"]
    symptoms = patient_info["symptoms"]
    
    if symptoms:
        symptoms_text = " ".join(symptoms)
        
        try:
            context_docs = retriever.invoke(symptoms_text)
            medical_context = "\n\n".join([doc.page_content for doc in context_docs])
        except Exception as e:
            medical_context = "Unable to retrieve specific medical context at this time."
        
        analysis_prompt = f"""As a medical professional, analyze the patient's symptoms in the context of available medical information.

Patient Symptoms: {', '.join(symptoms)}
Medical Context: {medical_context}

Provide a brief analysis of what these symptoms might indicate."""
        
        llm = get_llm()
        response = llm.invoke(analysis_prompt)
        
        messages.append(AIMessage(content=f"**Symptom Analysis:** {str(response.content)}"))
    
    return {
        **state,
        "messages": messages,
        "symptoms_analyzed": True,
        "current_agent": "recommendation_agent"
    }

def recommendation_agent(state: DoctorConversationState) -> DoctorConversationState:
    messages = state["messages"]
    patient_info = state["patient_info"]
    symptoms = patient_info["symptoms"]
    
    symptoms_text = " ".join(symptoms) if symptoms else "general symptoms"
    
    try:
        context_docs = retriever.invoke(symptoms_text)
        medical_context = "\n\n".join([doc.page_content for doc in context_docs])
    except Exception as e:
        medical_context = "Unable to retrieve specific medical context."
    
    recommendation_prompt = f"""Based on the complete patient consultation and medical knowledge, provide a comprehensive triage recommendation.

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
    response = llm.invoke(recommendation_prompt)
    
    messages.append(AIMessage(content=f"**Final Medical Assessment:**\n\n{str(response.content)}"))
    
    return {
        **state,
        "messages": messages,
        "conversation_complete": True,
        "final_recommendation": str(response.content),
        "current_agent": "end"
    }

def should_continue_conversation(messages: List[Any]) -> bool:
    conversation_text = " ".join([msg.content for msg in messages if hasattr(msg, 'content')])
    conversation_lower = conversation_text.lower()
    
    has_symptoms = any(word in conversation_lower for word in ['pain', 'fever', 'headache', 'cough', 'nausea', 'vomiting', 'dizziness', 'shortness of breath', 'chest pain', 'abdominal pain', 'fatigue', 'sore throat', 'runny nose', 'congestion', 'diarrhea', 'constipation', 'rash', 'swelling', 'bleeding', 'bruising', 'weakness', 'numbness', 'sweating', 'chills', 'loss of appetite', 'weight loss', 'weight gain'])
    has_duration = any(word in conversation_lower for word in ['hours', 'days', 'weeks', 'months', 'yesterday', 'today', 'morning', 'evening', 'last night', 'this morning', 'this evening'])
    has_severity = any(word in conversation_lower for word in ['severe', 'mild', 'moderate', 'bad', 'terrible', 'okay', 'manageable', 'uncomfortable', 'slight', 'awful'])
    
    enough_info = has_symptoms
    enough_exchanges = len(messages) >= 6
    
    if has_symptoms and has_duration and has_severity:
        enough_exchanges = len(messages) >= 4
    
    too_many_exchanges = len(messages) >= 12
    
    return not (enough_info and enough_exchanges) and not too_many_exchanges

def route_to_next_agent(state: DoctorConversationState) -> str:
    return state["current_agent"]

def create_triage_workflow():
    workflow = StateGraph(DoctorConversationState)
    
    workflow.add_node("initial_assessment", initial_assessment_agent)
    workflow.add_node("conversation_agent", conversation_agent)
    workflow.add_node("symptom_analysis", symptom_analysis_agent)
    workflow.add_node("recommendation_agent", recommendation_agent)
    
    workflow.add_edge("initial_assessment", "conversation_agent")
    workflow.add_edge("conversation_agent", "symptom_analysis")
    workflow.add_edge("symptom_analysis", "recommendation_agent")
    workflow.add_edge("recommendation_agent", END)
    
    workflow.add_conditional_edges(
        "conversation_agent",
        route_to_next_agent,
        {
            "conversation_agent": "conversation_agent",
            "symptom_analysis": "symptom_analysis"
        }
    )
    
    workflow.set_entry_point("initial_assessment")
    
    return workflow.compile()

def visualize_workflow():
    workflow = create_triage_workflow()
    
    graph_image = workflow.get_graph().draw_mermaid_png()
    
    with open("triage_workflow.png", "wb") as f:
        f.write(graph_image)
    
    print("✅ Workflow visualization saved as 'triage_workflow.png'")
    return "triage_workflow.png"

def process_conversation_with_workflow(user_input: str, conversation_history: Optional[List[Any]] = None) -> Dict[str, Any]:
    if conversation_history is None:
        conversation_history = []
    
    messages = conversation_history + [HumanMessage(content=user_input)]
    
    initial_state = DoctorConversationState(
        messages=messages,
        patient_info={},
        symptoms_analyzed=False,
        conversation_complete=False,
        final_recommendation="",
        current_agent="initial_assessment"
    )
    
    workflow = create_triage_workflow()
    
    try:
        result = workflow.invoke(initial_state)
        
        return {
            "success": True,
            "messages": result["messages"],
            "patient_info": result["patient_info"],
            "conversation_complete": result["conversation_complete"],
            "final_recommendation": result["final_recommendation"]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    visualize_workflow() 