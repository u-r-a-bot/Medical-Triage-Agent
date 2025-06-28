#!/usr/bin/env python3
"""
Display the LangGraph workflow structure for the medical triage system.
"""

from langgraph_workflow import create_triage_workflow

def display_workflow_structure():
    """Display the workflow structure in a readable format."""
    
    print("🏥 Medical Triage Agent - LangGraph Workflow Structure")
    print("=" * 60)
    
    # Create the workflow
    workflow = create_triage_workflow()
    graph = workflow.get_graph()
    
    print("\n📋 Workflow Nodes:")
    print("-" * 30)
    
    # Display nodes
    nodes = list(graph.nodes.keys())
    for i, node in enumerate(nodes, 1):
        print(f"{i}. {node}")
    
    print("\n🔄 Workflow Flow:")
    print("-" * 30)
    print("1. __start__ → Initial Assessment")
    print("2. Initial Assessment → Conversation Agent")
    print("3. Conversation Agent → Symptom Analysis (when enough info gathered)")
    print("4. Conversation Agent → Conversation Agent (for follow-up questions)")
    print("5. Symptom Analysis → Recommendation Agent")
    print("6. Recommendation Agent → __end__")
    
    print("\n🤖 Agent Functions:")
    print("-" * 30)
    print("• Initial Assessment: Analyzes first patient input and asks initial question")
    print("• Conversation Agent: Asks intelligent follow-up questions based on context")
    print("• Symptom Analysis: Uses RAG to analyze symptoms with medical knowledge")
    print("• Recommendation Agent: Generates final triage recommendation")
    
    print("\n📊 State Management:")
    print("-" * 30)
    print("• messages: Conversation history with add_messages annotation")
    print("• patient_info: Extracted patient information (age, gender, symptoms, etc.)")
    print("• symptoms_analyzed: Boolean flag for symptom analysis completion")
    print("• conversation_complete: Boolean flag for conversation completion")
    print("• final_recommendation: Generated triage recommendation")
    print("• current_agent: Tracks which agent is currently active")
    
    print("\n🔄 Conditional Routing:")
    print("-" * 30)
    print("• Conversation Agent uses conditional edges to decide:")
    print("  - Continue asking questions (loop back to conversation_agent)")
    print("  - Move to symptom analysis (when enough information is gathered)")
    print("• Decision based on: symptoms present, conversation length, and information completeness")
    
    print("\n✅ Workflow visualization saved as 'triage_workflow.png'")
    print("📁 You can view the PNG file to see the visual representation of the graph.")
    
    print("\n🔧 LangGraph Features Used:")
    print("-" * 30)
    print("• StateGraph: For managing conversation state")
    print("• add_messages: For automatic message history management")
    print("• Conditional Edges: For dynamic routing based on conversation state")
    print("• Multiple Agents: Each with specific responsibilities")
    print("• RAG Integration: In symptom analysis and recommendation agents")

if __name__ == "__main__":
    display_workflow_structure() 