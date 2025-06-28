# Clinical Triage Agent

AI-powered medical triage system using LangGraph multi-agent workflow with RAG capabilities.

## Features

- 🤖 **AI Symptom Analysis**: Identifies and categorizes patient symptoms
- 🔍 **RAG Integration**: Retrieves medical information from knowledge base
- 💡 **Triage Recommendations**: Provides guidance (self-care, primary care, urgent care, emergency)
- 🎨 **Streamlit Interface**: Modern web interface with real-time analysis
- 🔄 **Multi-Agent Workflow**: Specialized agents for conversation, analysis, and recommendations

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment**:
   ```bash
   echo "GOOGLE_API_KEY=your_api_key_here" > .env
   ```

3. **Build knowledge base**:
   ```bash
   python prep_dataset.py
   python rag.py
   ```

4. **Run the app**:
   ```bash
   streamlit run app.py
   ```

## Architecture

**Multi-Agent Workflow:**
- **Conversation Agent**: Asks follow-up questions
- **Symptom Analysis Agent**: Uses RAG to analyze symptoms
- **Recommendation Agent**: Generates triage recommendations

## Knowledge Base

41 medical conditions including cardiovascular, respiratory, infectious, and endocrine disorders.

## Triage Levels

1. **Self-care**: Mild symptoms
2. **Primary Care**: Moderate symptoms
3. **Urgent Care**: Severe symptoms
4. **Emergency**: Critical symptoms

## File Structure

```
├── app.py                 # Streamlit web app
├── triage_agent.py        # Multi-agent system
├── langgraph_workflow.py  # LangGraph implementation
├── rag.py                 # RAG functionality
├── prep_dataset.py        # Knowledge base setup
├── knowledge_base/        # Medical knowledge files
└── vector_store/          # FAISS vector database
```

## Usage

### Web Interface
```bash
streamlit run app.py
```

### Programmatic
```python
from triage_agent import process_triage_request
result = process_triage_request("severe chest pain")
```

### Workflow Visualization
```bash
python langgraph_workflow.py  # Creates triage_workflow.png
```

## Safety

⚠️ **Educational use only**. Not a substitute for professional medical advice. Always consult healthcare professionals.

## Tech Stack

- **LangChain** + **LangGraph**: Multi-agent workflows
- **Streamlit**: Web interface
- **FAISS**: Vector database
- **Google Gemini**: LLM provider

## License

MIT License 