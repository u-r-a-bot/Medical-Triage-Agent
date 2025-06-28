# Clinical Triage Agent

A sophisticated clinical triage system built with LangGraph and Streamlit that uses AI to analyze patient symptoms, assess severity, and provide triage recommendations with RAG (Retrieval-Augmented Generation) capabilities.

## Features

- 🤖 **AI-Powered Symptom Analysis**: Automatically identifies and categorizes patient symptoms
- 📊 **Severity & Urgency Assessment**: Determines the severity level and urgency of medical conditions
- 🔍 **RAG Integration**: Retrieves relevant medical information from a comprehensive knowledge base
- 💡 **Intelligent Recommendations**: Provides appropriate triage recommendations (self-care, primary care, urgent care, emergency)
- 🎨 **Modern Web Interface**: Beautiful Streamlit interface with real-time analysis
- 🔄 **LangGraph Workflow**: Structured decision-making process using LangGraph

## Architecture

The system uses a LangGraph workflow with the following components:

1. **Symptom Analysis Node**: Analyzes patient descriptions to extract symptoms and assess severity
2. **Context Retrieval Node**: Uses RAG to retrieve relevant medical information from the knowledge base
3. **Recommendation Node**: Generates triage recommendations based on symptoms and medical context

## Prerequisites

- Python 3.8+
- Google Gemini API key
- Internet connection for model downloads

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd medical-triage-agent
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Prepare the knowledge base** (if not already done):
   ```bash
   python prep_dataset.py
   ```

5. **Build the vector store**:
   ```bash
   python rag.py
   ```

## Usage

### Running the Web Application

Start the Streamlit application:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Using the Triage Agent Programmatically

```python
from triage_agent import process_triage_request

# Example usage
result = process_triage_request("Patient reports severe chest pain and shortness of breath")
print(result)
```

## Knowledge Base

The system includes a comprehensive knowledge base with information about various medical conditions:

- **Diseases Covered**: 41 different medical conditions including:
  - Cardiovascular (heart attack, hypertension)
  - Respiratory (pneumonia, asthma, tuberculosis)
  - Infectious (malaria, dengue, typhoid)
  - Endocrine (diabetes, thyroid disorders)
  - And many more...

- **Data Source**: The knowledge base is derived from a medical dataset containing symptom-disease mappings

## Triage Recommendations

The system provides four levels of triage recommendations:

1. **Self-care**: Mild symptoms manageable at home
2. **Primary Care**: Moderate symptoms requiring medical attention within days
3. **Urgent Care**: Severe symptoms requiring prompt medical attention
4. **Emergency**: Critical symptoms requiring immediate emergency care

## Safety and Disclaimers

⚠️ **Important**: This system is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.

- Always consult qualified healthcare professionals for medical decisions
- The system may not be accurate for all medical conditions
- Emergency situations should be handled by calling emergency services immediately

## Technical Details

### Dependencies

- **LangChain**: Framework for building LLM applications
- **LangGraph**: For creating structured workflows
- **Streamlit**: Web interface framework
- **FAISS**: Vector database for similarity search
- **Sentence Transformers**: For text embeddings
- **Google Gemini**: LLM provider

### File Structure

```
medical-triage-agent/
├── app.py                 # Streamlit web application
├── triage_agent.py        # Main LangGraph triage agent
├── rag.py                 # RAG implementation
├── prep_dataset.py        # Knowledge base preparation
├── requirements.txt       # Python dependencies
├── knowledge_base/        # Medical knowledge files
├── vector_store/          # FAISS vector database
└── Training.csv           # Original dataset
```

## Customization

### Adding New Medical Conditions

1. Add new condition files to the `knowledge_base/` directory
2. Rebuild the vector store: `python rag.py`

### Modifying the Workflow

Edit `triage_agent.py` to modify the LangGraph workflow:
- Add new nodes for additional processing steps
- Modify the decision logic in routing functions
- Update the state structure for new data fields

### Customizing the Interface

Modify `app.py` to:
- Add new input fields
- Change the styling and layout
- Add additional features like patient history tracking

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your Gemini API key is correctly set in the `.env` file
2. **Vector Store Not Found**: Run `python rag.py` to build the vector store
3. **Import Errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`

### Performance Optimization

- The system uses FAISS for efficient similarity search
- Consider using a GPU for faster embedding generation
- Adjust the number of retrieved documents in `rag.py` based on your needs

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Medical dataset from Kaggle
- LangChain and LangGraph communities
- Google for providing the Gemini language models 