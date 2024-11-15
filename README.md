---

## **Assignment 4 - End-to-End Research Tool with Multi-Agent System and Document Vectorization**

### **Contributors**:
- Sahiti Nallamolu
- Vishodhan Krishnan
- Vismay Devjee
  
### **Project Resources**:
---
a. **Diagrams**: [Architecture Diagram](https://github.com/BigDataIA-Fall2024-TeamA6/Assignment_4/tree/main/architecture_diagram)  
b. **Fully Documented Code Labs**: [Codelabs Preview](https://codelabs-preview.appspot.com/?file_id=1nxG6yaI3yDjiQZIPSdsig2Jwk55nCH0C4LKvD3lSe8s#0)  
c. **Video of the Submission**: [Solution Video](https://northeastern.zoom.us/rec/share/r0MEH4EqWnbwIpAGH5udZcRMk_HdL4oX0V1C3T267zfe2rMcWxzC7HYy0PB1ryx1.edGMDYD8TQrekura)  
d. **GitHub Project**: [GitHub Repository](https://github.com/BigDataIA-Fall2024-TeamA6/Assignment_4)  
e. **Deployed Project URL**: [https://team6a4.streamlit.app/](https://team6a4.streamlit.app/)

## **Synopsis**

This project creates an end-to-end research tool with a multi-agent system that facilitates document parsing, vector storage, and research-based interaction. By utilizing **Docling**, **Pinecone**, and **Langraph**, the solution offers document-based querying and multi-agent responses for research. The project is containerized and deploys a front-end interface in **Streamlit**, with secure back-end integration for structured, interactive document searches.

### **Technologies Used**
- **Airflow**: for pipeline automation of document parsing and storage.
- **Docling**: for document parsing and text extraction.
- **Pinecone**: for vector storage and document similarity search.
- **Langraph**: to implement a multi-agent system for advanced research capabilities.
- **Streamlit**: for the user interface, allowing interactive document research and Q&A.

## **Problem Statement**

- **Objective**: To develop a research platform that streamlines document processing, vector storage, and retrieval-augmented generation (RAG) using a multi-agent system.
- **Challenges**: Efficient storage and retrieval of document vectors, ensuring accurate and contextually relevant search results.
- **Goal**: Integrate document parsing, vectorization, and user-friendly interfaces for streamlined research interaction.

## **Desired Outcome**

The application allows users to:
- **Select Processed Documents**: Offers parsed documents stored in Pinecone, ready for research.
- **Multi-Agent Research**: Interact with Arxiv and Web Search Agents for comprehensive context on topics.
- **Query Documents**: Use a RAG agent to answer document-specific questions, stored in Pinecone.
- **Generate Reports**: Export research results in both PDF and Codelabs formats for instructional clarity.

## **File Structure**
```
Assignment_4/
  ├── airflow/
  │   ├── dags/
  │   │   └── docling_pinecone_pipeline.py      # Airflow pipeline DAG for document parsing and vector storage  
  │   └── docker-compose.yaml   # Airflow deployment configuration
  ├── architecture_diagram/
  │   ├── arch_diagram.py
  ├── streamlit/
  │   ├── chat.py               #main app
  │   ├── research_agent.py      # Research agent logic with Pinecone indexing and tool routing
  │   ├── state.py               # Defines the state management for the agent
  │   ├── tools.py               # Implements various tools for task execution
  ├── utils/
  │   ├── document_processors.py   # Document processing utilities
  ├── .env                     # load API keys
  ├── requirements.txt               # installation dependencies file
  └── README.md 
```
## **How It Works**

1. **Document Parsing and Vectorization**: 
   - The Airflow pipeline parses documents with Docling, then stores vectors in Pinecone for similarity-based retrieval.
   
2. **Multi-Agent Research**:
   - Users can select documents for querying.
   - The Arxiv and Web Search Agents retrieve relevant external data for research.
   - The RAG Agent provides document-based answers to user questions.

3. **User Interaction Interface**:
   - Streamlit interface lets users ask questions and conduct research.
   - A summary of each research session is saved and can be exported in PDF or Codelabs format.

## **Architecture Diagram**

![Architecture Diagram](https://github.com/BigDataIA-Fall2024-TeamA6/Assignment_4/blob/main/architecture_diagram/end-to-end_research_tool_with_multi-agent_system_&_document_vectorization.png)

---

### **Steps to Run this Application**

1. **Clone this repository** to your local machine:

   ```bash
   git clone https://github.com/BigDataIA-Fall2024-TeamA6/Assignment_4.git
   ```

2. **Install dependencies**:

   ```bash
    pip install -r requirements.txt
   ```

3. **Add credentials** to a `.env` file in `Assignment_4` folder:

   - Pinecone API Key
   - OPENAI API key
   - Tavily API Key
   - SERP API Key

4. **Run the applications**:

   - **Streamlit** for the user interface:

     ```bash
     streamlit run streamlit/chat.py
     ```

5. **Using the Application**:
   - **Research Interaction**: Use the chatbox to interact with multi-agent responses and save research findings.

---

### **References**

1. [Airflow Documentation](https://airflow.apache.org/)
2. [Docling Documentation](https://docling.io/)
3. [Pinecone Documentation](https://www.pinecone.io/docs/)
4. [Langraph Documentation](https://langraph.ai/)
5. [Streamlit Documentation](https://docs.streamlit.io/)

---
