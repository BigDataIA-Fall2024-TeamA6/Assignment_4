from diagrams import Diagram, Cluster, Edge
from diagrams.aws.storage import S3
from diagrams.custom import Custom
from diagrams.onprem.workflow import Airflow

# Diagram setup
with Diagram("End-to-End Research Tool with Multi-Agent System & Document Vectorization", show=False):
    
    
    with Cluster("Apache Airflow Orchestrator"):
        
        pdf = Custom("PDF", "D:/DAMG7245/Assignment_4/architecture_diagram/pdf.png")  
        dockling = Custom("", "D:/DAMG7245/Assignment_4/architecture_diagram/docling.png")  
        pinecone = Custom("PineCone", "D:/DAMG7245/Assignment_4/architecture_diagram/pinecone.png")
        
        
        pdf >> Edge(color='black') >> dockling
        dockling >> Edge(color='black') >> pinecone



    with Cluster("Client-Facing Application"):
        streamlit = Custom("Streamlit", "D:/DAMG7245/Assignment_4/architecture_diagram/streamlit.png")  
        q_a_interface = Custom("Q&A Interface", "D:/DAMG7245/Assignment_4/architecture_diagram/q&a.png")  

    pinecone >> Edge(color='black') >> streamlit
    pinecone >> Edge(color='slateblue') >> q_a_interface
    
    with Cluster("AI Agent Framework", direction="BT"):
        langgraph = Custom("AI Agent", "D:/DAMG7245/Assignment_4/architecture_diagram/langgraph.png")  
        rag_search = Custom("RAG_Search", "D:/DAMG7245/Assignment_4/architecture_diagram/rag.png") 
        web = Custom("Web Search", "D:/DAMG7245/Assignment_4/architecture_diagram/web.png")
        arxiv = Custom("Research Paper db", "D:/DAMG7245/Assignment_4/architecture_diagram/arxiv.png")

    streamlit << Edge(color='black') << langgraph
    streamlit >> Edge(color='black') >> langgraph
    langgraph >> Edge(color="red", label="Routes to") >> [rag_search, web, arxiv]

    with Cluster("Print files"):
        pdf2 = Custom("PDF", "D:/DAMG7245/Assignment_4/architecture_diagram/pdf.png")
        codelabs = Custom("CodeLabs", "D:/DAMG7245/Assignment_4/architecture_diagram/codelabs.png")
    # Q&A interactions with PDF
    q_a_interface >> Edge(color='black') >> pdf2
    q_a_interface >> Edge(color='black') >> codelabs