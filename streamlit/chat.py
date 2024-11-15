import streamlit as st
from research_agent import get_graph
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
from jinja2 import Environment, FileSystemLoader

def get_ai_response(question):
    ai_agent = get_graph()
    output = ai_agent.invoke({
        "input": question,
        "chat_history":[]
    })
    
    return output
 

def build_html_report(output):
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('report_template.html')

    research_steps = output.get("research_steps", [])
    if isinstance(research_steps, list):
        research_steps = "<br>".join([f"- {step}" for step in research_steps])
    else:
        research_steps = ""

    sources = output.get("sources", [])
    if isinstance(sources, list):
        sources = "<br>".join([f"- {source}" for source in sources])
    else:
        sources = ""

    context = {
        "introduction": output.get("introduction", ""),
        "research_steps": research_steps,
        "main_body": output.get("main_body", ""),
        "conclusion": output.get("conclusion", ""),
        "sources": sources
    }

    html_content = template.render(context)
    return html_content

def build_report(output):
    research_steps = output["research_steps"]
    if isinstance(research_steps, list):
        research_steps = "\n".join([f"- {step}" for step in research_steps])
    
    sources = output["sources"]
    if isinstance(sources, list):
        sources = "\n".join([f"- {source}" for source in sources])
    
    return f"""
**INTRODUCTION**
{output["introduction"]}

**RESEARCH STEPS**
{research_steps}

**REPORT**
{output["main_body"]}

**CONCLUSION**
{output["conclusion"]}

**SOURCES**
{sources}
""" 
 
def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'formatted_output' not in st.session_state:
        st.session_state.formatted_output = None

def generate_pdf(chat_history):
    # Create a BytesIO buffer to hold PDF data
    pdf_buffer = io.BytesIO()
    pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
    
    # Set the PDF title
    pdf.setTitle("Chat History")

    # Define text properties
    width, height = letter
    text = pdf.beginText(40, height - 40)
    text.setFont("Helvetica", 10)

    # Add chat history to PDF content
    for entry in chat_history:
        role = entry["role"].capitalize()
        content = entry["content"]
        text.textLine(f"{role}: {content}")
        text.textLine("")  # Empty line for separation

    # Finalize and save the PDF content
    pdf.drawText(text)
    pdf.showPage()
    pdf.save()
    pdf_buffer.seek(0)
    return pdf_buffer
 
def main():
    st.title("AI Chat Assistant")
   
    # Initialize session state
    initialize_session_state()
   
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
   
    # Check if chat history is empty to show welcome message
    if len(st.session_state.messages) == 0:
        WELCOME = ("Hello! I'm an AI-powered research assistant equipped with advanced tools. Ask me anything!")
        with st.chat_message("ai"):
            st.write(WELCOME)
            # Add welcome message to chat history
            st.session_state.messages.append({"role": "ai", "content": WELCOME})

    if prompt := st.chat_input("Ask your question here..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
       
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
       
        # Display assistant response with a loading spinner
        response = get_ai_response(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                intermediate_steps = response.get('intermediate_steps', [])
                last_step = intermediate_steps[-1]

                tool_input = last_step.tool_input
                main_body = tool_input['main_body']

                # Get the last step's tool_input for the main content
                last_step = intermediate_steps[-1].tool_input

                # Get the tools used
                tools_used = list()
                for step in intermediate_steps:
                    tools_used.append(step.tool)
                first_tool = intermediate_steps[0].tool

                formatted_output = f"""
                **INTRODUCTION:** 

{last_step["introduction"]}  


**RESEARCH STEPS**

{last_step["research_steps"]}


**CONCLUSION** 

{last_step["conclusion"]}


**SOURCES**

{last_step["sources"]}

**TOOL USED:** {first_tool}
                """
                st.session_state.formatted_output = formatted_output
                print(formatted_output)

                st.markdown(main_body)
       
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
 
    # Sidebar with chat controls
    with st.sidebar:
        st.title("Chat Controls")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("Create HTML"):
            response = get_ai_response(prompt)
            html_content = build_html_report(response)
            st.download_button(
                label="Download HTML",
                data=html_content,
                file_name="chat_report.html",
                mime="text/html"
            )

        # Download PDF button
        if st.button("Create PDF"):
            pdf_buffer = generate_pdf(st.session_state.messages)
            st.download_button(
                label="Download PDF",
                data=pdf_buffer,
                file_name="chat_history.pdf",
                mime="application/pdf"
            )

        st.markdown("---")
        st.markdown("### Current Response Details")
        if st.session_state.formatted_output:
            st.markdown(st.session_state.formatted_output)
        
        st.markdown("---")
        st.markdown("### Chat History")
        st.write(st.session_state.messages)

if __name__ == "__main__":
    # Set page configuration
    st.set_page_config(
        page_title="AI Chat Assistant",
        page_icon="💬",
        layout="wide"
    )
    main()