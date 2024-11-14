import streamlit as st
from research_agent import get_graph

def get_ai_response(question):
    ai_agent = get_graph()
    output = ai_agent.invoke({
        "input": question,
        "chat_history":[]
    })
    
    return output
 
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
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_ai_response(prompt)
                intermediate_steps = response.get('intermediate_steps', [])
                last_step = intermediate_steps[-1]
                # # Access the tool_input dictionary
                tool_input = last_step.tool_input
                # # Extract conclusion and sources
                conclusion = tool_input['conclusion']
                # introduction = tool_input['introduction']
                # sources = tool_input['sources']
                # first_tool = intermediate_steps[0].tool  # This would give you 'web_search'
                # print(intermediate_steps)
                # print('introduction:', introduction)
                # print('conclusion:', conclusion)
                # print('sources:', sources)
                # print('first_tool:', first_tool)

                # Get the last step's tool_input for the main content
                last_step = intermediate_steps[-1].tool_input

                # Get the first tool
                first_tool = intermediate_steps[0].tool

                formatted_output = f"""
                **INTRODUCTION:** 

{last_step["introduction"]}  


**RESEARCH STEPS**

{last_step["research_steps"]}


**REPORT** 

{last_step["main_body"]}


**SOURCES**

{last_step["sources"]}

**TOOL USED:** {first_tool}
                """
                st.session_state.formatted_output = formatted_output
                print(formatted_output)

                st.markdown(conclusion)
       
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
 
    # Sidebar with chat controls
    with st.sidebar:
        st.title("Chat Controls")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
       
        st.markdown("---")
        st.markdown("### Current Response Details")
        if st.session_state.formatted_output:
            st.markdown(st.session_state.formatted_output)
        
 
if __name__ == "__main__":
    # Set page configuration
    st.set_page_config(
        page_title="AI Chat Assistant",
        page_icon="💬",
        layout="wide"
    )
    main()
