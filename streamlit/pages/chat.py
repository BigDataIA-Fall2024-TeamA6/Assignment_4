import streamlit as st
from research_agent import get_graph
 
def get_ai_response(question):
    ai_agent = get_graph()
    output = ai_agent.invoke({
        "input": question,
        "chat_history":[]
    })
    
    return output
 
 
def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
 
def main():
    st.title("AI Chat Assistant")
   
    # Initialize session state
    initialize_session_state()
   
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
   
    # Chat input
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
                st.markdown(response)
       
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
 
    # Sidebar with chat controls
    with st.sidebar:
        st.title("Chat Controls")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
       
        st.markdown("---")
        st.markdown("### Chat History")
        st.write(f"Messages in conversation: {(st.session_state.messages)}")
 
if __name__ == "__main__":
    # Set page configuration
    st.set_page_config(
        page_title="AI Chat Assistant",
        page_icon="💬",
        layout="wide"
    )
    main()
