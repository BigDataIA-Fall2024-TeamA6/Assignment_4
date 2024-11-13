import streamlit as st
import time
from research_agent import get_graph

def get_ai_response(question):
    """
    Placeholder function - replace this with your actual AI code integration
    """
    # This is where you'll integrate your AI code
    # Return the response from your AI model
    value = """
    ### Key Cybersecurity Threats
    1. **Malware**: This includes various forms of malicious software such as viruses, worms, trojans, and ransomware that can infiltrate systems to steal data, disrupt operations, or demand ransom for data recovery.

    2. **Phishing Attacks**: Cybercriminals use deceptive emails and messages to trick individuals into revealing sensitive information, such as passwords and credit card numbers, often leading to identity theft or financial loss.

    3. **Denial-of-Service (DoS) Attacks**: These attacks overwhelm a system or network with traffic, rendering it unavailable to legitimate users, which can disrupt business operations and lead to significant financial losses.

    4. **Social Engineering**: This involves manipulating individuals into divulging confidential information by exploiting psychological tactics, often leading to unauthorized access to systems or data.

    5. **Emerging Threats from AI and Quantum Computing**: As technology evolves, new threats such as AI-enabled attacks and vulnerabilities associated with quantum computing are emerging, posing significant risks to cybersecurity.
    """
    return value


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
        ai_agent = get_graph()
        with st.chat_message("user"):
            report = ai_agent.invoke({"input":str(prompt),
                             "chat_history":[]})
            st.markdown(prompt)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display assistant response with a loading spinner
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # response = get_ai_response(prompt)
                st.markdown(report)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": report})

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