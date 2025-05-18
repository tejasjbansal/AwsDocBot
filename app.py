
import streamlit as st
import lancedb
import os
from typing import Generator
from groq import Groq
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

st.set_page_config(page_icon="🔥", layout="wide",
                   page_title="AwsDocBot..")



# Initialize OpenAI client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


# Initialize LanceDB connection
@st.cache_resource
def init_db():
    """Initialize database connection.

    Returns:
        LanceDB table object
    """
    db = lancedb.connect("lancedb/ard")
    return db.open_table("awsdb")


def get_context(query: str, table, num_results: int = 5) -> str:
    """Search the database for relevant context.

    Args:
        query: User's question
        table: LanceDB table object
        num_results: Number of results to return

    Returns:
        str: Concatenated context from relevant chunks with source information
    """
    results = table.search(query).limit(num_results).to_pandas()
    contexts = []

    for _, row in results.iterrows():
        # Extract metadata
        filename = row["metadata"].get("filename", "")
        page_numbers = row["metadata"].get("page_numbers", [])
        title = row["metadata"].get("title", "")

        # Build source citation
        source_parts = []
        if isinstance(filename, str) and filename:
            source_parts.append(filename)
        
        # Handle page_numbers properly - check if it's an array-like object with length > 0
        if isinstance(page_numbers, (list, np.ndarray)) and len(page_numbers) > 0:
            source_parts.append(f"p. {', '.join(str(p) for p in page_numbers)}")
        elif page_numbers:  # Handle scalar case
            source_parts.append(f"p. {page_numbers}")

        source = f"\nSource: {' - '.join(source_parts)}" if source_parts else "\nSource: Unknown"
        if isinstance(title, str) and title:
            source += f"\nTitle: {title}"

        contexts.append(f"{row['text']}{source}")

    return "\n\n".join(contexts)


def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# Sidebar with app information
with st.sidebar:
    st.image("https://api.dicebear.com/7.x/bottts/svg?seed=ardbot", width=100)
    st.title("About AwsDocBot")
    
    st.markdown("""
    **AwsDocBot** is your intelligent document assistant powered by:
    - LanceDB for vector search
    - Llama 3.3 70B for natural language understanding
    - Streamlit for the user interface
    """)
    
    st.divider()
    
    st.subheader("📊 Document Information")
    st.info("""
    Currently loaded: **AWS Certified Cloud Practitioner**
    
    These documents cover all information related to AWS Certified Cloud Practitioner exam.
    """)
    
    st.divider()
    
    st.subheader("💡 Sample Questions")
    st.markdown("""
    Try asking:
    - What are services available in AWs?
    - What is AWS?
    - What is AWS Auto Scaling?
    - What Is Elastic Load Balancer (Elb)
    """)

# Main content area
st.markdown('<div class="main-header"><h1>📚 AwsdDocBot - Your Document Assistant</h1><p>Ask questions about the AWS Certified Cloud Practitioner documents and get instant answers with source references.</p></div>', unsafe_allow_html=True)


# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize database connection
table = init_db()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the document"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get relevant context
    with st.status("Searching document...", expanded=False) as status:
        try:
            context = get_context(prompt, table)
            st.markdown(
                """
                <style>
                .search-result {
                    margin: 10px 0;
                    padding: 10px;
                    border-radius: 6px;
                    background-color: #f2f2f2; /* Light gray for better contrast */
                    color: #000000; /* Black text for readability */
                }
                .search-result summary {
                    cursor: pointer;
                    color: #0056b3; /* Accessible blue */
                    font-weight: 600;
                }
                .search-result summary:hover {
                    color: #d9534f; /* Bootstrap red for hover */
                }
                .metadata {
                    font-size: 0.9em;
                    color: #333333; /* Dark gray for better readability */
                    font-style: italic;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            st.write("Found relevant sections:")
            for chunk in context.split("\n\n"):
                # Split into text and metadata parts
                parts = chunk.split("\n")
                text = parts[0]
                metadata = {}
                
                # Safely parse metadata
                for line in parts[1:]:
                    if ": " in line:
                        key, value = line.split(": ", 1)  # Split on first occurrence only
                        metadata[key] = value

                source = metadata.get("Source", "Unknown source")
                title = metadata.get("Title", "Untitled section")

                st.markdown(
                    f"""
                    <div class="search-result">
                        <details>
                            <summary>{source}</summary>
                            <div class="metadata">Section: {title}</div>
                            <div style="margin-top: 8px;">{text}</div>
                        </details>
                    </div>
                """,
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.error(f"Error retrieving context: {str(e)}", icon="🚨")
            context = "No context could be retrieved due to an error."

    # Prepare messages for API
    system_prompt = f"""You are a helpful assistant that answers questions based on the provided context.
    Use only the information from the context to answer questions. If you're unsure or the context
    doesn't contain the relevant information, say so.
    
    Context:
    {context}
    """

    # Correcting the messages structure for the API call
    messages_for_api = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add the conversation history (excluding the system message)
    for msg in st.session_state.messages:
        messages_for_api.append({"role": msg["role"], "content": msg["content"]})

    try:
        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages_for_api,
            stream=True
        )

        # Use the generator function with st.write_stream
        with st.chat_message("assistant", avatar="🤖"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
            
    except Exception as e:
        st.error(f"Error generating response: {str(e)}", icon="🚨")