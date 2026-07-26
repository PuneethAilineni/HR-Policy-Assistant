import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from consts import Consts 
from ReRanker.reRanker import ReRanker      
from langchain_core.documents import Document

def format_docs(docs: list[Document]) -> str:
    """Join document chunks into one context string."""
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(doc.page_content.strip() for doc in docs)

def main():
    llm = Consts()._get_llm()
    st.set_page_config(
        page_title="Resilience X HR Policy Assistant",
        layout="centered"
    )
    st.title("Resilience X HR Policy Assistant")
    st.markdown("""
        Ask questions about our company HR policies.
        The assistant will only provide answers based on the official policy documents.
        Each question is treated independently.
    """)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an HR policy assistant for Resilience X.\n"
         "ONLY answer based on the provided context from documents.\n"
         "Do NOT use any knowledge outside the provided documents.\n"
         "If the answer is not in the documents, respond that you do not have enough information to answer.\n"
         "Keep answers concise and professional."),
        ("human",
         "Context from Documents:\n{context}\n\n"
         "User Query: {input}\n\n"
         "Final Answer:")
    ])

    strOutputParser = StrOutputParser()
    
    chat_chain = prompt | llm | strOutputParser

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask a question about our HR policies..."):
        with st.chat_message("user"):
            st.markdown(user_input)
        
        st.session_state.messages.append({"role": "user", "content": user_input})

        try:
            with st.spinner("Searching official HR documents..."):
                retriever = ReRanker()
                reranked_docs = retriever.invoke(query=user_input)
                context = format_docs(reranked_docs)

            with st.spinner("Generating response..."):
                response = chat_chain.invoke({
                    "context": context,
                    "input": user_input
                })

            with st.chat_message("assistant"):
                st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()