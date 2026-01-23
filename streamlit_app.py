from rag_module import build_index, generate_answer, load_pdf
import streamlit as st

st.title("RAG PDF Question Answering")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file is not None:
    chunks = load_pdf(uploaded_file)   # use your load_pdf function
    index = build_index(chunks)        # create index
    st.success("PDF processed successfully!")

    query = st.text_input("Enter your question:")
    if query:
        context = index               # your context for RAG
        answer = generate_answer(query, context)
        st.write("Answer:", answer)

