import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=1024
)

st.title('Paper Summarizer',text_alignment='center')

st.header('Enter the paper you want to know about', width='stretch')

paper_title = st.selectbox(label='Select paper title', options=['Attention is all you need', 'Prompt Engineering by Google x google', 'RAG for Knowledge-Intensive NLP Tasks'])

style = st.selectbox(label='Style', options=['Math Oriented', 'Casual', 'Coding Oriented'])

size = st.selectbox(label='Context Size', options=['Large', 'Mediam', 'Small'])


tamplate = PromptTemplate(
        template="""
            Please summarize the research paper titled "{paper_title}" with the following specifications:
            Explanation Style: {style}
            Explanation Length: {size}
            1. Mathematical Details:
                - Include relevant mathematical equations if present in the paper.
                - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
            2. Analogies:
                - Use relatable analogies to simplify complex ideas.
            If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
            Ensure the summary is clear, accurate, and aligned with the provided style and length.
        """,
        input_variables=['paper_title', 'style', 'size'],
        validate_template=True
    )

prompt = tamplate.invoke({
        'paper_title': paper_title,
        'style': style,
        'size': size
    })

if st.button('Summerize', type='primary'):
    with st.spinner('Generating...'):
        responce = llm.invoke(prompt)
        st.write(responce.content)

