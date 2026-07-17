import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=1024
)

st.title('Enter the paper you want to know about')

st.header('Paper Summarizer',text_alignment='center', width='stretch')

paper_title = st.selectbox(label='Select paper title', options=['Attention is all you need', 'Prompt Engineering by Google x google', 'RAG for Knowledge-Intensive NLP Tasks'])

style = st.selectbox(label='Style', options=['Math Oriented', 'Casual', 'Coding Oriented'])

size = st.selectbox(label='Context Size', options=['Large', 'Mediam', 'Small'])


tamplate = PromptTemplate(
        template="""
        you are a smart Computer Science teacher. your task is to summarize this paper: {paper_title} in the tone of: {style} and in {size} length in context size,
        explain math in there is any in the Paper mentioned.
        give sample coding example if only asked
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

