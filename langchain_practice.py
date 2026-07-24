from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnableBranch
from langchain_groq import ChatGroq


load_dotenv()
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=1024
)


parser = StrOutputParser()

class Feedback(BaseModel):
    sentement : Literal['positive', 'negative'] = Field(description="Generate sentiment from given sentence.")

parser_pyd = PydanticOutputParser(pydantic_object=Feedback)


prompt = PromptTemplate(
    template="Generate sentiment from following sentence: {text}\n in this format: {format_instructions}",
    input_variables=['text'],
    partial_variables={'format_instructions': parser_pyd.get_format_instructions()}
    )

prompt_pos = PromptTemplate(
    template="Generate a short feedback for user from given positive review:\n {text}",
    input_variables=['text']
    )

prompt_neg = PromptTemplate(
    template="Generate a short feedback for user from given negative review:\n {text}",
    input_variables=['text']
    )

sentiment_chain = prompt | model | parser_pyd

# get_sentiment = sentiment_chain.invoke({
#     'text': 'this is a good phone'}
# )

get_feedback = RunnableBranch(
    (lambda x : x.sentement == 'positive', prompt_pos | model | parser),
    (lambda x : x.sentement == 'negative', prompt_neg | model | parser),
    RunnableLambda(lambda x : 'No valid review found')
    )

chain = sentiment_chain | get_feedback





# Implementing a PineCode vector store

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.documents import Document


doc1 = Document(
    page_content="Sakib-al-hasan in an all rounder in bangladesh cricket team",
    metadata = {'team': 'bangladesh'}
)

doc2 = Document(
    page_content="Mehidi hasan in a batter in bangladesh cricket team",
    metadata = {'team': 'bangladesh'}
)

doc3 = Document(
    page_content="Taskin in a bowler in bangladesh cricket team",
    metadata = {'team': 'bangladesh'}
)

docs = [doc1, doc2, doc3]

embedding = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-small-en-v1.5"
)

vectorstore = Chroma.from_documents(
    persist_directory="vector_store",
    collection_name='Temp',
    documents=docs, 
    embedding=embedding)



get_query  = vectorstore.similarity_search_with_score(
    query="best batter",
    k = 2
)

print(get_query)