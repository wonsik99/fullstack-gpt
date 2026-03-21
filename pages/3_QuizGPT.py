import streamlit as st
from pydantic import BaseModel, Field
from typing import List
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Pydantic models for Function Calling (Structured Output)
# These define the exact JSON schema that the LLM must follow when generating quiz data.
class Answer(BaseModel):
    answer: str = Field(description="The answer text")
    correct: bool = Field(description="Whether this answer is correct")

class Question(BaseModel):
    question: str = Field(description="The question text")
    answers: List[Answer] = Field(description="A list of 4 possible answers")

class Quiz(BaseModel):
    questions: List[Question] = Field(description="A list of 10 quiz questions")

st.set_page_config(page_title="QuizGPT", page_icon="❓")

st.title("QuizGPT")

# Initialize the LLM with low temperature for accurate quiz generation
llm = ChatOpenAI(
    model_name="gpt-5.4-nano",
    temperature=0.1
)

# Combines page_content from all documents into a single string
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

# Prompt that instructs the LLM to generate quiz questions from the given context
questions_prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
    You are a helpful assistant that is role playing as a teacher.

    Based ONLY on the following context make 10 questions to test the user's knowledge about the text.

    Each question should have 4 answers, three of them must be incorrect and one should be correct.

    Context: {context}
    """)
])

# Create a structured LLM that enforces the Quiz schema via OpenAI Function Calling
# This replaces the need for a separate formatting chain and JSON output parser
structured_llm = llm.with_structured_output(Quiz)

# Single chain: format docs → prompt → structured LLM → Quiz object
quiz_chain = {"context": format_docs} | questions_prompt | structured_llm


# Loads and splits an uploaded file into smaller document chunks for processing
@st.cache_resource(show_spinner="Splitting file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

# Runs the quiz chain and caches the result to avoid redundant LLM calls
# _docs is prefixed with underscore to tell Streamlit not to hash it
@st.cache_resource(show_spinner="Generating quiz...")
def run_quiz_chain(_docs, topic):
    return quiz_chain.invoke(_docs)

# Fetches relevant Wikipedia articles for a given search term
@st.cache_resource(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    return retriever.invoke(term)


# Sidebar: lets the user choose between file upload or Wikipedia search
with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose what you want to use",
        ("File", "Wikipedia Article")
    )

    if choice == "File":
        file = st.file_uploader(
            "Upload a .txt .pdf or .docx file",
            type=["pdf", "txt", "docx"]
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)

# Main content area
if not docs:
    st.markdown(
        """
        Welcome to QuizGPT!

        I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

        Get started by uploading a file or searching on Wikipedia in the sidebar
        """
    )
else:
    # Generate quiz from the loaded documents
    response = run_quiz_chain(docs, topic if topic else file.name)
    # Display quiz questions inside a form so all answers are submitted together
    with st.form("questions_form"):
        correct_count = 0
        for question in response.questions:
            st.write(question.question)
            value = st.radio(
                "Select an option",
                [answer.answer for answer in question.answers],
                index=None
            )
            # Check the selected answer against the correct one
            if value:
                correct_answer = next(a for a in question.answers if a.correct)
                if value == correct_answer.answer:
                    st.success("Correct!")
                    correct_count += 1
                else:
                    st.error("Wrong!")
        button = st.form_submit_button()
    # Display total score after submitting
    if button:
        total = len(response.questions)
        st.balloons() if correct_count == total else None
        st.info(f"🎯 Your score: **{correct_count} / {total}**")