import streamlit as st
from langchain_community.document_loaders import SitemapLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

llm = ChatOpenAI(
    model_name="gpt-5.4-nano",
    temperature=0.1
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question.
    If you can't just say you don't know, don't make anything up.

    Then, give a score to the answer between 0 and 5.
    The score should be high if the answer is related to the user's question, and low otherwise.
    If there is no relevant content, the score is 0.
    Always provide scores with your answers
    Context: {context}

    Examples:

    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!
    Question: {question}
    """
)

def get_answers(inputs):
    docs = inputs['docs']
    question = inputs['question']
    answers_chain = answers_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke({
    #         "context":doc.page_content,
    #         "question":question
    #     })
    #     answers.append(result.content)
    # st.write(answers)
    answers = [
        {
            "answer": answers_chain.invoke(
                {"context": doc.page_content, "question": question}
            ).content,
            "source": doc.metadata["source"],
            "date": doc.metadata["lastmod"],
        }
        for doc in docs
    ]
    return {
        "question": question,
        "answers": answers
    }

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.
            Use the answers that have the highest score (more helpful) and favor the most recent ones.
            Cite sources and return the sources of the answers as they are, do not change them.
            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = ""
    for answer in answers:
        condensed += f"Answer: {answer['answer']}\nSource: {answer['source']}\nDate: {answer['date']}\n\n"
    return choose_chain.invoke({
        "answers": condensed,
        "question": question
    })
    
def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return soup.get_text()

@st.cache_resource(show_spinner="Loading website...")
def load_website(url):

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200
    )

    loader = SitemapLoader(
        url,
        # filter_urls=[r"^(.*\/api).*$"],
        parsing_function=parse_page,
        requests_per_second=0.1
    )
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()

st.set_page_config(page_title="SiteGPT", page_icon="🌐")

st.title("SiteGPT")

st.markdown(
    """
    Welcome to SiteGPT!

    Ask questions about the content of a website.

    Upload your files to the sidebar
    """
)

with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com"
    )

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down Sitemap URL")
    else:
        retriever = load_website(url)
        query = st.text_input("Ask a question about the website")
        if query:
            chain = {
                "docs":retriever,
                "question": RunnablePassthrough()
            } | RunnableLambda(get_answers) | RunnableLambda(choose_answer)
 
            with st.chat_message("ai"):
                with st.spinner("Finding the best answer..."):
                    result = chain.invoke(query)
                    st.markdown(result.content)

    