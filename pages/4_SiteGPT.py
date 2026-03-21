import streamlit as st
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
# from langchain_community.document_loaders import SitemapLoader

st.set_page_config(page_title="SiteGPT", page_icon="🌐")

st.title("SiteGPT")

html2text_transformer = Html2TextTransformer()

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
    loader = AsyncChromiumLoader([url])
    docs = loader.load()
    transformed = html2text_transformer.transform_documents(docs)
    st.write(transformed)
    


    