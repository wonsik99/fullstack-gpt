import streamlit as st
from langchain_community.document_loaders import SitemapLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
        filter_urls=[r"^(.*\/api).*$"],
        parsing_function=parse_page,
        requests_per_second=0.1
    )
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

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
        docs = load_website(url)
        st.write(docs)


    