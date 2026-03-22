from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.document_loaders import WebBaseLoader
from openai import AssistantEventHandler
import streamlit as st
import openai, json

ASSISTANT_NAME = "Search Assistant"

class EventHandler(AssistantEventHandler):

    message = ""
    run_id = ""
    thread_id = ""

    def on_text_created(self, text) -> None:
        self.message_box = st.empty()

    def on_text_delta(self, delta, snapshot):
        self.message += delta.value
        self.message_box.markdown(self.message)

    def on_event(self, event):
        if event.event == "thread.run.created":
            self.run_id = event.data.id
            self.thread_id = event.data.thread_id

        if event.event == "thread.run.requires_action":
            submit_tool_outputs(self.run_id, self.thread_id)

st.set_page_config(
    page_title="AssistantGPT",
    page_icon="🤖",
)

st.title("AssistantGPT")

st.markdown(
    """
    Welcome!
    
    Use this streaming chatbot to ask an AI to do deep research for you!
    
    It will autonomously use DuckDuckGo, Wikipedia, navigate directly into websites to scrape content, and ultimately save the conclusion as a TXT file.
    """
)

# OpenAI API key의 유효성을 검사합니다.
def validate_key(api_key: str) -> bool:
    try:
        openai.OpenAI(api_key=api_key).models.list()
        return False
    except Exception as e:
        return True

with st.sidebar:
    st.markdown("### 🤖 Research Assistant")
    st.markdown("Powered by OpenAI Assistants API (Streaming)")
    st.write("---")
    API_KEY = st.text_input("Enter your OpenAI API key", type="password")
    is_invalid = validate_key(API_KEY)

# ================= Tools (4가지 파이썬 함수) =================
def wikipedia_search(inputs):
    query = inputs["query"]
    wp = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wp.run(query)

def duckduckgo_search(inputs):
    query = inputs["query"]
    ddg = DuckDuckGoSearchRun()
    urls = ddg.run(query)
    return urls

def scrape_website(inputs):
    url = inputs["url"]
    loader = WebBaseLoader([url])
    docs = loader.load()
    text = "\n\n".join([doc.page_content for doc in docs])
    return text

def save_to_txt(inputs):
    text = inputs["text"]
    with open("research_results.txt", "w") as file:
        file.write(text)
    return "Research results saved to research_results.txt"

functions_map = {
    "wikipedia_search": wikipedia_search,
    "duckduckgo_search": duckduckgo_search,
    "scrape_website": scrape_website,
    "save_to_txt": save_to_txt,
}

# ================= OpenAI Assistant에 넘겨줄 도구(Functions) 스키마 =================
functions = [
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Given the query, return the search result from Wikipedia.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The keyword of the information you want to search.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "duckduckgo_search",
            "description": "Given the query, return the search result from DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The keyword of the information you want to search.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_website",
            "description": "If you found the website link in DuckDuckGo, Use this to get the content of the link for your research.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the website you want to scrape"
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_to_txt",
            "description": "Use this tool to save the content as a .txt file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text you will save to a file."
                    }
                },
                "required": ["text"],
            },
        },
    }
]


#### Utilities
def get_run(run_id, thread_id):
    return openai.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )

def send_message(thread_id, content):
    return openai.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content,
    )

def get_messages(thread_id):
    messages = openai.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    return messages

def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        
        # Tool status update in Streamlit UI
        with st.chat_message("assistant"):
            st.write(f"🔧 도구 사용 중: `{function.name}`")
            
        try:
            result = functions_map[function.name](json.loads(function.arguments))
        except Exception as e:
            result = f"Error: {e}"
            
        outputs.append(
            {
                "output": str(result),
                "tool_call_id": action_id,
            }
        )
    return outputs

def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    with openai.beta.threads.runs.submit_tool_outputs_stream(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outputs,
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()

def insert_message(message, role):
    with st.chat_message(role):
        st.markdown(message)


def paint_history(thread_id):
    messages = get_messages(thread_id)
    for message in messages:
        insert_message(
            message.content[0].text.value,
            message.role,
        )

# ==============================================================
# 메인 로직
if "run" in st.session_state:
    pass

if not is_invalid:
    openai.api_key = API_KEY
    if "assistant" not in st.session_state:
        assistants = openai.beta.assistants.list(limit=10)
        for a in assistants:
            if a.name == ASSISTANT_NAME:
                assistant = openai.beta.assistants.retrieve(a.id)
                break
        else:
            assistant = openai.beta.assistants.create(
                name=ASSISTANT_NAME,
                instructions="""
                You are a research expert.
                Your task is to use Wikipedia or DuckDuckGo to gather comprehensive and accurate information about the query provided. 
                When you find a relevant website through DuckDuckGo, you must scrape the content from that website. Use this scraped content to thoroughly research and formulate a detailed answer to the question. 
                Combine information from Wikipedia, DuckDuckGo searches, and any relevant websites you find. Ensure that the final answer is well-organized and detailed, and include citations with links (URLs) for all sources used.
                Your research should be saved to a .txt file, and the content should match the detailed findings provided. Make sure to include all sources and relevant information.
                The information from Wikipedia must be included.
                """,
                model="gpt-4o-mini",
                tools=functions,
            )
        thread = openai.beta.threads.create()
        st.session_state["assistant"] = assistant
        st.session_state["thread"] = thread
    else:
        assistant = st.session_state["assistant"]
        thread = st.session_state["thread"]

    paint_history(thread.id)
    content = st.chat_input("What do you want to search?", disabled=is_invalid)
    if content:
        send_message(thread.id, content)
        insert_message(content, "user")

        with st.chat_message("assistant"):
            with openai.beta.threads.runs.stream(
                thread_id=thread.id,
                assistant_id=assistant.id,
                event_handler=EventHandler(),
            ) as stream:
                stream.until_done()
else:
    st.sidebar.warning("Input OpenAI API Key.")
