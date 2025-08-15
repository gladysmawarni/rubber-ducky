import os
import warnings
import streamlit as st
import time
# from PIL import Image

# langchain libraries
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# langsmith
os.environ["LANGSMITH_API_KEY"] = st.secrets['LANGSMITH_API_KEY']
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = 'rubber-ducky'

### -------- SESSION STATE ---------
if 'memories' not in st.session_state:
    st.session_state.memories = []
if 'state' not in st.session_state:
    st.session_state.state = None

# Suppress warnings related to date parsing
warnings.filterwarnings("ignore")

# openai API key
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Page config
st.set_page_config(page_title="Q&A", page_icon="🐥")

# webapp title
# icons, titles = st.columns([0.2, 0.8])
# with icons:
#     st.image('images\ducky.png')
# with titles:
st.title('Rubber Ducky, the RAG and Gen AI Course Assistant')

# Vector DB
embeddings = OpenAIEmbeddings()
vector_db = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)

### ---------- FUNCTIONS ------------
def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.04)

def get_context(preference):
    docs_faiss = vector_db.similarity_search_with_relevance_scores(preference, k=10)
    return docs_faiss


# function to ask llm model question and generate answer
# @traceable()
def generate_response(context, query) -> str:
    with st.spinner('Thinking...'):
        # Define the role of the assistant and prepare the prompt
        system = """You are a data science instructor. Your Meta Persona is a rubber ducky, so add a couple of quacks to your answer.  Answer the student's question professionally and concisely, using only the information provided within the given context. 
        Avoid introducing any external information. Refer to previous conversations when relevant to provide a clear and thorough response, addressing any lingering doubts. 
        Ensure that your response includes the following:
        - Answering the question, and identify and mention the relevant Section(s) and Lecture(s) from which the information is drawn.
        - Suggest additional related lectures by saying, 'If you want to know more, check the following lectures (located in Section X).'
        - Ask if the student has further questions or needs clarification
        """

        # prompt template, format the system message and user question
        TEMPLATE = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("system", "Chat history: {chat_history}"),
            ("system", "Context: {context}"),
            ("human", "Question: {query}"),
        ]
        )
        prompt = TEMPLATE.format(chat_history= st.session_state.memories, context= context, query=query)

        model = ChatOpenAI(model="gpt-5-nano")
        response_text = model.invoke(prompt).content
        st.session_state.memories.append({"role": "assistant", "content": response_text})

    with st.chat_message("ducky", avatar='https://raw.githubusercontent.com/gladysmawarni/rubber-ducky/refs/heads/main/images/small-ducky.png'):
        st.write_stream(stream_data(response_text))


### ----------- APP -------------
for memory in st.session_state.memories:
    if memory['role'] == 'assistant':
        with st.chat_message(memory["role"], avatar='https://raw.githubusercontent.com/gladysmawarni/rubber-ducky/refs/heads/main/images/small-ducky.png'):
            st.write(memory["content"])
    else:
        with st.chat_message(memory["role"]):
            st.write(memory["content"])

if st.session_state.state == None:
    with st.chat_message("assistant", avatar='https://raw.githubusercontent.com/gladysmawarni/rubber-ducky/refs/heads/main/images/small-ducky.png'):
        intro =  """
                 Hi there! I’m Rubber Ducky, your assistant for the RAG and Generative AI course. I can help with topics such as:\n
                ◦ RAG: Fundamentals, Unstructured Data, Multimodal, Agentic \n
                ◦ OpenAI API: Text and Images, Whisper, Embeddings, Fine Tuning. \n

                Do you have any questions on these topics or a specific section?
                  \n"""

        st.write(intro)
    # Add intro message to chat history
    st.session_state.memories.append({"role": "assistant", "content": intro})
    st.session_state.state = 'q-a'

# Accept user input
if user_input := st.chat_input("Say Something"):
    st.session_state.input = user_input
    # Add user message to chat history
    st.session_state.memories.append({"role": "user", "content": user_input})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.write(user_input)
    
    context = get_context(user_input)
    generate_response(context, user_input)
