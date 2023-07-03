import streamlit as st

from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

import os
import pandas as pd

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

st.title('GPT-3.5 LangChain Dataframe Agent')
st.text('natural language chat interface with dataframe grounding for question answering')

url = "https://raw.github.com/mattdelhey/kaggle-titanic/master/Data/train.csv"
if 'df' not in st.session_state:
    st.session_state['df'] = pd.read_csv(url)

st.divider()
st.subheader('Select Data Source')
st.caption('Default Dataset = Titanic')

upload = st.file_uploader("Upload Custom Dataset (.csv file)")
if upload:
    st.session_state['df'] = pd.read_csv(upload)

st.dataframe(st.session_state['df'])

st.divider()
st.subheader('Chat Interface')

with st.form('chat_area'):
    text = st.text_area('Enter Question(s) - regular messages should still work but you might as well use default chatgpt', 'what percentage of first class passengers were females?\nand how many 3rd class passengers aged over 40 embakred from S?')
    submitted = st.form_submit_button('Submit Question')

    if submitted:

        st.write('hi')

        with st.spinner('Generating Response'):
    
            agent = create_pandas_dataframe_agent(
                ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
                st.session_state['df'],
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
            )
            response = agent.run(text)

        st.info(response, icon="🔥")
