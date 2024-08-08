import streamlit as st
from langchain.prompts.prompt import PromptTemplate
from PIL import Image
from get_elastic import get_metrics, get_process, get_hostnames
import os
import plotly.express as px
import io
import time
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
from datetime import datetime
from contextlib import redirect_stdout
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.callbacks import get_openai_callback
from langchain_openai import AzureChatOpenAI
import GenAIPlatformLLM
from GenAIPlatformLLM import GenAIPlatformLLM

from langchain.agents.agent_types import AgentType
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_core.output_parsers import StrOutputParser
from streamlit_feedback import streamlit_feedback
import logging
import warnings
import sqlite3
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)  
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub

@st.cache_data
def get_csv(file_path):
    df = pd.read_csv(file_path)
    print(df)
    return df

process_df = get_csv('testdata.csv')
df = process_df

def execute_python_code(code):
    """
    Executes Python code safely, capturing both plot outputs and textual outputs.

    Parameters:
    - code (str): The Python code to execute.

    Returns:
    - tuple: A tuple containing the textual output and an optional BytesIO object for the plot output.
    """
    # sanitize the code
    try:
    
        local_namespace = {}
        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            exec(code, {"plt": plt}, local_namespace)
        text_output = stdout_buf.getvalue()

        # Check for plot
        if plt.get_fignums():
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            plot_output = buf
        else:
            plot_output = None

        return text_output, plot_output

    except SyntaxError as e:
        error = f"Syntax error in the provided code: {e}"
        return (error, None)    
    except Exception as e:
        return (f"An error occurred during code execution: {e}", None)
    
def render_llm_response2(response):

    sections = response.split("```")

    for i, section in enumerate(sections, start=1):
        logging.debug(f"Section {i}: {section}\n")
      
        if section.startswith("python\n"):
            section = section.strip("```")
            section = section.strip("python")

            text_output, plot_output = execute_python_code(section)
            if plot_output:
                logging.debug("Render image")
                plot_area = st.empty()
                plot_area.pyplot(exec(section))
            if text_output:
                logging.debug("Render text")
                st.code(text_output)

        else:
            st.code(section)
            logging.debug("Render code")

user_input = st.text_area("Enter your text here:")
st.header("Output")
# Display the input text
st.write("You entered:")
st.write(user_input)

if st.button("Run Code"):
    try:
        # Ensure we only run the user-provided code and nothing else
        exec_globals = {"df": df, "px": px, "pd": pd, "st": st}
        exec_locals = {}
        exec(user_input, exec_globals, exec_locals)  # Execute the user-provided code
        # Check if a figure is created and display it
        if 'fig' in exec_locals:
            st.plotly_chart(exec_locals['fig'])
        else:
            st.write("No figure to display. Make sure to assign your Plotly figure to a variable named 'fig'.")
    except Exception as e:
        st.error(f"Error running code: {e}")

