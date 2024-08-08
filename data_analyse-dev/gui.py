import streamlit as st
import tempfile
import random
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

load_dotenv()
python_repl_ast = PythonAstREPLTool()

########################### Logging #############################

def setup_logger():
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  

    # Create handlers for file and console
    file_handler = logging.FileHandler('data_analyse.log')
    console_handler = logging.StreamHandler()

    # Create formatters and add it to handlers
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_formatter = logging.Formatter(log_format)
    console_formatter = logging.Formatter(log_format)

    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logging = setup_logger()

logging.info("Application started")

########################### ENVIRONMENT #############################
# Load environment variables from .env file
#url = os.getenv('GAI_GTW_URL')
#key = os.getenv('GAI_GTW_API_KEY')
#password = os.getenv('pass')

########################### FEEDBACK #############################


def initialize_db(db_path='feedback.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Create table
    c.execute('''CREATE TABLE IF NOT EXISTS user_feedback
                 (id INTEGER PRIMARY KEY,
                  user_question TEXT,
                  user_feedback INTEGER,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    # Save (commit) the changes and close the connection
    conn.commit()
    conn.close()

def store_feedback(user_question, user_feedback, db_path='feedback.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Insert a row of data
    c.execute("INSERT INTO user_feedback (user_question, user_feedback) VALUES (?, ?)",
              (user_question, user_feedback))
    # Save (commit) the changes
    conn.commit()
    # Close the connection
    conn.close()

def get_all_feedback(db_path='feedback.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT * FROM user_feedback")
    all_feedback = c.fetchall()
    conn.close()
    return all_feedback

def aggregate_feedback(db_path='feedback.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # SQL query to aggregate positive feedback by user_question and order by count of positive feedback in desc order
    query = """
    SELECT user_question, SUM(user_feedback) AS positive_feedback_count
    FROM user_feedback
    GROUP BY user_question
    ORDER BY positive_feedback_count DESC
    """
    
    c.execute(query)
    results = c.fetchall()
    conn.close()
    
    return results

initialize_db()

#store_feedback('How to reset my password?', 1)
questions_list=["Choose an option","Custom Question ..."]
aggregated_feedback = aggregate_feedback()
for question, positive_count in aggregated_feedback:
    questions_list.append(question)


#############################


process_metrics = """The meaning of each column in the dataframe is as follows:

- `timestamp`: The timestamp of the record.
- `command`: The command being executed.
- `pid`: The process ID.
- `tid`: The thread ID.
- `cpu`: The CPU usage.
- `mem`: The memory usage.
- `cpu_time`: The CPU time.
- `elapsed`: The elapsed time.
- `start_time`: The start time of the process.
- `state`: The state of the process.
- `user`: The user associated with the process.
- `ruser`: The real user associated with the process.
- `os`: The operating system.
- `global_cpu`: The global CPU usage.
- `global_ram`: The global RAM usage.
- `global_swap`: The global swap usage.
"""

metrics = """The meaning of each column in the dataframe is as follows:

- `TIMESTAMP`: Represents the timestamp of the data entry.
- `SWAP`: Represents the swap value.
- `RAM`: Represents the RAM value.
- `CPU`: Represents the CPU value.
"""

result = {
    "Tokens Used": 0,
    "Prompt Tokens": 0,
    "Completion Tokens": 0,
    "Total Cost (USD)": 0,
    "Total chat cost (USD)": 0
}





######## COMMMON FUNCTIONS ############



def get_GenAI_llm(model_id, model_kwargs = None):
    ######## Config LLM  GenAI Platform ############

    if model_kwargs is None:
        model_kwargs = {"temperature": 0.3}

    if model_id == "openai-gpt-35-turbo-16k":
        model_kwargs = {"temperature": 0.3, "timeout": 300}

    url = os.getenv('BR_URL')
    api_key = os.getenv('BR_KEY')


    model_id = model_id
    model_kwargs = model_kwargs


    # model definition
    llm = GenAIPlatformLLM(
        api_key=api_key,
        url=url,
        provider = GenAIPlatformLLM.get_provider_by_model_id(model_id),
        model_id=model_id,
        model_kwargs=model_kwargs,
    )
    return llm

def show_cost(input):
    st.session_state['total_cost'] = st.session_state['total_cost'] + result['Total Cost (USD)']
    st.sidebar.subheader("Cost Summary")
    st.sidebar.caption(f"Total usage Cost (USD): {st.session_state['total_cost']}")
    st.sidebar.caption(f"Request Cost (USD): {result['Total Cost (USD)']}")
    st.sidebar.caption(f"Prompt Tokens: {result['Prompt Tokens']}")
    st.sidebar.caption(f"Completion Tokens: {result['Completion Tokens']}")
    st.sidebar.caption(f"Tokens Used: {result['Tokens Used']}")
    return

def classify_intent_with_llm(llm, user_query):
    try:
        prompt = f"""Classify the following user query. Does the user want a plot, a graph, a chart, or are they asking for data information? 
        Answer only with one word 'plot' or 'data'.

        Examples:
        user query: 'Can you show me a graph of the data?'
        answer : 'plot'

        user query: 'What is the average temperature for the last month?'
        answer : 'data'

        user query: 'Plot cpu usage ?'
        answer : 'plot'

        now let's classify the following user query : '{user_query}'.
        ANSWER:
        """
        response = llm.invoke(prompt)
        
        # Ensure response is a string
        if isinstance(response, dict):
            response = response.get('output', '')  # Adjust this if the key is different

        logging.info(f"Classified intent: {response.lower()}")
        return response.lower().strip()
        
    except Exception as e:
        logging.error(f"Error classifying intent: {e}")
        return f"Error classifying intent: {e}"


    
@st.cache_resource
def wiki(prompt):
    wiki_research = WikipediaAPIWrapper().run(prompt)
    return wiki_research

@st.cache_data
def prompt_templates():
    data_problem_template = PromptTemplate(
        input_variables=['business_problem'],
        template='Convert the following business problem into a data science problem: {business_problem}.'
    )
    model_selection_template = PromptTemplate(
        input_variables=['data_problem', 'wikipedia_research'],
        template='Give a list (only the name) of machine learning algorithms that are suitable to solve this problem: {data_problem}, while using this wikipedia research: {wikipedia_research}.'
    )
    return data_problem_template, model_selection_template

@st.cache_resource
def chains():
    data_problem_chain = LLMChain(llm=llm, prompt=prompt_templates()[0], verbose=True, output_key='data_problem')
    model_selection_chain = LLMChain(llm=llm, prompt=prompt_templates()[1], verbose=True, output_key='model_selection')
    sequential_chain = SequentialChain(chains=[data_problem_chain, model_selection_chain], input_variables=['business_problem', 'wikipedia_research'], output_variables=['data_problem', 'model_selection'], verbose=True)
    return sequential_chain

@st.cache_data
def chains_output(prompt, wiki_research):
    my_chain = chains()
    my_chain_output = my_chain({'business_problem': prompt, 'wikipedia_research': wiki_research})
    my_data_problem = my_chain_output["data_problem"]
    my_model_selection = my_chain_output["model_selection"]
    return my_data_problem, my_model_selection

@st.cache_data
def list_to_selectbox(my_model_selection_input):
    algorithm_lines = my_model_selection_input.split('\n')
    algorithms = [algorithm.split(':')[-1].split('.')[-1].strip() for algorithm in algorithm_lines if algorithm.strip()]
    algorithms.insert(0, "Select Algorithm")
    formatted_list_output = [f"{algorithm}" for algorithm in algorithms if algorithm]
    return formatted_list_output

def clicked(button):
    st.session_state.clicked[button]= True

def process_transformation(process):
    """
    Transforms raw process data into a cleaned and formatted DataFrame.

    Parameters:
    - process (list): The raw process data.

    Returns:
    - DataFrame: The transformed process data as a DataFrame with standardized column names and types.
  
    """

    records = []
    for item in process:
        fields = item["fields"]
        record = {key: value[0] if isinstance(value, list) and len(value) == 1 else value 
                    for key, value in fields.items()}
        records.append(record)

    df_process = pd.DataFrame(records)

    ### Filtered dat_process to keep only some fields
    df_filtered_process = df_process[['TIMESTAMP', 'command', 'pid', 'tid', 'cpu_percent', 'mem_percent', 'time' , 'ELAPSED','STIME', 'state' , 'user', 'ruser','os']].copy()


    ### Data Convertion for process

    ## Convert Timestamp to a datetime field and remove timezone
    df_filtered_process['TIMESTAMP'] = pd.to_datetime(df_filtered_process['TIMESTAMP'])
    df_filtered_process['TIMESTAMP'] = df_filtered_process['TIMESTAMP'].dt.tz_localize(None)
    ## Round to the minute
    df_filtered_process['TIMESTAMP'] = df_filtered_process['TIMESTAMP'].dt.round('min')
    ## Convert to String
    df_filtered_process[['command', 'state', 'user','ruser','os']] = df_filtered_process[['command', 'state', 'user','ruser','os']].astype('string')
    ## Convert to Int64
    df_filtered_process[['pid', 'tid']] = df_filtered_process[['pid', 'tid']].astype('int64')
    ## STIME epoch to datetime
    df_filtered_process['STIME'] = pd.to_datetime(df_filtered_process['STIME'], unit='s')
    ## ELAPSED duration in sec to timedelta
    df_filtered_process['ELAPSED'] = pd.to_timedelta(df_filtered_process['ELAPSED'], unit='s')
    df_filtered_process['time'] = pd.to_timedelta(df_filtered_process['time'])
    ## Round to 2 git column cpu and mem
    df_filtered_process[['cpu_percent', 'mem_percent']] = df_filtered_process[['cpu_percent', 'mem_percent']].round(2)
    df_filtered_process[['cpu_percent', 'mem_percent']] = df_filtered_process[['cpu_percent', 'mem_percent']] * 100
    df_filtered_process['command'] = df_filtered_process['command'].str.slice(0, 50)



    df_filtered_process = df_filtered_process.rename(columns={
        'TIMESTAMP': 'timestamp',
        'STIME': 'start_time',
        'ELAPSED' : 'elapsed' ,
        'time' : 'cpu_time',
        'cpu_percent' : 'cpu',
        'mem_percent' : 'mem'
    })
    
    return df_filtered_process

def metrics_transformation(metrics):
    """
    Transforms raw metrics data into a cleaned and formatted DataFrame.

    Parameters:
    - metrics (list): The raw metrics data.

    Returns:
    - DataFrame: The transformed metrics data as a DataFrame with selected columns and rounded values.
    """
    
    df_metrics = pd.DataFrame(metrics)

    # Extract and expand the specific fields
    df_metrics['CPU'] = df_metrics['fields'].apply(lambda x: x['CPU'][0] if 'CPU' in x else None)
    df_metrics['SWAP'] = df_metrics['fields'].apply(lambda x: x['SWAP'][0] if 'SWAP' in x else None)
    df_metrics['RAM'] = df_metrics['fields'].apply(lambda x: x['RAM'][0] if 'RAM' in x else None)
    df_metrics['TIMESTAMP'] = df_metrics['fields'].apply(lambda x: x['TIMESTAMP'][0] if 'TIMESTAMP' in x else None)

    df_metrics_filtered = df_metrics[['TIMESTAMP', 'SWAP', 'RAM', 'CPU']].copy()

    ## Round to the minute
    df_metrics_filtered['TIMESTAMP'] = pd.to_datetime(df_metrics_filtered['TIMESTAMP'])
    df_metrics_filtered['TIMESTAMP'] = df_metrics_filtered['TIMESTAMP'].dt.tz_localize(None)
    df_metrics_filtered['TIMESTAMP'] = df_metrics_filtered['TIMESTAMP'].dt.round('min')
    ## round every float64 to 2 decimal
    df_metrics_filtered = df_metrics_filtered.round(2)
    
    return df_metrics_filtered

def merge_metrics_process(df_filtered_process, df_metrics_filtered):
    """
    Merges process data with metrics data based on the closest timestamp.

    Parameters:
    - df_filtered_process (DataFrame): The DataFrame containing filtered process data.
    - df_metrics_filtered (DataFrame): The DataFrame containing filtered metrics data.

    Returns:
    - DataFrame: The merged DataFrame including process and metrics data.
    """
    
    df_global = df_filtered_process[['timestamp', 'command', 'pid', 'tid', 'cpu', 'mem', 'cpu_time' , 'elapsed','start_time', 'state' , 'user', 'ruser','os']].copy()


    # Initialize columns for CPU, RAM, and SWAP in df_global
    df_global['global_cpu'] = pd.NA
    df_global['gloal_ram'] = pd.NA
    df_global['global_swap'] = pd.NA

    # Find the nearest TIMESTAMP for each row in df_process and merge the values
    for i, row in df_filtered_process.iterrows():
        # Calculate the absolute time difference between the current process timestamp and all system metrics timestamps
        time_diff = abs(df_metrics_filtered['TIMESTAMP'] - row['timestamp'])

        
        # Find the index of the minimum time difference
        closest_index = time_diff.idxmin()
        
        # Merge the values from the closest row in df_metrics to df_process
        df_global.at[i, 'global_cpu'] = df_metrics_filtered.loc[closest_index, 'CPU']
        df_global.at[i, 'gloal_ram'] = df_metrics_filtered.loc[closest_index, 'RAM']
        df_global.at[i, 'global_swap'] = df_metrics_filtered.loc[closest_index, 'SWAP']


    df_global[['global_cpu', 'gloal_ram', 'global_swap']] = df_global[['global_cpu', 'gloal_ram', 'global_swap']].astype('float64')
    ## Round to 2 git column cpu and mem
    df_global[['global_cpu', 'gloal_ram', 'global_swap']] = df_global[['global_cpu', 'gloal_ram', 'global_swap']].round(2)

    df_global = df_global.rename(columns={"cpu": "process_cpu"})
    df_global = df_global.rename(columns={"mem": "process_mem"})
    df_global = df_global.rename(columns={"timestamp": "TIMESTAMP"})

    return df_global

def execute_python_code(code):
    """
    Executes Python code safely, capturing both plot outputs and textual outputs.

    Parameters:
    - code (str): The Python code to execute.

    Returns:
    - tuple: A tuple containing the textual output and an optional BytesIO object for the plot output.
    """
    # sanitize the code
    if not is_code_safe(code):
        return "Unsafe code detected. Execution aborted.", None

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

def clean_llm_response(response):
    start_idx = response.find("```python")
    end_idx = response.rfind("```")
    if start_idx != -1 and end_idx != -1:
        response = response[start_idx + len("```python"):end_idx].strip()
    
    # Remove comments
    lines = response.split("\n")
    code_lines = [line for line in lines if not line.strip().startswith("#")]
    
    necessary_imports = [
        "import pandas as pd",
        "import numpy as np",  
    ]
    
    for import_statement in necessary_imports:
        if import_statement not in code_lines:
            code_lines.insert(0, import_statement)
    
    cleaned_code = "\n".join(code_lines)
    return cleaned_code



def clean_llm_response_plot(response):
    start_idx = response.find("```python")
    end_idx = response.rfind("```")
    if start_idx != -1 and end_idx != -1:
        response = response[start_idx + len("```python"):end_idx].strip()
        response = response.replace("\n", "\n")
    return response

def query_llm_for_validation(query, llm):
    validation_prompt = f"""
    Given the following user query: '{query}', check if the query is too complex. 
    Ensure all column names in the query exist in the dataframe and that the query can be simplified. 
    If the query is too complex, return: "Query is too complex, prompt the user to try again". 
    If the query can be simplified, return a simplified version of the query. 
    Remember their will be plotting questions and data questions, make sure that neither are too complex!
    Your response should be the simplified query or the message if the query is too complex.
    """
    validation_response = llm(validation_prompt)
    return validation_response

def run_agent_data(query, llm, df):
    history = st.session_state.get('conversation_history', [])
    intent = classify_intent_with_llm(llm, query)
    st.write(f"Intent : {intent}")
    logging.info(f"Question: {query} Intent : {intent}")
    query = query + " Use the tool python_repl_ast!"
    # validation_response = query_llm_for_validation(query, llm)

    # if "Query is too complex" in validation_response:
    #     return {"output": validation_response}

    pandas_agent = create_pandas_dataframe_agent(
        llm,
        df,
        tools=[PythonAstREPLTool()],
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_executor_kwargs={"handle_parsing_errors": True},
        max_iterations=10
    )

    try:
        prompt = f"""
        Given the following user query '{query}' and historical conversation '{history}', decide how to best provide the required data analysis. 
        Your output should be a direct answer or python code, if ANY PLOTTING OR GRAPHING INTENT IS IDENTIFIED TELL THE USER THIS: "Plotting intent has been identified, please use the interactive plotting section for your query!".
        Another thing to remember if your output is code please return your final answer as code with NO COMMENTS
        
        Use the given dataframe 'df' as the input dataframe. USE the python_repl_ast tool if needed as that's your only tool available, and if you have code which will run have the code be your final output no comments! Please for any questions related to the column global_ram, it is misspelled in the dataframe as gloal_ram remember that!
        
        This is what your Action Step Should look like every time. Another important thing your Action step should always be: (Action: python_repl_ast) never include any backslashes or forward slashes at all:
        Action: python_repl_ast

        This is an EXAMPLE of what your Action Input step should be structured like, do not just use this every time, it's just what it should be structured like:
        Action Input:
        # Data analysis code
        df_sorted['cpu_time_diff'] = df_sorted['cpu_time'].diff()
        df_sorted_cleaned = df_sorted.dropna()
        trend_summary = df_sorted_cleaned['cpu_time_diff'].describe()
        trend_summary
        
        MAKE SURE TO SPELL python_repl_ast CORRECTLY THEIR SHOULD BE NO BACK SLASHES OR FORWARD SLASHES IN IT!
        DONT FORGET, if you have code in your final answer ONLY DISPLAY THE CODE NO COMMENTS! This is an example of what your output should look like:
        ```python
        "example code"
        ```

        It should look like that so a helper function can clean your final answer and run the code!
        Also Remember your final answer should always be CODE NEVER LEAVE COMMENTS IN THE CODE!
        """
        output = pandas_agent.invoke(prompt)
        logging.info(output)
        
        if isinstance(output, dict) and "output" in output:
            response = output["output"]
            response = response.replace('\\', '').replace('/', '')
            return {"output": response}
        else:
            logging.error(f"Unexpected output format: {output}")
            st.error("An error occurred: Unexpected output format")
            return {"output": "Unexpected output format"}

    except Exception as e:
        logging.error(f"Exception during pandas agent run: {e}")
        st.error(f"Exception during pandas agent run: {e}")
        return {"output": f"Error: {e}"}



def run_agent_plot(query, llm, df):
    history = st.session_state.get('conversation_history', [])
    intent = classify_intent_with_llm(llm, query)
    st.write(f"Intent : {intent}")
    logging.info(f"Question: {query} Intent : {intent}")
    query = query + " Use the tool python_repl_ast!"

    # validation_response = query_llm_for_validation(query, llm)

    # if "Query is too complex" in validation_response:
    #     return {"output": validation_response}

    pandas_agent = create_pandas_dataframe_agent(
        llm,
        df,
        tools=[PythonAstREPLTool()],
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_executor_kwargs={"handle_parsing_errors": True},
        max_iterations=10
    )

    try:
        prompt = f"""
        Given the following user query '{query}' and historical conversation '{history}', identify if the intent is to create a plot or to analyze data.
        If it's to create a plot, identify the plot type and the columns to use and generate the appropriate Python code using plotly express to create the plot. 
        Your output should be python code only, no comments, and make sure to include necessary imports.
        Example for plotting:
        
        ```python
        import pandas as pd
        import plotly.express as px

        fig = px.scatter(df, x='x_column', y='y_column')
        ```

        Your final answer and output should be just three lines of code that look like the above, nothing more. The plot_type should be determined based on the user's query, such as scatter, line, bar, etc. Please be creative with the px command and the fields you include based on the user's query. Below are some of the different fields you can add between the parentheses of the px command:
            title='Plot Title',          # Title of the plot
            labels={{'x_column':'X Axis', 'y_column':'Y Axis'}}, # Custom labels for axes
            color='color_column',        # Column to determine the color of points
            size='size_column',          # Column to determine the size of points
            hover_name='hover_column',   # Column to determine the hover name of points
            facet_row='facet_row',       # Column to create facets in rows
            facet_col='facet_col',       # Column to create facets in columns
            log_x=True,                  # Logarithmic scale for x-axis
            log_y=True,                  # Logarithmic scale for y-axis
            animation_frame='frame',     # Column to create animation frames
            animation_group='group',     # Column to group animation frames

        If it's to analyze data, provide the required analysis.

        Use the given dataframe 'df' as the input dataframe. USE the python_repl_ast tool if needed as that's your only tool available, and if you have code which will run have the code be your final output no comments!
        This is what your Action Step Should look like every time. Another important thing your Action step should always be: (Action: python_repl_ast) never include any backslashes at all, if you are getting this error (python\\_repl\\_ast is not a valid tool, try one of [python_repl_ast]), it is because you are putting backslashes in the tool name python_repl_ast:
        Action: python_repl_ast
        This is an EXAMPLE of what your Action Input step should be structured like, do not just use this every time its just what it should be structured like:
        Action Input:
        # Data analysis code
        df_sorted['cpu_time_diff'] = df_sorted['cpu_time'].diff()
        df_sorted_cleaned = df_sorted.dropna()
        trend_summary = df_sorted_cleaned['cpu_time_diff'].describe()
        trend_summary

        MAKE SURE TO SPELL python_repl_ast CORRECTLY THEIR SHOULD BE NO BACK SLASHES OR FORWARD SLASHES IN IT!
        DONT FORGET, if you have code in your final answer ONLY DISPLAY THE CODE NO COMMENTS!
        Also if you see any user querieis for global_ram the column name is misspelled it is gloal_ram!
        """
        output = pandas_agent.invoke(prompt)
        logging.info(output)
        
        if isinstance(output, dict) and "output" in output:
            response = output["output"]
            # Ensure no backslashes or slashes in the response
            response = response.replace('\\', '').replace('/', '')
            return {"output": response}
        else:
            logging.error(f"Unexpected output format: {output}")
            st.error("An error occurred: Unexpected output format")
            return {"output": "Unexpected output format"}

    except Exception as e:
        logging.error(f"Exception during pandas agent run: {e}")
        st.error(f"Exception during pandas agent run: {e}")
        return {"output": f"Error: {e}"}

def format_response( res):
    # Remove the load_csv from the answer if it exists
    csv_line = res.find("read_csv")
    if csv_line > 0:
        return_before_csv_line = res[0:csv_line].rfind("\n")
        if return_before_csv_line == -1:
            # The read_csv line is the first line so there is nothing to need before it
            res_before = ""
        else:
            res_before = res[0:return_before_csv_line]
        res_after = res[csv_line:]
        return_after_csv_line = res_after.find("\n")
        if return_after_csv_line == -1:
            # The read_csv is the last line
            res_after = ""
        else:
            res_after = res_after[return_after_csv_line:]
        res = res_before + res_after
    return res

@st.cache_data
def python_solution(my_data_problem, selected_algorithm, user_csv):
    solution = pandas_agent.invoke(
        f"Write a Python script to solve this: {my_data_problem}, using this algorithm: {selected_algorithm}, using this as your dataset: {temp_file_name}."
    )
    return solution

def is_code_safe(code):
    # List of unsafe keywords or patterns you might want to check for
    unsafe_patterns = ['import os', 'open(', 'subprocess', 'sys.exit', 'eval(', 'exec(']
    for pattern in unsafe_patterns:
        if pattern in code:
            return False
    return True

@st.cache_data
def run_get_process(user_question_host, start_datetime_str, end_datetime_str):
    return get_process(user_question_host, start_datetime_str, end_datetime_str)

@st.cache_data
def get_csv(file_path):
    df = pd.read_csv(file_path)
    print(df)
    return df

@st.cache_data
def run_get_metrics(user_question_host, start_datetime_str, end_datetime_str):
    metric = get_metrics(user_question_host, start_datetime_str, end_datetime_str)

    if metrics.startswith("Error pulling metrics"):
        st.error("An error occurred while pulling the metrics")
        st.stop()
    else:
        return metric
    
def update_conversation_history(user_query, llm_response):
    # Add the new query and response to the conversation history
    st.session_state.conversation_history.append(f"User: {user_query}")
    st.session_state.conversation_history.append(f"LLM: {llm_response}")

    # Keep only the last two user questions and their responses
    if len(st.session_state.conversation_history) > 4:
        st.session_state.conversation_history = st.session_state.conversation_history[-2:]

@st.cache_data
def getting_hostname():
    # hostnames = get_hostnames()
    # hostnames.insert(0, "Choose a hostname")
    hostnames = {"MyServer"}

    return hostnames

######## MAIN ############



######## Key for st.state ############

if 'clicked' not in st.session_state:
    st.session_state.clicked ={1:False}

if 'result_data' not in st.session_state:
    st.session_state.result_data = None

if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []

if "datasets" not in st.session_state:
    st.session_state["datasets"] = None

if "total_cost" not in st.session_state:
    st.session_state["total_cost"] = 0

####################


st.set_page_config(page_title="Data Analysis", page_icon="", layout="wide")
st.title("POC for Data analysis", anchor=False)

# Get list of available models from GenAIPlatformLLM
model_list = GenAIPlatformLLM.get_available_model_ids()
model_list = [model for model in model_list if model.startswith(("anthropic", "mistral", "openai", "meta"))]



######## SIDEBAR ############
with st.sidebar:
    with st.expander("General options"):
        st.caption('''**Choose a LLM to use**''')    
        
        model = st.selectbox("Select Model", options=model_list, key="model_selection")
        #temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1, key="temperature_slider") 
        temperature = 0.0
    
    st.divider()


    with st.expander("CSV options"):
 
        st.caption('''**Choose CSV file properties**''')
        delimiter = st.selectbox("delimiter", options=[",", ";"], key="delimiter_selection")
        encoding = st.selectbox("Encoding", options=["utf8", "latin1"], key="encoding_selection")
    
    st.divider()


######## MAIN PAGE ############

# Tabs for Summarize and Daily Reports
tab1, tab2 = st.tabs(["Data analyse", "Data analyse CSV"])

llm = get_GenAI_llm("mistral.mixtral-8x7b-instruct-v0:1", model_kwargs = {"temperature": 0, "timeout": 5000})


######## TAB1 ############
with tab1:

    if 'data_tab1' not in st.session_state:
        st.session_state['data_tab1'] = None

    st.header(f"AI Assistant for Data Science ({model})")

    with st.form("my_form"):
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start date", key="start_date")
            start_time = st.time_input("Start time", key="start_time")

        with col2:
            end_date = st.date_input("End date", key="end_date")
            end_time = st.time_input("End time", key="end_time")
            
        # Combine date and time into a single datetime object
        start_datetime = datetime.combine(start_date, start_time)
        end_datetime = datetime.combine(end_date, end_time)
        
        # Format datetime object to the specified format
        start_datetime_str = start_datetime.isoformat() + "Z"
        end_datetime_str = end_datetime.isoformat() + "Z"

        # For debugging purposes, display the formatted datetime strings
        logging.debug(f"Start DateTime: {start_datetime_str}")
        logging.debug(f"End DateTime: {end_datetime_str}")

        # create a select box for the hostnames
        user_question_host = st.selectbox("Which hostname are you interested in ?", getting_hostname(), key="hostname_list")

        # Checkbox for pulling process data
        pull_process_data = st.checkbox("Pull process data?", key="pull_process")

        #user_question_host = st.text_input("Which hostname are you interested in ?", key="hostname", max_chars=100 )

        # Submit button for the form
        submitted = st.form_submit_button("Submit")
            
    if submitted and start_date and end_date or st.session_state["datasets"] is not None:
    
        # Convert strings to datetime objects
        start_time = datetime.fromisoformat(start_datetime_str.rstrip("Z"))
        end_time = datetime.fromisoformat(end_datetime_str.rstrip("Z"))

        # Calculate the time delta
        delta = end_time - start_time
        total_hours = delta.total_seconds() / 3600  # Convert delta to total hours
        
        # Extract days, hours, and minutes from the delta
        days = delta.days
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Format the output to exclude seconds
        if days > 0:
            timeframe = f"{int(days)} days, {int(hours)} hours, {int(minutes)} minutes"
        else:
            timeframe =  f"{int(hours)} hours, {int(minutes)} minutes"

        
        st.write(f"You choose a period of {timeframe} for the hostname {user_question_host} ")

        if pull_process_data == True and total_hours > 12:
            st.error("timeframe is too long to pull process data, please select a shorter timeframe")
            st.stop()

        modified_values = f"You choose a period of {timeframe} for the hostname {user_question_host} with process data {pull_process_data}"

        modif = False
        if modified_values != st.session_state['data_tab1']:
            st.session_state['data_tab1'] = f"You choose a period of {timeframe} for the hostname {user_question_host} with process data {pull_process_data}"
            modif = True

        try:

            with st.spinner('Pulling hostname datas...'):
                if "datasets" not in st.session_state or modif == True:
                    
                    if pull_process_data == False:
                        df = metrics_transformation(run_get_metrics(user_question_host, start_datetime_str, end_datetime_str))
                    else:
                        # metrics_df = metrics_transformation(run_get_metrics(user_question_host, start_datetime_str, end_datetime_str))
                        process_df = get_csv('JeanLuc8am.csv')
                        df = process_df

                    st.session_state["datasets"] = df

                else:
                    if total_hours > 12 :
                        process = pull_process_data
                    else:
                        process = pull_process_data

                    df = st.session_state["datasets"]

            st.success('Datas pulled !')
        except Exception as e:
            logging.error(f"An error occurred while pulling the data: {e}")
            st.error(f"An error occurred while pulling the data: {e}")
            st.stop()

        
        if df is not None: 
        
            pandas_agent = create_pandas_dataframe_agent(
                llm,
                df,
                tools=[PythonAstREPLTool()],
                verbose=True,
                agent_type= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                #agent_type = AgentType.SELF_ASK_WITH_SEARCH,
                #agent_type = "openai-functions",
                #agent_type=AgentType.OPENAI_FUNCTIONS,
                #agent_type = "openai-functions",
                #agent_type="tool-calling",
                #agent_type = "openai-tools",
                agent_executor_kwargs={"handle_parsing_errors": True},
                max_iterations=15
            )
            print(pandas_agent)
           
            @st.cache_data
            def global_overview(input):
                
                st.write("**Data Overview**")
                st.write("The first rows of your dataset look like this:")
                st.dataframe(df)
                #st.write(df.head())
                nb_line = len(df)

                st.write(f"Dataset contains {nb_line} rows")
                
                missing_values = df.isnull().sum().sum()
                st.write(f"There are {missing_values} missing value(s)")

                if pull_process_data == True:
                    st.markdown(process_metrics)
                else:
                    st.markdown(metrics)

                #result = pandas_agent.invoke("Identify trends in the data that may be useful for analysis using python_repl_ast")
                #st.write(result["output"])
                #try:
                    #run_agent("How many rows in the dataframe?", llm, df)
                # except Exception as e:
                #     logging.error(f"An error occurred while running the agent: {e}")
                #     st.error(f"An error occurred while running the agent: {e}")    

                st.write("**Data Summarization**")
                #columns_to_describe = [col for col in df.columns if col != 'TIMESTAMP']
                columns_to_describe = [col for col in df.columns if col not in ['TIMESTAMP', 'start_time']]
                description = df[columns_to_describe].describe()
                st.write(description)

                return


            global_overview(modified_values)

    # Sample questions for Interactive Data Analysis
    data_analysis_sample_questions = [
        "What is the average CPU usage over time?",
        "Provide a summary of RAM usage statistics.",
        "Identify trends in SWAP usage.",
        "Show the correlation between CPU, RAM, and SWAP usage."
    ]

    # Friendly error messages
    friendly_errors = [
        "Oops! Something went wrong with the AI. Please try again.",
        "An unexpected error occurred with the AI. Please try again.",
        "The AI had an issue processing your request, try again.",
    ]

    def update_progress_bar(progress, message, bar):
        bar.progress(progress)
        st.write(message)
        time.sleep(1) 

    @st.cache_data
    def answer_data_question(input):
        agent_response = run_agent_data(input, llm, df)
        return agent_response["output"]


    st.header("Interactive Data Analysis")
    user_data_question = None

    selected_data_question = st.selectbox("Would you like to analyze any of the following data aspects?", ["Choose an option", "Custom Question ..."] + data_analysis_sample_questions)

    if selected_data_question == "Custom Question ...":
        user_data_question = st.text_input("Enter your data analysis question ", max_chars=300, key="user_data_question")
    elif selected_data_question not in ("Choose an option"):
        user_data_question = selected_data_question

    log_placeholder = st.empty()

    if user_data_question:
        if user_data_question not in ("", "no", "No"):
            with st.spinner('Analysing..'):
                progress_bar = st.progress(0)
                update_progress_bar(10, "Initializing LLM connection...", progress_bar)
                llm_response = answer_data_question(user_data_question)
                update_progress_bar(50, "Processing your request...", progress_bar)
                # st.write("Generated Python Code:")
                #st.code(llm_response, language="python")
                update_conversation_history(user_data_question, llm_response)
                update_progress_bar(70, "Running the analysis...", progress_bar)

                # Check for agent timeout
                if "Agent stopped due to iteration limit or time limit." in llm_response:
                    st.error(random.choice(friendly_errors))
                else:
                    try:
                        exec_globals = {"df": df, "pd": pd, "st": st}
                        exec_locals = {}
                        clean_response = clean_llm_response(llm_response)
                        
                        # Debug output for clean response
                        # st.write("Cleaned Python Code:")
                        #st.code(clean_response, language="python")

                        # Capture the output of the exec call
                        from io import StringIO
                        import sys
                        
                        old_stdout = sys.stdout
                        sys.stdout = mystdout = StringIO()

                        # Verify that clean_response is not empty and is valid Python code
                        if clean_response:
                            exec(clean_response, exec_globals, exec_locals)
                        else:
                            st.error("The cleaned response is empty or invalid. Please try again.")
                        
                        sys.stdout = old_stdout
                        output = mystdout.getvalue()
                        
                        update_progress_bar(90, "Displaying the results...", progress_bar)
                        if output:
                             st.write(output)
                        for key, value in exec_locals.items():
                            if isinstance(value, (pd.DataFrame, pd.Series)):
                                st.write(value)
                            else:
                                st.write({key: value})
                        update_progress_bar(100, "Done.", progress_bar)
                        st.write("Direct output from LLM response:")
                        st.write(llm_response)
                    except Exception as e:
                        st.write("Direct output from LLM response:")
                        st.write(llm_response)


                col1, col2 = st.columns(2)  # This creates two columns for the buttons to sit side-by-side
                with col1:
                    thumbs_up = st.button('👍', key='thumbs_up')
                with col2:
                    thumbs_down = st.button('👎', key='thumbs_down')

                if thumbs_up:
                    store_feedback(user_data_question, 1)
                    st.success("Thank you for your feedback!")

                if thumbs_down:
                    store_feedback(user_data_question, 0)
                    st.error("Thank you for your feedback! We'll work to improve.")
        elif user_data_question in ("no", "No"):
            st.write("Thanks")



    # Sample questions for Interactive Plotting
    plotting_sample_questions = [
        "Show me a line graph of CPU usage over time.",
        "Create a scatter plot of RAM vs. CPU usage.",
        "Display a bar chart comparing CPU, RAM, and SWAP usage.",
        "Generate a histogram of SWAP usage distribution."
    ]

    @st.cache_data
    def answer_plotting_question(input):
        agent_response = run_agent_plot(input, llm, df)
        return agent_response["output"]


    # Main application
    st.header("Interactive Plotting")
    user_plotting_question = None

    selected_plotting_question = st.selectbox("Would you like to create any of the following plots?", ["Choose an option", "Custom Question ..."] + plotting_sample_questions)

    if selected_plotting_question == "Custom Question ...":
        user_plotting_question = st.text_input("Enter your plotting question ", max_chars=300, key="user_plotting_question")
    elif selected_plotting_question not in ("Choose an option"):
        user_plotting_question = selected_plotting_question

    log_placeholder = st.empty()

    if user_plotting_question:
        if user_plotting_question not in ("", "no", "No"):
            with st.spinner('Analysing..'):
                progress_bar = st.progress(0)
                update_progress_bar(10, "Initializing LLM connection...", progress_bar)
                llm_response = answer_plotting_question(user_plotting_question)
                update_progress_bar(50, "Processing your request...", progress_bar)
                # st.write("Generated Python Code:")
                # st.code(llm_response, language="python")
                update_conversation_history(user_plotting_question, llm_response)
                if "Agent stopped due to iteration limit or time limit." in llm_response:
                    st.error(random.choice(friendly_errors))
                else:
                    try:
                        exec_globals = {"df": df, "px": px, "pd": pd, "st": st}
                        exec_locals = {}

                        clean_response = clean_llm_response_plot(llm_response)
                        for line in clean_response.split("\n"):
                            if "df['" in line:
                                col_name = line.split("df['")[1].split("']")[0]
                                if col_name not in df.columns:
                                    raise ValueError(f"Invalid column name in the generated code: {col_name}")

                        update_progress_bar(70, "Running the analysis...", progress_bar)
                        exec(clean_response, exec_globals, exec_locals)
                        
                        update_progress_bar(90, "Displaying the results...", progress_bar)
                        if 'fig' in exec_locals:
                            st.plotly_chart(exec_locals['fig'])
                        else:
                            st.write(exec_locals)
                        update_progress_bar(100, "Done.", progress_bar)
                    except Exception as e:
                        st.write(random.choice(friendly_errors))
        elif user_plotting_question in ("no", "No"):
            st.write("Thanks")    


######## TAB2 ############

llm = get_GenAI_llm("mistral.mixtral-8x7b-instruct-v0:1", model_kwargs = {"temperature": 0, "timeout": 5000})

friendly_errors = [
    "Oops! Something went wrong with the AI. Please try again.",
    "An unexpected error occurred with the AI. Please try again.",
    "The AI had an issue processing your request, try again.",
]

def update_progress_bar(progress, message, bar):
    bar.progress(progress)
    st.write(message)
    time.sleep(1) 

@st.cache_data
def csv_answer_data_question(input):
    agent_response = csvrun_agent_data(input, llm, df)
    return agent_response["output"]

@st.cache_data
def csv_answer_plotting_question(input):
    agent_response = csvrun_agent_plot(input, llm, df)
    return agent_response["output"]

@st.cache_data
def load_csv_data(file, delimiter, encoding='utf-8'):
    df = pd.read_csv(file, delimiter=delimiter, encoding=encoding)
    return df


@st.cache_data
def check_csv_organization(file_path):
    delimiters = [',', '\t', ';', '|', ' ']
    
    with open(file_path, 'r') as file:
        sample_lines = [next(file) for _ in range(5)]
    
    for delimiter in delimiters:
        if all(delimiter in line for line in sample_lines):
            print(f"Delimiter found: {delimiter}")
            return f'delimiter: {delimiter}'
    
    temp_df = pd.read_csv(file_path)
    if temp_df.shape[1] > 1:
        return 'organized'
    else:
        return 'unorganized'

@st.cache_data
def organize_csv(uploaded_file, delimiter):
    df = pd.read_csv(uploaded_file, delimiter=delimiter, encoding='utf-8')
    return df

def csvrun_agent_data(query, llm, df):
    history = st.session_state.get('csv_conversation_history', [])
    intent = classify_intent_with_llm(llm, query)
    st.write(f"Intent : {intent}")
    logging.info(f"Question: {query} Intent : {intent}")
    query = query + " Use the tool python_repl_ast!"



    pandas_agent = create_pandas_dataframe_agent(
        llm,
        df,
        tools=[PythonAstREPLTool()],
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_executor_kwargs={"handle_parsing_errors": True},
        max_iterations=10
    )

    try:
        prompt = f"""
        Given the following user query '{query}' and historical conversation '{history}', decide how to best provide the required data analysis. 
        Your output should be a direct answer or python code, if ANY PLOTTING OR GRAPHING INTENT IS IDENTIFIED TELL THE USER THIS: "Plotting intent has been identified, please use the interactive plotting section for your query!".
        Another thing to remember if your output is code please return your final answer as code with NO COMMENTS
        
        Use the given dataframe 'df' as the input dataframe. USE PythonAstREPL tool if needed as that's your only tool available, and if you have code which will run have the code be your final output no comments! Please for any questions related to the column global_ram, it is misspelled in the dataframe as gloal_ram remember that!
        
        This is what your Action Step Should look like every time. Another important thing your Action step should always be: (Action: python_repl_ast) never include any backslashes or forward slashes at all:
        Action: python_repl_ast
        This is an EXAMPLE of what your Action Input step should be structured like, do not just use this every time, it's just what it should be structured like:
        Action Input:
        # Data analysis code
        df_sorted['cpu_time_diff'] = df_sorted['cpu_time'].diff()
        df_sorted_cleaned = df_sorted.dropna()
        trend_summary = df_sorted_cleaned['cpu_time_diff'].describe()
        trend_summary

        DONT FORGET, if you have code in your final answer ONLY DISPLAY THE CODE NO COMMENTS! This is an example of what your output should look like:
        ```python
        "example code"
        ```

        It should look like that so a helper function can clean your final answer and run the code!
        Also Remember your final answer should always be CODE NEVER LEAVE COMMENTS IN THE CODE!
        """
        output = pandas_agent.invoke(prompt)
        logging.info(output)
        
        if isinstance(output, dict) and "output" in output:
            response = output["output"]
            response = response.replace('\\', '').replace('/', '')
            return {"output": response}
        else:
            logging.error(f"Unexpected output format: {output}")
            st.error("An error occurred: Unexpected output format")
            return {"output": "Unexpected output format"}

    except Exception as e:
        logging.error(f"Exception during pandas agent run: {e}")
        st.error(f"Exception during pandas agent run: {e}")
        return {"output": f"Error: {e}"}

def csvrun_agent_plot(query, llm, df):
    history = st.session_state.get('csv_conversation_history', [])
    intent = classify_intent_with_llm(llm, query)
    st.write(f"Intent : {intent}")
    logging.info(f"Question: {query} Intent : {intent}")
    query = query + " Use the tool python_repl_ast!"


    pandas_agent = create_pandas_dataframe_agent(
        llm,
        df,
        tools=[PythonAstREPLTool()],
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_executor_kwargs={"handle_parsing_errors": True},
        max_iterations=10
    )

    try:
        prompt = f"""
        Given the following user query '{query}' and historical conversation '{history}', identify if the intent is to create a plot or to analyze data.
        If it's to create a plot, identify the plot type and the columns to use and generate the appropriate Python code using plotly express to create the plot. 
        Your output should be python code only, no comments, and make sure to include necessary imports.
        Example for plotting:
        
        ```python
        import pandas as pd
        import plotly.express as px

        fig = px.scatter(df, x='x_column', y='y_column')
        ```

        Your final answer and output should be just three lines of code that look like the above, nothing more. The plot_type should be determined based on the user's query, such as scatter, line, bar, etc. Please be creative with the px command and the fields you include based on the user's query. Below are some of the different fields you can add between the parentheses of the px command:
            title='Plot Title',          # Title of the plot
            labels={{'x_column':'X Axis', 'y_column':'Y Axis'}}, # Custom labels for axes
            color='color_column',        # Column to determine the color of points
            size='size_column',          # Column to determine the size of points
            hover_name='hover_column',   # Column to determine the hover name of points
            facet_row='facet_row',       # Column to create facets in rows
            facet_col='facet_col',       # Column to create facets in columns
            log_x=True,                  # Logarithmic scale for x-axis
            log_y=True,                  # Logarithmic scale for y-axis
            animation_frame='frame',     # Column to create animation frames
            animation_group='group',     # Column to group animation frames

        If it's to analyze data, provide the required analysis.

        Use the given dataframe 'df' as the input dataframe. USE PythonAstREPL tool if needed as that's your only tool available, and if you have code which will run have the code be your final output no comments!
        This is what your Action Step Should look like every time. Another important thing your Action step should always be: (Action: python_repl_ast) never include any backslashes at all, if you are getting this error (python\\_repl\\_ast is not a valid tool, try one of [python_repl_ast]), it is because you are putting backslashes in the tool name python_repl_ast:
        Action: python_repl_ast
        This is an EXAMPLE of what your Action Input step should be structured like, do not just use this every time its just what it should be structured like:
        Action Input:
        # Data analysis code
        df_sorted['cpu_time_diff'] = df_sorted['cpu_time'].diff()
        df_sorted_cleaned = df_sorted.dropna()
        trend_summary = df_sorted_cleaned['cpu_time_diff'].describe()
        trend_summary

        DONT FORGET, if you have code in your final answer ONLY DISPLAY THE CODE NO COMMENTS!
        Also if you see any user querieis for global_ram the column name is misspelled it is gloal_ram!
        """
        output = pandas_agent.invoke(prompt)
        logging.info(output)
        
        if isinstance(output, dict) and "output" in output:
            response = output["output"]
            response = response.replace('\\', '').replace('/', '')
            return {"output": response}
        else:
            logging.error(f"Unexpected output format: {output}")
            st.error("An error occurred: Unexpected output format")
            return {"output": "Unexpected output format"}

    except Exception as e:
        logging.error(f"Exception during pandas agent run: {e}")
        st.error(f"Exception during pandas agent run: {e}")
        return {"output": f"Error: {e}"}

with tab2:
    st.header('AI Assistant for Data Science')

    st.divider()
    
    user_csv = st.file_uploader("Upload your file here", type="csv", accept_multiple_files=False)
    
    st.divider()
    
    if user_csv is not None:
        try:
            with st.spinner('Reading CSV file...'):
                user_csv.seek(0)
                
                # Create a temporary file to use with pandas
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmpfile:
                    tmpfile.write(user_csv.read())
                    temp_file_name = tmpfile.name
                
                # Check CSV organization
                response = check_csv_organization(temp_file_name)
                
                if "delimiter" in response:
                    delimiter = response.split("delimiter: ")[1].strip()
                    st.write(f"Delimiter suggested: {delimiter}")
                    df = load_csv_data(temp_file_name, delimiter)
                elif response == "organized":
                    st.write("CSV is already organized.")
                    df = pd.read_csv(temp_file_name)
                else:
                    st.error("CSV is too unorganized to process.")
                    st.stop()
                
                st.success('Dataframe ready!')
                st.write("Here's a preview of your data:")
                st.dataframe(df)  # Display the first few rows of the DataFrame

        except pd.errors.EmptyDataError:
            st.error("Uploaded CSV file is empty.")
            st.stop()
        except Exception as e:
            st.error(f"An error occurred while reading the CSV file: {e}")
            st.stop()

        st.header("CSV Data Interaction")
        user_data_question = st.text_input("Enter your data analysis question", max_chars=300, key="csv_user_data_question")

        if user_data_question not in ("", "no", "No"):
            with st.spinner('Analysing..'):
                progress_bar = st.progress(0)
                update_progress_bar(10, "Initializing LLM connection...", progress_bar)
                llm_response = csv_answer_data_question(user_data_question)
                update_progress_bar(50, "Processing your request...", progress_bar)
                update_progress_bar(70, "Running the analysis...", progress_bar)

                if "Agent stopped due to iteration limit or time limit." in llm_response:
                    st.error(random.choice(friendly_errors))
                else:
                    try:
                        exec_globals = {"df": df, "pd": pd, "st": st}
                        exec_locals = {}
                        clean_response = llm_response.strip("```").strip("python")

                        from io import StringIO
                        import sys

                        old_stdout = sys.stdout
                        sys.stdout = mystdout = StringIO()

                        if clean_response:
                            exec(clean_response, exec_globals, exec_locals)
                        else:
                            st.error("The cleaned response is empty or invalid. Please try again.")
                        
                        sys.stdout = old_stdout
                        output = mystdout.getvalue()
                        
                        update_progress_bar(90, "Displaying the results...", progress_bar)
                        if output:
                            st.write(output)
                        for key, value in exec_locals.items():
                            if isinstance(value, (pd.DataFrame, pd.Series)):
                                st.write(value)
                            else:
                                st.write({key: value})
                        update_progress_bar(100, "Done.", progress_bar)
                        st.write("Direct output from LLM response:")
                        st.write(llm_response)
                    except Exception as e:
                        st.write("Direct output from LLM response:")
                        st.write(llm_response)
                        st.error(f"Error executing code: {e}")

            col1, col2 = st.columns(2)
            with col1:
                thumbs_up = st.button('👍', key='csv_thumbs_up')
            with col2:
                thumbs_down = st.button('👎', key='csv_thumbs_down')

            if thumbs_up:
                store_feedback(user_data_question, 1)
                st.success("Thank you for your feedback!")

            if thumbs_down:
                store_feedback(user_data_question, 0)
                st.error("Thank you for your feedback! We'll work to improve.")
        elif user_data_question in ("no", "No"):
            st.write("Thanks")

        st.header("CSV Plot Interaction")
        user_plotting_question = st.text_input("Enter your plotting question", max_chars=300, key="csv_user_plotting_question")

        if user_plotting_question:
            if user_plotting_question not in ("", "no", "No"):
                with st.spinner('Analysing..'):
                    progress_bar = st.progress(0)
                    update_progress_bar(10, "Initializing LLM connection...", progress_bar)
                    llm_response = csv_answer_plotting_question(user_plotting_question)
                    update_progress_bar(50, "Processing your request...", progress_bar)
                    # st.write("Generated Python Code:")
                    # st.code(llm_response, language="python")
                    update_progress_bar(70, "Running the analysis...", progress_bar)

                    if "Agent stopped due to iteration limit or time limit." in llm_response:
                        st.error(random.choice(friendly_errors))
                    else:
                        try:
                            exec_globals = {"df": df, "px": px, "pd": pd, "st": st}
                            exec_locals = {}

                            clean_response = llm_response.strip("```").strip("python")

                            # Verify column names
                            for line in clean_response.split("\n"):
                                if "df['" in line:
                                    col_name = line.split("df['")[1].split("']")[0]
                                    if col_name not in df.columns:
                                        raise ValueError(f"Invalid column name in the generated code: {col_name}")

                            update_progress_bar(70, "Running the analysis...", progress_bar)
                            exec(clean_response, exec_globals, exec_locals)
                            
                            update_progress_bar(90, "Displaying the results...", progress_bar)
                            if 'fig' in exec_locals:
                                st.plotly_chart(exec_locals['fig'])
                            else:
                                st.write(exec_locals)
                            update_progress_bar(100, "Done.", progress_bar)
                        except Exception as e:
                            st.write(random.choice(friendly_errors))
            elif user_plotting_question in ("no", "No"):
                st.write("Thanks")