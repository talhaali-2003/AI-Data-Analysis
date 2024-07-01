import streamlit as st
from get_elastic import get_metrics, get_process
import os
import json
import time
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import AzureChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain_experimental.agents.agent_toolkits import create_python_agent

##Chat
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
                                SystemMessagePromptTemplate,
                                HumanMessagePromptTemplate,
                                ChatPromptTemplate,
                                MessagesPlaceholder
)
from streamlit_chat import message
from auxiliary_functions import *


es_host = os.getenv('ES_HOST', '')
es_port = int(os.getenv('ES_PORT', 9250))
es_scheme = os.getenv('SCHEME', 'https')
index_name = os.getenv('INDEX', '')
user = os.getenv('username')
model_name = os.getenv('MODEL_NAME', 'gpt-35-turbo-16k')
password = os.getenv('pass')


results_folder = "results"
results_file = os.path.join(results_folder, "summary_results.json")
results_file_day = os.path.join(results_folder, "summary_day_results.json")

######## COMMMON FUNCTIONS ############


def function_question_dataframe():
    dataframe_info = pandas_agent.run(user_question_dataframe)
    st.write(dataframe_info)
    return
    
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

@st.cache_resource
def python_agent():
    agent_executor = create_python_agent(
        llm=llm,
        tool=PythonREPLTool(),
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
    )
    return agent_executor
    
@st.cache_resource
def python_agent_tab2():
    agent_executor = create_python_agent(
        llm=llm,
        tool=PythonREPLTool(),
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
    )
    return agent_executor

@st.cache_data
def python_solution(my_data_problem, selected_algorithm, user_csv):
    solution = python_agent().run(f"Write a Python script to solve this: {my_data_problem}, using this algorithm: {selected_algorithm}, using current dataset. Using tool python_repl_ast"
    )
    return solution



#Function to udpate the value in session state
def clicked(button):
    st.session_state.clicked[button]= True



######## MAIN ############

#Initialise the key in session state
if 'clicked' not in st.session_state:
    st.session_state.clicked ={1:False}
    


st.set_page_config(page_title="Data Analysis", page_icon="", layout="wide")
st.title("POC for Data analysis", anchor=False)


with st.sidebar:

    st.write('*General options.*')
    st.caption('''**Choose a LLM to use**''')    
    
    model = st.selectbox("Select Model", options=["gpt-35-turbo-16k", "gpt-4"], key="model_selection")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1, key="temperature_slider") 

    st.write('*ChatBot.*')
    st.caption('''**Chatbot based on knowledge base loaded into 2 kind of database. Purpose is to try relevant answer for 2 database oriented for AI**
    ''')    
    
    options = [ 'Chroma', 'Pinecone']
    selected_options = st.selectbox(
        'Select one database:',
        options
    )


# Tabs for Summarize and Daily Reports
tab1, tab2, tab3 = st.tabs(["Data analyse", "Data analyse CSV" ,"ChatBot"])

llm = AzureChatOpenAI(
    #deployment_name="gpt-35-turbo-16k",
    deployment_name=model,
    temperature=temperature
)   

######## TAB1 ############
with tab1:

    st.header(f"AI Assistant for Data Science ({model})")
    
    left_column, right_column = st.columns([2, 1]) 

    #with right_column:
    #    with st.form(key='my_form'):
    #        model = st.selectbox("Select Model", options=["gpt-35-turbo-16k", "gpt-4"], key="model_selection")
    #        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1, key="temperature_slider") 

     #       submit_button = st.form_submit_button(label='Save')


    with left_column:
        st.divider()
        
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
        #st.write("Start DateTime:", start_datetime_str)
        #st.write("End DateTime:", end_datetime_str)
        
        user_question_host = st.text_input("What hostname are you interested in ?", key="hostname")
        
        st.divider()
        
                    
        if user_question_host and start_date and end_date:  # Checks if incident_number is not empty 
            with st.spinner('Pulling hostname datas...'):
                datas = get_metrics(user_question_host, start_datetime_str, end_datetime_str)
                datas_process = get_process(user_question_host, start_datetime_str, end_datetime_str)
                print(datas_process[1]["fields"])
                

            st.success('Datas pulled !')
            
            if datas is not None: 
                # Convert to DataFrame
                df = pd.DataFrame(datas)
                #df_process = pd.DataFrame(datas_process)
                records = []
                
                for item in datas_process:
                    fields = item["fields"]
                    record = {key: value[0] if isinstance(value, list) and len(value) == 1 else value 
                              for key, value in fields.items()}
                    records.append(record)
                
                df_process = pd.DataFrame(records)
                df_process['TIMESTAMP'] = pd.to_datetime(df_process['TIMESTAMP'])
                
                # Extract and expand the specific fields
                df_filtered_process = df_process[['TIMESTAMP', 'command', 'pid', 'tid', 'cpu_percent', 'mem_percent','time' , 'elapsed', 'start_time', 'nice', 'pri' , 'state', 'user', 'ruser', 'rss', 'vsz' , 'sz', 'psr', 'os']].copy()
                
                

                # Extract and expand the specific fields
                df['CPU'] = df['fields'].apply(lambda x: x['CPU'][0] if 'CPU' in x else None)
                df['SWAP'] = df['fields'].apply(lambda x: x['SWAP'][0] if 'SWAP' in x else None)
                df['RAM'] = df['fields'].apply(lambda x: x['RAM'][0] if 'RAM' in x else None)
                df['TIMESTAMP'] = df['fields'].apply(lambda x: x['TIMESTAMP'][0] if 'TIMESTAMP' in x else None)

                # Now, create a new DataFrame with just these specific columns
                df_filtered = df[['TIMESTAMP', 'SWAP', 'RAM', 'CPU']].copy()
                df_filtered['TIMESTAMP'] = pd.to_datetime(df_filtered['TIMESTAMP'])
                
                #df_filtered = df_filtered.sort_values('TIMESTAMP')
                #df_filtered.to_csv('data.csv', index=True)
                
                #concatenated_df = pd.merge(df_filtered, df_process, on='TIMESTAMP', how='inner')
                df_filtered = df_filtered.sort_values('TIMESTAMP')
                df_filtered_process = df_filtered_process.sort_values('TIMESTAMP')

                #merged_df = pd.merge_asof(df_filtered, df_process, on='TIMESTAMP')
                merged_df = pd.concat([df_filtered, df_filtered_process], ignore_index=True, sort=False)

                
                pandas_agent = create_pandas_dataframe_agent(
                    #AzureChatOpenAI(deployment_name="gpt-35-turbo-16k",temperature=0),
                    llm,
                    merged_df,
                    verbose=True,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    agent_executor_kwargs={"handle_parsing_errors": True}
                )
                
               
                  #Functions main
                @st.cache_data
                def function_agent():
         
                    st.write("**Data Overview**")
                    st.write("The first rows of your dataset look like this:")
                    st.write(df_filtered.head())
                    st.write(len(df_filtered))
                    st.write(df_filtered_process.head())
                    st.write(len(df_filtered_process))
                    #st.write(merged_df.head())
                    #st.write(len(merged_df))
                    st.write("**Data Cleaning**")
                    columns_df = pandas_agent.run("What are the meaning of the columns?")
                    st.write(columns_df)
                    missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
                    st.write(missing_values)
                    #duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
                    #st.write(duplicates)
                    timeframe = pandas_agent.run("Calculate the timeframe of the dataframe?")
                    st.write(timeframe)
                    #row = pandas_agent.run("How many row ?")
                    #st.write(row)
                    st.write("**Data Summarisation**")
                    #st.write(merged_df.describe())
                    correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships using tool python_repl_ast.")
                    st.write(correlation_analysis)
                    #outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
                    #st.write(outliers)
                    new_features = pandas_agent.run("What new features would be interesting to create? .")
                    st.write(new_features)
                    return
                    
                def test():
                    # Plotting the trends over time for SWAP, RAM, and CPU usage
                    plt.figure(figsize=(14, 8))

                    plt.subplot(3, 1, 1)
                    plt.plot(df_filtered.index, df_filtered['SWAP'], label='SWAP', color='blue')
                    plt.ylabel('SWAP Usage')
                    plt.legend()

                    plt.subplot(3, 1, 2)
                    plt.plot(df_filtered.index, df_filtered['RAM'], label='RAM', color='green')
                    plt.ylabel('RAM Usage')
                    plt.legend()

                    plt.subplot(3, 1, 3)
                    plt.plot(df_filtered.index, df_filtered['CPU'], label='CPU', color='red')
                    plt.ylabel('CPU Usage')
                    plt.xlabel('Timestamp')
                    plt.legend()

                    plt.tight_layout()
                    
                    st.pyplot(plt)
                
                @st.cache_data
                def distribution():
                                   

                    # Convert the TIMESTAMP column to datetime format
                    df_filtered['TIMESTAMP'] = pd.to_datetime(df_filtered['TIMESTAMP'])

                    # Sort the dataframe by TIMESTAMP in ascending order
                    my_df = df_filtered.sort_values('TIMESTAMP')

                    # Streamlit app starts here
                    st.title('CPU Trends')

                    # Plotting the line graph using Streamlit
                    st.line_chart(my_df.set_index('TIMESTAMP')['CPU'])
                    
                    return
                
                @st.cache_data
                def function_question_variable():
                    st.line_chart(df_filtered.set_index('TIMESTAMP')[user_question_variable])
                    summary_statistics = pandas_agent.run(f"Run Give me a summary of the statistics of {user_question_variable} using tool Python REPL")
                    st.write(summary_statistics)
                    test()
                    normality = pandas_agent.run(f"Check plot for normality or specific distribution shapes of {user_question_variable}. Render a png file using tool Python REPL")
                    st.write(normality)
                    #distribution()
                    #outliers = pandas_agent.run(f" Assess the presence of outliers of {user_question_variable} using tool python_repl_ast")
                    #st.write(outliers)
                    #trends = pandas_agent.run(f"run Analyse trends of {user_question_variable} using tool python_repl_ast")
                    #st.write(trends)
                    return
                    
                  

                function_agent()
                
                st.subheader('Variable of study')
                user_question_variable = st.text_input('What variable are you interested in')
                if user_question_variable is not None and user_question_variable !="":
                    function_question_variable()
                    
                    st.subheader('Further study')
                    
                if user_question_variable:    

                    user_question_dataframe = st.text_input( "Is there anything else you would like to know about your dataframe?")
                    if user_question_dataframe is not None and user_question_dataframe not in ("","no","No"):
                        function_question_dataframe()
                    if user_question_dataframe in ("no", "No"):
                        st.write("")

                    if user_question_dataframe:
                        st.divider()
                        st.header("Data Science Problem")
                        st.write("Now that we have a solid grasp of the data at hand and a clear understanding of the variable we intend to investigate, it's important that we reframe our business problem into a data science problem.")
                        
                        prompt = st.text_area('What is the business problem you would like to solve?')
                        
                        if prompt:                    
                            wiki_research = wiki(prompt)
                            my_data_problem = chains_output(prompt, wiki_research)[0]
                            my_model_selection = chains_output(prompt, wiki_research)[1]
                                
                            st.write(my_data_problem)
                            st.write(my_model_selection)

                            formatted_list = list_to_selectbox(my_model_selection)
                            selected_algorithm = st.selectbox("Select Machine Learning Algorithm", formatted_list)

                            if selected_algorithm is not None and selected_algorithm != "Select Algorithm":
                                st.subheader("Solution")
                                solution = python_solution(my_data_problem, selected_algorithm, datas)
                                st.write(solution)



######## TAB2 ############
with tab2:

    st.header('AI Assistant for Data Science')
    
    left_column, right_column = st.columns([2, 1]) 

    with left_column:
        st.divider()
        
        
        user_csv = st.file_uploader("Upload your file here", type="csv")
        
        st.divider()
        
                    
        if user_csv is not None:
            with st.spinner('Reading csv file...'):
                user_csv.seek(0)
                df = pd.read_csv(user_csv, low_memory=False)
                
            st.success('dataframe ready !')


            pandas_agent = create_pandas_dataframe_agent(
                AzureChatOpenAI(deployment_name="gpt-35-turbo-16k",temperature=0),
                df,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                agent_executor_kwargs={"handle_parsing_errors": True}
            )
            
              #Functions main
            @st.cache_data
            def function_agent_tab2():
                st.write("**Data Overview**")
                st.write("The first rows of your dataset look like this:")
                st.write(df.head())
                st.write("**Data Cleaning**")
                columns_df = pandas_agent.run("What are the meaning of the columns?")
                st.write(columns_df)
                missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
                st.write(missing_values)
                duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
                st.write(duplicates)
                st.write("**Data Summarisation**")
                st.write(df.describe())
                correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships using tool python_repl_ast.")
                st.write(correlation_analysis)
                outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
                st.write(outliers)
                new_features = pandas_agent.run("What new features would be interesting to create? .")
                st.write(new_features)
                return
                
            
            @st.cache_data
            def function_question_variable_tab2():
                st.line_chart(df.set_index('TIMESTAMP')[user_question_variable_tab2])
                summary_statistics = pandas_agent.run(f"Run Give me a summary of the statistics of {user_question_variable_tab2} using tool Python REPL")
                st.write(summary_statistics)
                normality = pandas_agent.run(f"Check plot for normality or specific distribution shapes of {user_question_variable_tab2} using tool Python REPL")
                st.write(normality)
                outliers = pandas_agent.run(f" Assess the presence of outliers of {user_question_variable_tab2} using tool python_repl_ast")
                st.write(outliers)
                trends = pandas_agent.run(f"run Analyse trends of {user_question_variable_tab2} using tool python_repl_ast")
                st.write(trends)
                return
                
              

                

            function_agent_tab2()
            
            st.subheader('Variable of study')
            user_question_variable_tab2 = st.text_input('What variable are you interested in', key="user_question_variable_tab2")
            if user_question_variable_tab2 is not None and user_question_variable_tab2 !="":
                function_question_variable_tab2()
                
                st.subheader('Further study')
                
            if user_question_variable_tab2:    

                user_question_dataframe_tab2 = st.text_input( "Is there anything else you would like to know about your dataframe?", key="user_question_dataframe_tab2" )
                if user_question_dataframe_tab2 is not None and user_question_dataframe_tab2 not in ("","no","No"):
                    function_question_dataframe()
                if user_question_dataframe_tab2 in ("no", "No"):
                    st.write("")

                if user_question_dataframe_tab2:
                    st.divider()
                    st.header("Data Science Problem")
                    st.write("Now that we have a solid grasp of the data at hand and a clear understanding of the variable we intend to investigate, it's important that we reframe our business problem into a data science problem.")
                    
                    prompt_tab2 = st.text_area('What is the business problem you would like to solve?', key="prompt_tab2")
                    
                    if prompt_tab2:                    
                        wiki_research = wiki(prompt_tab2)
                        my_data_problem = chains_output(prompt_tab2, wiki_research)[0]
                        my_model_selection = chains_output(prompt_tab2, wiki_research)[1]
                            
                        st.write(my_data_problem)
                        st.write(my_model_selection)

                        formatted_list = list_to_selectbox(my_model_selection)
                        selected_algorithm = st.selectbox("Select Machine Learning Algorithm", formatted_list)

                        if selected_algorithm is not None and selected_algorithm != "Select Algorithm":
                            st.subheader("Solution")
                            solution = python_solution(my_data_problem, selected_algorithm, datas)
                            st.write(solution)


########TAB3 ############################
with tab3:
    
    st.header("ChatBox")
    st.write(f"Welcome to the AI Assistant ChatBox! you are using **{selected_options}** database") 

    st.write("")

    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["How can I assist you?"]
    if 'requests' not in st.session_state:
        st.session_state['requests'] = []


    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


    system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
    and if the answer is not contained within the text below, say 'I don't know'""")
    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
    prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

    conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

    response_container = st.container()
    textcontainer = st.container()

   
    with textcontainer:
        query = st.text_input("Hello! How can I help you? ", key="input")
        if query:
            with st.spinner("thinking..."):
                conversation_string = get_conversation_string()
                refined_query = query_refiner(conversation_string, query)
                st.subheader("Refined Query:")
                st.write(refined_query)
                if selected_options == "Chroma":
                    context = find_match_chroma(conversation_string)
                    print(context)
                else:
                    context = find_match(conversation_string)
                response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)

    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i],key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
                    

