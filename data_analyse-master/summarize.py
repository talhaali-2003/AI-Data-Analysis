import os
import unicodedata
import re
import pprint
from langchain_openai import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredHTMLLoader
import nltk
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
from langchain.schema.document import Document
from prompt import map_prompt_template, combine_prompt_template, daily_prompt, question_prompt, refine_prompt, prompt_stuff

load_dotenv()

os.environ["OPENAI_API_VERSION"]= os.getenv("OPENAI_API_VERSION")
os.environ["AZURE_OPENAI_ENDPOINT"]= os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
#model_name = os.getenv('MODEL_NAME', 'gpt-35-turbo-16k')



def replace_accented_characters(text):
    # Replace French accented characters with their English equivalents
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
    return text

def remove_pattern0(text):
    # Define the regex pattern to match the specific format
    # E525385 PAMPILLON DOMINGUEZ CONCEPCION *** HDCE/FD *** HRT/ICT/GTO/EUES *** INITIAL/OPEN (Interaction)
    # Adjust the pattern as necessary to match all variations you need to remove
    pattern = r'([A-Z0-9]+ [A-Z ]+ \*\*\* [A-Z/]+ \*\*\* [A-Z/]+ \*\*\* [A-Z/]+ \(.*?\))'

    match = re.search(pattern, text)
    if match:
        #print("Match found:", match.group())
        text = re.sub(pattern, '', text)
        #print("Text after removing pattern:", text)
    
    return text

def remove_pattern1(text):
    # Define the regex pattern to match the specific format
    # E525385 PAMPILLON DOMINGUEZ CONCEPCION *** HDCE/FD *** HRT/ICT/GTO/EUES *** INITIAL/OPEN
    # Adjust the pattern as necessary to match all variations you need to remove
    pattern = r'([A-Z0-9]+ [A-Z\s]+ \*\*\* ([A-Z\/]+ \*\*\* )+[A-Z]+)'

    match = re.search(pattern, text)
    if match:
        #print("Match found:", match.group())
        text = re.sub(pattern, '', text)
        #print("Text after removing pattern:", text)
    
    return text

def remove_pattern2(text):
    # Define the regex pattern to match the specific format
    # E563457 ALICIA REBOREDA DELGADO *** OAP - UPDATE
    # Adjust the pattern as necessary to match all variations you need to remove
    pattern = r'([A-Z0-9]+) ([A-Z\s]+) \*\*\* ([A-Z\s\-]+)'

    match = re.search(pattern, text)
    if match:
        #print("Match found:", match.group())
        text = re.sub(pattern, '', text)
        #print("Text after removing pattern:", text)
    
    return text


def remove_pattern3(text):
    # Define the regex pattern to match the specific format
    # [OAP1708319328735I111145]
    # Adjust the pattern as necessary to match all variations you need to remove
    pattern = r'(\[[A-Z]+[0-9]+[A-Z][0-9]+\])'

    match = re.search(pattern, text)
    if match:
        #print("Match found:", match.group())
        text = re.sub(pattern, '', text)
        #print("Text after removing pattern:", text)
    
    return text


def remove_pattern4(text):
    # Define the regex pattern to match the specific format
    # *** MZPAOP00 *** Update OPER/BE by WS
    # Adjust the pattern as necessary to match all variations you need to remove
    pattern = r'(MZPAOP00 \*\*\* Update OPER\/BE by WS)'

    match = re.search(pattern, text)
    if match:
        #print("Match found:", match.group())
        text = re.sub(pattern, '', text)
        #print("Text after removing pattern:", text)
    
    return text


def replace_accented_characters(text):
    # Replace French accented characters with their English equivalents
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
    return text


def daily_cleaning(text):
    cleaned_text = ""
    pattern = r'\*\*\* ADD SYNTHESIS \*\*\*(.*?)\*\*\* END SYNTHESIS \*\*\*'

    match = re.search(pattern, text, re.DOTALL)

    # Extracting and printing the matched text if found
    if match:
        extracted_text = match.group(1).strip()  # .strip() to remove leading/trailing whitespace
        return extracted_text
    else:
        return "No match found."
        
    


def cleaning(text):

  cleaned_text = ""

  for line in text.split('\n'):
    line = remove_pattern0(line)
    line = remove_pattern1(line)
    line = remove_pattern2(line)
    line = remove_pattern3(line)
    line = remove_pattern4(line)

    line = line.lower()
    
    cleaned_text += line + '\n'
  
  #nltk_tokens = nltk.sent_tokenize(cleaned_text)
  #print(f"nb of token : {len(nltk_tokens)}")

  return cleaned_text.strip()



def get_summarize(text, chain_type, model, temperature):

    llm = AzureChatOpenAI(
        deployment_name=model,
        temperature=temperature
    )
    

    try:
      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size = 1000,
          chunk_overlap  = 200,
          length_function = len,
          separators = ["\n", "\r\n"]
      )

      nb_token = 0
      docs = [Document(page_content=x) for x in text_splitter.split_text(text)]

      #Cleaning Text
      for doc in docs:
        doc.page_content = cleaning(doc.page_content)
        nb_token += llm.get_num_tokens(doc.page_content) 

      if chain_type == "map_reduce":

        summary_chain = load_summarize_chain(llm=llm,
                                            chain_type=chain_type,
                                            map_prompt=map_prompt_template,
                                            combine_prompt=combine_prompt_template,
                                            verbose=False
                                            )
      elif chain_type == "refine":
        summary_chain = load_summarize_chain(llm=llm,
                                            chain_type=chain_type,
                                            question_prompt=question_prompt,
                                            refine_prompt=refine_prompt,
                                            return_intermediate_steps=False,
                                            verbose=False
                                            )
      elif chain_type == "stuff":
        summary_chain = load_summarize_chain(llm=llm,
                                            chain_type=chain_type,
                                            prompt=prompt_stuff,
                                            verbose=False
                                            )

      output = summary_chain.invoke(docs)
      
      result={}
      result["summary"] = output["output_text"]
      result["nb_token"] = nb_token

      return result

    
    except Exception as e:
      return f"Error during LLM request : {str(e)}"

    

def get_day_summarize(text, model, temperature):

    llm = AzureChatOpenAI(
        deployment_name=model,
        temperature=temperature
    )
    
    extraction = daily_cleaning(text)

    try:
      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size = 1000,
          chunk_overlap  = 200,
          length_function = len,
          separators = ["\n", "\r\n"]
      )
      
      nb_token = llm.get_num_tokens(extraction)

      docs = [Document(page_content=x) for x in text_splitter.split_text(extraction)]

      stuff_chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=daily_prompt)

      output = stuff_chain.invoke(docs)
      
      result={}
      result["summary"] = output["output_text"]
      result["nb_token"] = nb_token


      return result
      
    
    except Exception as e:
      return f"Error during LLM request : {str(e)}"
      
      
def get_token(text, model, temperature):

    llm = AzureChatOpenAI(
        deployment_name=model,
        temperature=temperature
    )
    
    return llm.get_num_tokens(text)