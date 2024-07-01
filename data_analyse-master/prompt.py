from langchain.prompts import PromptTemplate



############### MAP REDUCE  ############################

map_prompt = """
Write a summary of the following:
"{text}"
"""

map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])


combine_prompt = """
You're an IT communication expert, write a professional concise summary of the following text :

"{text}"

Get to the point and use the following template :

Start:
End:
Duration:

**IMPACT:
**ROOT CAUSE:
**SUMMARY:  Use bullet point here
**RESOLUTION:
"""

combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])


############### DAILY PROMPT (STUFF)  ############################


daily_report_prompt = """
You're an IT communication expert, write a concise professional summary of the following text, go straight to the point.
Answer like following examples.

"{text}"

Example #1
Access to the Cherwell application is not possible due to a database issue. The STAR Case Management System for Concessionary Service Technician Support is unavailable. Service restored when support teams performed a fail over of the database and restarted SH servers.

Example #2
Accessing the application is impossible using https://refvr.inetpsa.com/rvr/accueil.action due to indus refresh on the two DB slave servers. Users can not provide any finance calculation for all brands (New and Used vehicles). Service restored when the applicative expert redirected all web-services request to the master database.

Example #3
Crisis opened because production is stopped at Rennes plant (FRANCE). QUALIF/RJ (Application of Quality in Manufacturing RJ) is also impacted due to a full filesystem (core files). Service restored when the applicative experts released space on the filesystem and reorganized the production between applications.
"""

daily_prompt = PromptTemplate.from_template(daily_report_prompt)


############### REFINE ############################



question_prompt_template = """
You're an IT communication expert, try to write a professional concise summary of the following text.
Use bullet point for the summary. Get to the point and use the following template
"{text}"

Start:
End:
Duration:

**IMPACT:
**ROOT CAUSE:
**SUMMARY: 
**RESOLUTION:
"""

question_prompt = PromptTemplate(template=question_prompt_template, input_variables=["text"])

refine_prompt_template = """
              Write a concise summary of the following text delimited by triple backquotes.
              Return your response in bullet points which covers the key points of the text.
              ```{text}```
              BULLET POINT SUMMARY:
              """

refine_prompt = PromptTemplate(template=refine_prompt_template, input_variables=["text"])



######################## STUFF  template  #################################

stuff_prompt_template = """
You're an IT communication expert, write a professional concise summary of the following text.Get to the point.
Try to extract the start date and time of the incident. 
Try to find the end date and time of the incident otherwise say ongoing.
Use the following template to answer and you will find below an example.

"{text}"

Template#

Start:
End:
Duration:


IMPACT:
ROOT CAUSE:
SUMMARY: 
RESOLUTION:

Example #1

Start : 2024/03/04 13:12
End : 2024/03/04 14:40
Duration : (00d) 01:27
Calls generated : 0
Users involved : 0


IMPACT
- Problem description :  Accessing the application is impossible using https://refvr.inetpsa.com/rvr/accueil.action.

************************
- Users impact :   Users can not provide any finance calculation for all brands (New and Used vehicles).

- Business / Job impact:  Major


ROOT CAUSE
- Change or incident potentially linked?  INCI12156102, INCI12155405
- Previous automatic incident / lack of monitoring?  NO
- Root Cause:   Indus launched a synchronisation on the two DB slave servers in the same time.


SUMMARY
- On  Friday that was asked indus to synchronize the 2  slave Database to master Database.(INCI12156102)
- He did that actions on the two servers in same time that impact the proper functioning of the application today
- This refresh action can take 2 hours.

The envisaged solution consists of pointing webservice requests only to the master database while the two slave database servers are resynchronized.

SOLUTION
Service restored when the applicative expert redirected all web-services request to the master database.

"""

prompt_stuff = PromptTemplate.from_template(stuff_prompt_template)