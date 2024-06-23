import os
import json
import requests
from dotenv import load_dotenv
load_dotenv()

import autogen
from autogen import UserProxyAgent, AssistantAgent
SERPER_API = os.getenv("SERPER")

# gpt-4o
config_list = [
        {
        "model": "gpt-4o", 
        "api_key": "api_key"
    }
]

# Define the Agents

code_interpreter = autogen.UserProxyAgent(
    "code_interpreter",
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    max_consecutive_auto_reply=1,
)

researcher = autogen.AssistantAgent(
    "researcher",
    system_message="""You are a helpful research assistant tasked with gathering information about the lead company and generating a research report with respect to our company value propositions. You can use the tools provided to you if required.
    <our company value propositions>
    VEnableAI is a leading AI startup that provides AI Agents, Chat Agents, and automates workflows for companies to drive efficiency, cost savings, and revenue growth.
    </our company value propositions>
    <task>
    You need to gather information about the company and generating a research report with respect to our company value propositions is highly required. The research report should include the following:
    1. Company Name
    2. Company Website
    3. Company Industry
    4. Comapny Value Propositions
    5. Company Description
    </task>
    """,
    llm_config={"config_list": config_list},
)


email_agent = autogen.AssistantAgent(
    "email_agent",
    system_message="""
    You are a helpful email agent who assists in drafting cold outreach emails to companies based on the information provided by the researcher. If the necessary information is not available, please request it from the researcher again.
    <task>
    Draft a cold email to the company using the information you have gathered from the researcher. You may use the provided template as a starting point, but ensure you tailor the email to reflect the specific value propositions and interests of the lead company.
    </task>
    <template>
    Hi [Company Name],

    I hope this message finds you well. My name is Hrushikesh, and I'm reaching out from VEnableAI, an AI-focused startup. I was excited to learn about your dedication to revolutionizing [company_value_propositions].

    Our clients aim to:
    1. Streamline operations and reduce costs through AI-driven automation.
    2. Enhance customer support with AI-powered chatbots and virtual assistants.
    3. Boost sales performance by leveraging AI insights and personalization.

    VEnableAI helps Fortune 500 companies as well as small and medium-sized enterprises to achieve these objectives with tailored AI solutions that drive efficiency, cost savings, and revenue growth.

    Could you let me know if you are available for a 20-minute conversation next week to explore this further?

    Best regards,
    Hrushikesh Dokala
    Lead GenAI Consultant,
    VEnableAI
    </template>
    """,
    llm_config={"config_list": config_list},
)


@code_interpreter.register_for_execution()
@researcher.register_for_llm(
    name="search_info", description="Get the information from the web about the company."
)
def search_info(company_name: str,
                company_url: str) -> str:

    query = f"gather information about {company_name} at {company_url}"

    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': SERPER_API,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    response_data = response.json()
    return response_data if response_data else "No information found."



# Put them in a virtual room
groupchat = autogen.GroupChat(
    agents=[code_interpreter, researcher, email_agent],
    messages=[],
    allow_repeat_speaker=True,
    max_round=15,
)


manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config={
        "config_list": config_list,
    },
)


# Give it a task or assign a goal

task = "Get the information about Cred from the web, this is their website: cred.club"
task1 = "Get the information about 18startup from the web, this is their website: 18startup.com"
task2 = "Get the information about Myfuturely from the web, this is their website: myfuturely.com"

user_proxy = UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    code_execution_config=False,
    default_auto_reply="",
)

# start the agents to work

user_proxy.initiate_chat(manager, message=task)
