import streamlit as st
import os

# --- UI Setup ---
st.set_page_config(page_title="Agentic Research Assistant", page_icon="🔍")
st.title("Agentic Research Assistant")
st.markdown("Generate deep research reports using a multi-agent workflow (Planner, Researcher, Writer, Editor).")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    topic = st.text_input("Research Topic", placeholder="e.g., Attention Is All You Need")
    run_button = st.button("Start Research")


# --- Standard library
from datetime import datetime
import re
import json
import ast


# --- Third-party --
from IPython.display import Markdown, display

import streamlit as st
from openai import OpenAI
import os

# Streamlit Cloud uses st.secrets to manage API keys securely
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
except Exception:
    # Fallback for local testing or Colab if secrets aren't set in Streamlit yet
    from google.colab import userdata
    GROQ_API_KEY = userdata.get('GROQ_API_KEY')
    TAVILY_API_KEY = userdata.get('TAVILY_API_KEY')

# Set the Tavily API key as an environment variable for the research tools
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY

CLIENT = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

MODEL_NAME = "llama-3.1-8b-instant"
LARGE_MODEL_NAME = "llama-3.3-70b-versatile"

# --- Standard library ---
import os
import xml.etree.ElementTree as ET

# --- Third-party ---
import requests
from dotenv import load_dotenv
from tavily import TavilyClient
import wikipedia

# Init env
load_dotenv()  # load variables

# Set user-agent for requests to arXiv
session = requests.Session()
session.headers.update({
    "User-Agent": "LF-ADP-Agent/1.0 (mailto:your.email@example.com)"
})

def arxiv_search_tool(query: str, max_results: int = 5) -> list[dict]:
    """
    Searches arXiv for research papers matching the given query.
    """
    url = f"https://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"

    try:
        response = session.get(url, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return [{"error": str(e)}]

    try:
        root = ET.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        results = []
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.strip()
            authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
            published = entry.find('atom:published', ns).text[:10]
            url_abstract = entry.find('atom:id', ns).text
            summary = entry.find('atom:summary', ns).text.strip()

            link_pdf = None
            for link in entry.findall('atom:link', ns):
                if link.attrib.get('title') == 'pdf':
                    link_pdf = link.attrib.get('href')
                    break

            results.append({
                "title": title,
                "authors": authors,
                "published": published,
                "url": url_abstract,
                "summary": summary,
                "link_pdf": link_pdf
            })

        return results
    except Exception as e:
        return [{"error": f"Parsing failed: {str(e)}"}]


arxiv_tool_def = {
    "type": "function",
    "function": {
        "name": "arxiv_search_tool",
        "description": "Searches for research papers on arXiv by query string.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for research papers."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}

x=datetime.now()
x=x.strftime("%c")
TIME_INJESTION=f"Now, It is {x}. You should make use of the time, if needed"
print(x)

def tavily_search_tool(query: str, max_results: int = 5, include_images: bool = False) -> list[dict]:
    """
    Perform a search using the Tavily API.

    Args:
        query (str): The search query.
        max_results (int): Number of results to return (default 5).
        include_images (bool): Whether to include image results.

    Returns:
        list[dict]: A list of dictionaries with keys like 'title', 'content', and 'url'.
    """
    params = {}
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not found in environment variables.")
    params['api_key'] = api_key

    client = TavilyClient(api_key)

    try:
        response = client.search(
            query=query,
            max_results=max_results,
            include_images=include_images
        )

        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "url": r.get("url", "")
            })

        if include_images:
            for img_url in response.get("images", []):
                results.append({"image_url": img_url})

        return results

    except Exception as e:
        return [{"error": str(e)}]  # For LLM-friendly agents


tavily_tool_def = {
    "type": "function",
    "function": {
        "name": "tavily_search_tool",
        "description": "Performs a general-purpose web search using the Tavily API.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for retrieving information from the web."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5
                },
                "include_images": {
                    "type": "boolean",
                    "description": "Whether to include image results.",
                    "default": False
                }
            },
            "required": ["query"]
        }
    }
}

# # Wikipedia search tool

def wikipedia_search_tool(query: str, sentences: int = 5) -> list[dict]:
    """
    Searches Wikipedia for a summary of the given query.

    Args:
        query (str): Search query for Wikipedia.
        sentences (int): Number of sentences to include in the summary.

    Returns:
        list[dict]: A list with a single dictionary containing title, summary, and URL.
    """
    try:
        page_title = wikipedia.search(query)[0]
        page = wikipedia.page(page_title)
        summary = wikipedia.summary(page_title, sentences=sentences)

        return [{
            "title": page.title,
            "summary": summary,
            "url": page.url
        }]
    except Exception as e:
        return [{"error": str(e)}]

# Tool definition
wikipedia_tool_def = {
    "type": "function",
    "function": {
        "name": "wikipedia_search_tool",
        "description": "Searches for a Wikipedia article summary by query string.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for the Wikipedia article."
                },
                "sentences": {
                    "type": "integer",
                    "description": "Number of sentences in the summary.",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}



# Tool mapping
tool_mapping = {
    "tavily_search_tool": tavily_search_tool,
    "arxiv_search_tool": arxiv_search_tool,
    "wikipedia_search_tool": wikipedia_search_tool
}

def planner_agent(topic: str, model: str = MODEL_NAME) -> list[str]:


    """
    Generates a plan as a Python list of steps (strings) for a research workflow.

    Args:
        topic (str): Research topic to investigate.
        model (str): Language model to use.

    Returns:
        List[str]: A list of executable step strings.
    """
    print("==================================")
    print("Planner Agent")
    print("==================================")

    user_prompt = f"""
    You are a planning agent responsible for organizing a research workflow. Your job is to write a clear, step-by-step research plan as a valid Python list, where each step is a string.{TIME_INJESTION}

    STRICT RULES:
    1. ONE AGENT PER STEP: Each step MUST use EXACTLY ONE agent.
    2. NO CHAINING: Never use 'then', 'and', 'followed by', or ';' to link tasks for different agents in one step.
    3. ATOMICITY: If you need an editor to revise AND a writer to finalize, these MUST be two separate steps.
    4. Format: 'Use the [agent_name] to [specific task]'.
    5. Do not extend more than 4 steps

    Available agents:
    - research_agent (searches web, Wikipedia, and arXiv)
    - writer_agent (drafts and writes content)
    - editor_agent (reflects, critiques, and suggests improvements)

    Constraints:
    - Do not exceed 5 steps total.
    - The editor_agent must be called at least once.
    - The final step must be the writer_agent generating the complete research report.
    - Do not include explanation text — return ONLY the Python list.

    Example of BAD step: \"Use the editor_agent to revise, then use the writer_agent to finalize.\"
    Example of GOOD steps:
    \"Use the editor_agent to provide feedback on the draft.\",
    \"Use the writer_agent to generate the final report based on feedback.\"

    Topic: \"{topic}\"
    """

    # Add the user prompt to the messages list
    messages = [{"role": "user", "content": user_prompt}]


    # Call the LLM
    response = CLIENT.chat.completions.create(
        model= model,
        messages= messages,
        temperature=0,
    )



    # Extract message from response
    steps_raw_str = response.choices[0].message.content.strip()

    # Attempt to clean markdown formatting first using the existing clean_json_block
    steps_cleaned_str = clean_json_block(steps_raw_str)

    # Now, try to extract only the list literal part by finding the first '[' and last ']'
    list_start = steps_cleaned_str.find('[')
    list_end = steps_cleaned_str.rfind(']')

    if list_start != -1 and list_end != -1 and list_end > list_start:
        steps_str = steps_cleaned_str[list_start : list_end + 1]
    else:
        # Fallback
        steps_str = steps_cleaned_str

    # Parse steps
    steps = ast.literal_eval(steps_str)

    return steps

import json

def research_agent(task, model=MODEL_NAME):
    print("==================================")
    print("🔍 Research Agent")
    print("==================================")

    # 1. Prepare the tools in the format Grok expects
    # Note: Using global tool definitions as they were copy-pasted locally
    tools_spec = [
        arxiv_tool_def,
        tavily_tool_def,
        wikipedia_tool_def
    ]

    system_prompt = (
        f"You are a helpful research assistant. Use available tools to find accurate information. {TIME_INJESTION}"
        "IMPORTANT: For any tool call, set 'max_results' to a maximum of 5 to stay within API limits. "
        "STRICT NEGATIVE CONSTRAINTS: "
        "1.Do not exceed 3 total tool calls per task to conserve tokens."
        "2.Do not call tool that is not available like attempting to call 'brave_search'"
    )


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task}
    ]

    # Manual Turn Loop (up to 6 turns)
    for _ in range(6):
        response = CLIENT.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools_spec,
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        messages.append(response_message)

        # Check if the model wants to call a tool
        if not response_message.tool_calls:
            break

        # Process each tool call
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"🛠️ Calling tool: {function_name}({function_args})")

            # Use the global tool_mapping
            function_to_call = tool_mapping[function_name]
            tool_output = function_to_call(**function_args)

            # Add the tool result to messages
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(tool_output)
                }
            )

    return response_message.content

def writer_agent(task: str, model: str = MODEL_NAME) -> str:
    """
    Executes writing tasks, such as drafting, expanding, or summarizing text.
    """
    print("==================================")
    print("✍️ Writer Agent")
    print("==================================")


    # Specialized in academic content without meta-talk, ratings, or process notes.
    system_prompt = (
        f"You are a research report writing agent specialized in generating professional academic and technical research reports.{TIME_INJESTION} "
        "Your goal is to provide a clean final document based on the provided context and feedback. "
        "STRICT NEGATIVE CONSTRAINTS: "
        "1. Do NOT include any 'Style and Grammar' or 'Consistency' review sections. "
        "2. Do NOT include any 'Rating' or scores (e.g., '9 out of 10'). "
        "3. Do NOT include meta-talk about the writing process or mentions of removing requested sections (e.g., 'I have removed the appendix'). "
        "4. Do NOT include an Appendix header or placeholder if it is empty. "
        "5. Do NOT include any self-evaluation or concluding remarks about the report's own quality."
        "6. Do NOT include templates like 'References:[List of references with corresponding DOIs or URLs]'"
        "7. Do NOT give the return the result as plain text instead of report structured markdown"
    )

    # Define the system msg by using the system_prompt and assigning the role of system
    system_msg = {"role" : "system", "content" : system_prompt}

    # Define the user msg. In this case the user prompt should be the task passed to the function
    user_msg = {"role" : "user", "content" : task}

    # Add both system and user messages to the messages list
    messages = [system_msg, user_msg]


    response = CLIENT.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1.0
    )

    return response.choices[0].message.content

def editor_agent(task: str, model: str = MODEL_NAME) -> str:
    """
    Executes editorial tasks such as reflection, critique, or revision.
    """
    print("==================================")
    print("🧠 Editor Agent")
    print("==================================")

    # This should assign the LLM the role of an editor agent specialized in reflecting on, critiquing, or improving existing drafts.
    system_prompt = (f"You are a research article editor. Your task is to reflect on, critique, or improve drafts. Focus on meaningfulness, structure, coherence of the report.{TIME_INJESTION}"
                    "STRICT NEGATIVE CONSTRAINTS: "
                    "Do Not miss to suggest the removal of appendix,if is at the last." 
                    "Do Not miss to suggest the removal of missing contents contents."
                    "Do Not miss to suggest the removal of random code." 
                    "Do Not miss to suggest the removal of repeated contents."
                    )

    # Define the system msg by using the system_prompt and assigning the role of system
    system_msg = {"role" : "system", "content" : system_prompt}

    # Define the user msg. In this case the user prompt should be the task passed to the function
    user_msg = {"role" : "user", "content" : task}

    # Add both system and user messages to the messages list
    messages = [system_msg , user_msg]

    response = CLIENT.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7
    )

    return response.choices[0].message.content

agent_registry = {
    "research_agent": research_agent,
    "editor_agent": editor_agent,
    "writer_agent": writer_agent,
}

def clean_json_block(raw: str) -> str:
    """
    Clean the contents of a JSON block that may come wrapped with Markdown backticks.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()

def executor_agent(topic, plan_steps, model: str = MODEL_NAME, limit_steps: bool = True):

    max_steps = 5

    N=len(plan_steps)
    
    agent_model=model

    print(f"Number of steps in initial plan: {N}")

    if limit_steps:
        plan_steps = plan_steps[:min(len(plan_steps), max_steps)]

    history = []

    print("==================================")
    print("🎯 Executor Agent")
    print("==================================")

    for plan in plan_steps:
        print(plan)

    for i, step in enumerate(plan_steps):

        agent_decision_prompt = f"""
        You are an execution manager for a multi-agent research team.

        {TIME_INJESTION}

        Given the instruction, identify which agent should perform it and extract the clean task.

        Do not change the agent suggested in the instruction.

        Available agents:
        - A research agent who can search the web, Wikipedia, and arXiv.
        - A writer agent who can draft research summaries.
        - An editor agent who can reflect and revise the drafts.


        Choose agent approprietly.


        Return only a valid JSON object with two keys:
        - "agent": one of ["research_agent", "editor_agent", "writer_agent"]
        - "task": a string with the instruction that the agent should follow

        Only respond with a valid JSON object. Do not include explanations or markdown formatting.

        Instruction: "{step}"
        """
        response = CLIENT.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": agent_decision_prompt}],
            temperature=0,
        )

        raw_content = response.choices[0].message.content
        cleaned_json = clean_json_block(raw_content)
        agent_info = json.loads(cleaned_json)

        agent_name = agent_info["agent"]
        task = agent_info["task"]


        if agent_name in agent_registry:
            
            context=""

            if agent_name=="research_agent":
                
              agent_model=LARGE_MODEL_NAME 

            else:
                
                context="Here is the context of what has been done so far:"
                
                if agent_name=="editor_agent":
            
                    context = context +"\n"+ "\n".join([
                        f"Step {j+1} executed by {a}:\n{r}"
                        for j, (s, a, r) in enumerate(history[-1:])
                    ])   
                    
                else:
                    
                    context = context +"\n"+ "\n".join([
                        f"Step {j+1} executed by {a}:\n{r}"
                        for j, (s, a, r) in enumerate(history)
                    ])    
                    
                    agent_model=LARGE_MODEL_NAME                         

            enriched_task = f"""
            You are {agent_name}.

            {context}

            Your next task is:
            {task}
            """

            print(f"\n🛠️ Executing with agent: `{agent_name}` on task: {task}")                    
                    
            output = agent_registry[agent_name](enriched_task, model=agent_model)
            history.append((step, agent_name, output))
        else:
            output = f"⚠️ Unknown agent: {agent_name}"
            history.append((step, agent_name, output))
            
        agent_model=model
        
        print(f"✅ Output:\n{output}")

    return history

# --- Logic Wrapper ---
if run_button and topic:
    status_container = st.container()
    with st.spinner(f"Planning research for: {topic}..."):
        try:
            # 1. Plan
            steps = planner_agent(topic)
            st.success("Plan done!")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            
    with st.spinner(f"Researching for: {topic}..."):
        try:

            # 1. Execute
            results = executor_agent(topic, limit_steps=True, plan_steps=steps)

            # 2. Display Final Output
            st.success("Research Complete!")
            if results:
                st.markdown(results[-1][2])


        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Enter a topic in the sidebar to begin.")