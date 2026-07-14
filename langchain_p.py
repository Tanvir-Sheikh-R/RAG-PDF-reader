from langchain_groq import ChatGroq
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()
message = [SystemMessage('You are a helpfull assistent who give answers in one line')]

model = ChatGroq(
    model='llama-3.3-70b-versatile',
    temperature=0.3
)

@tool('web_search')
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"


@tool
def add(x: int, y: int) -> int:
    """add two numbers"""
    return x + y

tools = {'web_search': search, 'add': add}
model_with_tools = model.bind_tools([search, add])


while True:
    get_message : str = input('Enter your query... ')
    if get_message.lower() in ('exit', 'quit'):
        break

    message.append(HumanMessage(get_message))
    response = model_with_tools.invoke(message)
    message.append(response)

    if response.tool_calls:
        for call in response.tool_calls:
            tool_fn = tools[call['name']]
            result = tool_fn.invoke(call['args'])
            message.append(ToolMessage(content=str(result), tool_call_id=call['id']))

        response = model_with_tools.invoke(message)
        message.append(response)

    print(response.content)

print(message)