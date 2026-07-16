from langchain_groq import ChatGroq
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.callbacks import StreamingStdOutCallbackHandler




# import os
# import openai

# client = openai.OpenAI(
#   base_url="https://api.groq.com/openai/v1",
#   api_key=os.environ.get("GROQ_API_KEY")
# )



load_dotenv()
message = [SystemMessage('You are a helpfull assistent who give short answers')]

model = ChatGroq(
    model='openai/gpt-oss-120b',
    temperature=0.3,
    streaming=True, 
    callbacks=[StreamingStdOutCallbackHandler()]
)

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    return f"It's sunny in {city}."

@tool
def add(x: int, y: int) -> int:
    """add two numbers"""
    return x + y

tools = {'get_weather': get_weather, 'add': add}
model_with_tools = model.bind_tools([get_weather, add])


while True:
    get_message : str = input('Enter your query... \n')
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

        try:
            response = model_with_tools.invoke(message)
        except Exception as e:
            print(e)
            continue
        message.append(response)


    
    print(response.content)

print(message)