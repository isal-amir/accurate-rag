from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

try:
    llm = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite', temperature=0, max_retries=0)
    print("Invoking model...")
    response = llm.invoke([HumanMessage(content='hello')])
    print("Response:", response.content)
except Exception as e:
    import traceback
    traceback.print_exc()
