from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
    temperature=0
)

class IntentClassification(BaseModel):
    intent: str = Field(description="One of: hr, tech, finance, unknown")

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert router for customer support tickets.
Classify the user query into exactly one department:
- hr: human resources, benefits, vacation, payroll questions, personal info
- tech: IT support, software, hardware, access, passwords, VPN
- finance: expenses, reimbursements, bonuses, invoices, payments
- unknown: anything else

Return only the classification."""),
    ("human", "{query}")
])

intent_chain = prompt | llm.with_structured_output(IntentClassification)

def classify_intent(query: str) -> str:
    result = intent_chain.invoke({"query": query})
    return result.intent