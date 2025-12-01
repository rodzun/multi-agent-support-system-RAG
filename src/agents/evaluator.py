from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import os


class Evaluation(BaseModel):
    score: int = Field(description="Quality score from 1 to 10")
    justification: str = Field(description="Brief explanation")


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a strict quality evaluator for customer support responses.
Score from 1-10 considering:
- Relevance to the question
- Completeness
- Accuracy and policy compliance
- Clarity and professionalism

Return a JSON object with fields:
- score
- justification
"""),
    ("human", "Query: {query}\n\nAnswer: {answer}")
])

evaluator_chain = prompt | llm.with_structured_output(Evaluation)
