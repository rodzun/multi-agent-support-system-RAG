import os
from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse

# Initialize Langfuse client
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

# Agents
from src.agents.orchestrator import classify_intent
from src.agents.hr_agent import hr_chain
from src.agents.tech_agent import tech_chain
from src.agents.finance_agent import finance_chain
from src.agents.evaluator import evaluator_chain

from langchain_core.runnables import RunnableBranch, RunnableLambda


# Router for LangChain
specialized_router = RunnableBranch(
    (lambda x: x["intent"] == "hr", hr_chain),
    (lambda x: x["intent"] == "tech", tech_chain),
    (lambda x: x["intent"] == "finance", finance_chain),
    RunnableLambda(
        lambda x: "I'm sorry, I cannot assist with that inquiry. It has been routed to the general support team."
    )
)


from src.agents.evaluator import evaluator_chain


def process_query(query: str):
    with langfuse.start_as_current_span(name="support-routing-root", input={"query": query}) as root:
        print(f"\nQuery: {query}")

        span_intent = langfuse.start_span(name="classify-intent", input={"query": query})
        intent = classify_intent(query)
        span_intent.update(output={"intent": intent})
        span_intent.end()

        print(f"Detected intent: {intent.upper()}")

        context = {"intent": intent, "question": query}

        span_router = langfuse.start_span(name="router-invoke", input=context)
        answer = specialized_router.invoke(context)
        span_router.update(output={"answer": answer})
        span_router.end()

        span_eval = langfuse.start_span(name="answer-evaluation", input={"query": query, "answer": answer})
        eval_result = evaluator_chain.invoke({"query": query, "answer": answer})
        # eval_result es Pydantic (tiene .score y .justification)
        span_eval.update(output=eval_result.dict())
        span_eval.end()

        langfuse.create_score(
            trace_id=root.trace_id,
            name="response_quality",
            value=eval_result.score,
            comment=eval_result.justification or ""
        )

        root.update(output={"intent": intent, "answer": answer, "score": eval_result.score})

        print(f"Answer: {answer}")
        print(f"Quality Score: {eval_result.score}/10")
        print(f"Trace: https://cloud.langfuse.com/public/traces/{root.trace_id}\n")

        return answer



if __name__ == "__main__":
    print("Multi-Agent Support Router with Langfuse 3.x Observability")
    print("Type 'quit' to exit\n")

    while True:
        query = input("User: ")
        if query.lower() in ["quit", "exit"]:
            break
        process_query(query)
