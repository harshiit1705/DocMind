from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

vector_store = Chroma(
    persist_directory="chroma-db",
    embedding_function= embedding_model
)

retriever = vector_store.as_retriever(
    search_type= "mmr",
    search_kwargs={
        "k":4,
        "fetch_k" : 10,
        "lambda_mult": 0.5
    }
)

llm = init_chat_model("groq:llama-3.1-8b-instant")

prompt_template = ChatPromptTemplate.from_messages(
    [
      ("system",
      """You are a helpful AI assistance.
      Use ONLY the provided context to answer the question.
      
      If the answer is not present in the context,
      say: "I couldn't find the answer in the document." 
      """),
      (
          "human",
          """Context:
          {context}
          
          Question:
          {question}
          """
      )
    ]
)

print("press 0 to exit.")

while True:
    query = input("You: ")

    if query == "0":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = prompt_template.invoke({
        "context":context,
        "question":query 
    })

    response = llm.invoke(prompt)

    print(f"\nAI: {response.content}")