"""
LangChain v1.0 - RAG (Retrieval-Augmented Generation)
Демонстрація побудови RAG систем з використанням нової LCEL архітектури
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from operator import itemgetter
from dotenv import load_dotenv

load_dotenv()


def create_sample_knowledge_base():
    """
    Створення простої бази знань для демонстрації
    """
    documents = [
        Document(
            page_content="LangChain v1.0 представляє LCEL - LangChain Expression Language, новий спосіб композиції ланцюгів.",
            metadata={"source": "langchain_docs", "topic": "LCEL"}
        ),
        Document(
            page_content="LangGraph - це бібліотека для побудови stateful агентів з можливістю циклів та умовної логіки.",
            metadata={"source": "langgraph_docs", "topic": "agents"}
        ),
        Document(
            page_content="Runnable інтерфейс в LangChain v1.0 дозволяє використовувати методи invoke, batch, stream та async варіанти.",
            metadata={"source": "langchain_docs", "topic": "runnables"}
        ),
        Document(
            page_content="В LangChain v1.0 покращена інтеграція з LangSmith для трейсингу та моніторингу ланцюгів.",
            metadata={"source": "langchain_docs", "topic": "observability"}
        ),
        Document(
            page_content="Structured Output в LangChain v1.0 дозволяє легко отримувати дані у форматі Pydantic моделей.",
            metadata={"source": "langchain_docs", "topic": "structured_output"}
        ),
        Document(
            page_content="LangGraph підтримує checkpointing для збереження стану агентів та можливість відновлення роботи.",
            metadata={"source": "langgraph_docs", "topic": "persistence"}
        ),
        Document(
            page_content="LCEL використовує оператор pipe (|) для з'єднання компонентів в єдиний ланцюг обробки.",
            metadata={"source": "langchain_docs", "topic": "LCEL"}
        ),
        Document(
            page_content="Human-in-the-loop pattern в LangGraph дозволяє зупиняти виконання агента для отримання людського вводу.",
            metadata={"source": "langgraph_docs", "topic": "human_in_loop"}
        ),
    ]

    return documents


def demo_basic_rag():
    """
    Базова демонстрація RAG з LCEL
    """
    print("=== Базовий RAG з LCEL ===\n")

    # Створення векторного сховища
    documents = create_sample_knowledge_base()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # RAG промпт
    template = """Відповідай на питання базуючись на наступному контексті:

Контекст: {context}

Питання: {question}

Відповідь:"""

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Функція для форматування документів
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL RAG ланцюг
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    # Тестове питання
    question = "Що таке LCEL?"
    answer = rag_chain.invoke(question)

    print(f"Питання: {question}")
    print(f"Відповідь: {answer}\n")

    return rag_chain


def demo_rag_with_sources():
    """
    RAG з поверненням джерел
    """
    print("=== RAG з джерелами ===\n")

    documents = create_sample_knowledge_base()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """Відповідай на питання базуючись на контексті. Будь стислим.

Контекст: {context}

Питання: {question}

Відповідь:"""

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Ланцюг який повертає і відповідь, і джерела
    rag_chain_with_source = RunnableParallel(
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
    ).assign(
        answer=(
            lambda x: format_docs(x["context"])
            | (lambda context: {"context": context, "question": x["question"]})
            | prompt
            | model
            | StrOutputParser()
        )
    )

    question = "Що таке LangGraph?"
    result = rag_chain_with_source.invoke(question)

    print(f"Питання: {question}")
    print(f"\nВідповідь: {result['answer']}\n")
    print("Джерела:")
    for i, doc in enumerate(result['context'], 1):
        print(f"  {i}. {doc.metadata['source']} (тема: {doc.metadata['topic']})")
        print(f"     {doc.page_content[:100]}...")
    print()

    return rag_chain_with_source


def demo_rag_with_filtering():
    """
    RAG з фільтрацією метаданих
    """
    print("=== RAG з фільтрацією метаданих ===\n")

    documents = create_sample_knowledge_base()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Ретрівер з фільтром по метаданим
    def get_filtered_retriever(topic_filter=None):
        if topic_filter:
            return vectorstore.as_retriever(
                search_kwargs={
                    "k": 2,
                    "filter": lambda metadata: metadata.get("topic") == topic_filter
                }
            )
        return vectorstore.as_retriever(search_kwargs={"k": 2})

    template = """Контекст: {context}

Питання: {question}

Відповідь (українською, стисло):"""

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Пошук тільки по темі "LCEL"
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    question = "Як працює оператор pipe?"
    answer = rag_chain.invoke(question)

    print(f"Питання: {question}")
    print(f"Відповідь: {answer}\n")


def demo_multi_query_rag():
    """
    RAG з множинними запитами для кращого пошуку
    """
    print("=== Multi-Query RAG ===\n")

    documents = create_sample_knowledge_base()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Генерація альтернативних запитів
    query_generation_template = """Ти помічник, який генерує альтернативні формулювання питання.

Оригінальне питання: {question}

Згенеруй 2 альтернативні формулювання цього питання, які допоможуть знайти релевантну інформацію.
Відповідай лише питаннями, по одному на рядок."""

    query_prompt = ChatPromptTemplate.from_template(query_generation_template)
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

    query_generation_chain = query_prompt | model | StrOutputParser()

    # Основний RAG промпт
    rag_template = """Контекст: {context}

Питання: {question}

Відповідь:"""

    rag_prompt = ChatPromptTemplate.from_template(rag_template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Основний ланцюг
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | model
        | StrOutputParser()
    )

    question = "Які нові можливості для агентів?"
    print(f"Оригінальне питання: {question}\n")

    # Генерація альтернативних запитів
    alternative_queries = query_generation_chain.invoke({"question": question})
    print(f"Альтернативні запити:\n{alternative_queries}\n")

    # Отримання відповіді
    answer = rag_chain.invoke(question)
    print(f"Відповідь: {answer}\n")


def demo_streaming_rag():
    """
    RAG зі стрімінгом відповіді
    """
    print("=== RAG зі стрімінгом ===\n")

    documents = create_sample_knowledge_base()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    template = """Контекст: {context}

Питання: {question}

Детальна відповідь:"""

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    question = "Розкажи про Runnable інтерфейс"
    print(f"Питання: {question}\n")
    print("Відповідь (streaming):\n")

    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)

    print("\n")


if __name__ == "__main__":
    print("=" * 60)
    print("LangChain v1.0 - RAG Demonstration")
    print("=" * 60 + "\n")

    try:
        demo_basic_rag()
        demo_rag_with_sources()
        demo_rag_with_filtering()
        demo_multi_query_rag()
        demo_streaming_rag()

        print("\n" + "=" * 60)
        print("Всі RAG демонстрації завершені!")
        print("=" * 60)

    except Exception as e:
        print(f"Помилка: {e}")
        print("Переконайтесь, що ви налаштували .env файл з OPENAI_API_KEY")
