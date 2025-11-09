"""
LangChain v1.0 - LCEL (LangChain Expression Language)
Демонстрація нового підходу до композиції ланцюгів з використанням Runnable інтерфейсу
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
import os
from dotenv import load_dotenv

load_dotenv()


def demo_basic_lcel_chain():
    """
    Базова демонстрація LCEL - нововведення v1.0
    Використання оператора | для композиції компонентів
    """
    print("=== Базовий LCEL ланцюг ===\n")

    # Створення компонентів
    prompt = ChatPromptTemplate.from_template(
        "Розкажи цікавий факт про {topic} українською мовою в одному реченні."
    )
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    output_parser = StrOutputParser()

    # LCEL композиція з оператором |
    chain = prompt | model | output_parser

    # Виклик ланцюга
    result = chain.invoke({"topic": "Python"})
    print(f"Результат: {result}\n")

    return chain


def demo_parallel_chains():
    """
    Демонстрація паралельного виконання ланцюгів - нововведення v1.0
    """
    print("=== Паралельне виконання ланцюгів ===\n")

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # Створення різних промптів
    joke_prompt = ChatPromptTemplate.from_template("Розкажи жарт про {topic}")
    fact_prompt = ChatPromptTemplate.from_template("Розкажи факт про {topic}")
    poem_prompt = ChatPromptTemplate.from_template("Напиши коротку строфу про {topic}")

    # Паралельна композиція
    parallel_chain = RunnableParallel(
        joke=joke_prompt | model | StrOutputParser(),
        fact=fact_prompt | model | StrOutputParser(),
        poem=poem_prompt | model | StrOutputParser()
    )

    result = parallel_chain.invoke({"topic": "штучний інтелект"})

    print(f"Жарт: {result['joke']}\n")
    print(f"Факт: {result['fact']}\n")
    print(f"Вірш: {result['poem']}\n")

    return parallel_chain


def demo_chain_with_passthrough():
    """
    Демонстрація RunnablePassthrough для передачі контексту
    """
    print("=== Ланцюг з RunnablePassthrough ===\n")

    prompt = ChatPromptTemplate.from_template(
        "Питання: {question}\nКонтекст: {context}\n\nВідповідь:"
    )
    model = ChatOpenAI(model="gpt-3.5-turbo")

    # Використання RunnablePassthrough для зберігання оригінального вводу
    chain = (
        RunnableParallel({
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough()
        })
        | prompt
        | model
        | StrOutputParser()
    )

    result = chain.invoke({
        "question": "Що таке LangChain?",
        "context": "LangChain - це фреймворк для розробки застосунків з LLM"
    })

    print(f"Результат: {result}\n")

    return chain


def demo_streaming():
    """
    Демонстрація стрімінгу - покращено в v1.0
    """
    print("=== Стрімінг відповіді ===\n")

    prompt = ChatPromptTemplate.from_template("Напиши коротку історію про {topic}")
    model = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)

    chain = prompt | model | StrOutputParser()

    print("Історія генерується частинами:")
    for chunk in chain.stream({"topic": "роботи майбутнього"}):
        print(chunk, end="", flush=True)
    print("\n")


def demo_batch_processing():
    """
    Демонстрація пакетної обробки - оптимізовано в v1.0
    """
    print("=== Пакетна обробка ===\n")

    prompt = ChatPromptTemplate.from_template("Дай визначення: {term}")
    model = ChatOpenAI(model="gpt-3.5-turbo")
    chain = prompt | model | StrOutputParser()

    # Обробка кількох запитів одночасно
    terms = [
        {"term": "LangChain"},
        {"term": "LangGraph"},
        {"term": "LCEL"}
    ]

    results = chain.batch(terms)

    for term, result in zip(terms, results):
        print(f"{term['term']}: {result}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("LangChain v1.0 - LCEL Demonstration")
    print("=" * 60 + "\n")

    try:
        demo_basic_lcel_chain()
        demo_parallel_chains()
        demo_chain_with_passthrough()
        demo_streaming()
        demo_batch_processing()

        print("\n" + "=" * 60)
        print("Всі демонстрації успішно завершені!")
        print("=" * 60)

    except Exception as e:
        print(f"Помилка: {e}")
        print("Переконайтесь, що ви налаштували .env файл з OPENAI_API_KEY")
