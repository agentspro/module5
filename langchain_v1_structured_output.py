"""
LangChain v1.0 - Structured Output
Демонстрація роботи зі структурованими даними та Pydantic моделями
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

load_dotenv()


# Визначення Pydantic моделей для структурованого виводу
class Person(BaseModel):
    """Інформація про персону"""
    name: str = Field(description="Ім'я персони")
    age: int = Field(description="Вік персони")
    occupation: str = Field(description="Професія")
    skills: List[str] = Field(description="Список навичок")


class Book(BaseModel):
    """Інформація про книгу"""
    title: str = Field(description="Назва книги")
    author: str = Field(description="Автор книги")
    year: int = Field(description="Рік публікації")
    genre: str = Field(description="Жанр")
    summary: str = Field(description="Короткий опис")
    rating: float = Field(description="Рейтинг від 1 до 10", ge=1, le=10)


class Analysis(BaseModel):
    """Аналіз тексту"""
    sentiment: str = Field(description="Емоційне забарвлення: позитивне, негативне, нейтральне")
    key_topics: List[str] = Field(description="Ключові теми")
    summary: str = Field(description="Короткий підсумок")
    confidence: float = Field(description="Рівень впевненості від 0 до 1", ge=0, le=1)


def demo_structured_person_extraction():
    """
    Демонстрація витягу структурованої інформації про персону
    """
    print("=== Структурований вивід: Персона ===\n")

    # Створення парсера
    parser = PydanticOutputParser(pydantic_object=Person)

    # Промпт з інструкціями форматування
    prompt = ChatPromptTemplate.from_template(
        "Витягни інформацію про персону з наступного тексту.\n"
        "{format_instructions}\n"
        "Текст: {text}\n"
    )

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # LCEL ланцюг
    chain = prompt | model | parser

    # Тестовий текст
    text = """
    Іван Петренко - 32-річний розробник програмного забезпечення з Києва.
    Він спеціалізується на Python, має досвід роботи з FastAPI, Django та машинним навчанням.
    """

    result = chain.invoke({
        "text": text,
        "format_instructions": parser.get_format_instructions()
    })

    print(f"Результат (Pydantic об'єкт):")
    print(f"  Ім'я: {result.name}")
    print(f"  Вік: {result.age}")
    print(f"  Професія: {result.occupation}")
    print(f"  Навички: {', '.join(result.skills)}")
    print(f"\nJSON представлення:")
    print(result.model_dump_json(indent=2))
    print()

    return result


def demo_structured_book_info():
    """
    Демонстрація генерації структурованої інформації про книгу
    """
    print("=== Структурований вивід: Книга ===\n")

    parser = PydanticOutputParser(pydantic_object=Book)

    prompt = ChatPromptTemplate.from_template(
        "Створи інформацію про книгу на тему: {topic}\n"
        "{format_instructions}\n"
    )

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    chain = prompt | model | parser

    result = chain.invoke({
        "topic": "штучний інтелект у майбутньому",
        "format_instructions": parser.get_format_instructions()
    })

    print(f"Назва: {result.title}")
    print(f"Автор: {result.author}")
    print(f"Рік: {result.year}")
    print(f"Жанр: {result.genre}")
    print(f"Рейтинг: {result.rating}/10")
    print(f"Опис: {result.summary}")
    print()

    return result


def demo_text_analysis():
    """
    Демонстрація аналізу тексту з структурованим виводом
    """
    print("=== Структурований аналіз тексту ===\n")

    parser = PydanticOutputParser(pydantic_object=Analysis)

    prompt = ChatPromptTemplate.from_template(
        "Проаналізуй наступний текст:\n"
        "{text}\n\n"
        "{format_instructions}\n"
    )

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = prompt | model | parser

    text = """
    LangChain v1.0 приніс революційні зміни в світ розробки з LLM.
    Нова архітектура LCEL дозволяє легко створювати складні ланцюги обробки,
    а підтримка структурованого виводу значно спрощує інтеграцію з бізнес-логікою.
    Розробники в захваті від нових можливостей!
    """

    result = chain.invoke({
        "text": text,
        "format_instructions": parser.get_format_instructions()
    })

    print(f"Емоційне забарвлення: {result.sentiment}")
    print(f"Ключові теми: {', '.join(result.key_topics)}")
    print(f"Підсумок: {result.summary}")
    print(f"Впевненість: {result.confidence * 100}%")
    print()

    return result


def demo_batch_structured_output():
    """
    Демонстрація пакетної обробки зі структурованим виводом
    """
    print("=== Пакетна обробка зі структурованим виводом ===\n")

    parser = PydanticOutputParser(pydantic_object=Person)

    prompt = ChatPromptTemplate.from_template(
        "Створи випадкову персону з професією: {occupation}\n"
        "{format_instructions}\n"
    )

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)
    chain = prompt | model | parser

    occupations = [
        {"occupation": "Data Scientist", "format_instructions": parser.get_format_instructions()},
        {"occupation": "UX Designer", "format_instructions": parser.get_format_instructions()},
        {"occupation": "DevOps Engineer", "format_instructions": parser.get_format_instructions()},
    ]

    results = chain.batch(occupations)

    for i, person in enumerate(results, 1):
        print(f"Персона {i}:")
        print(f"  {person.name}, {person.age} років")
        print(f"  {person.occupation}")
        print(f"  Навички: {', '.join(person.skills)}")
        print()

    return results


def demo_with_function_calling():
    """
    Демонстрація використання function calling для структурованого виводу (v1.0 feature)
    """
    print("=== Function Calling для структурованого виводу ===\n")

    # В LangChain v1.0 можна використовувати with_structured_output()
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Прив'язка схеми до моделі
    structured_llm = model.with_structured_output(Person)

    prompt = ChatPromptTemplate.from_template(
        "Витягни інформацію про персону: {text}"
    )

    chain = prompt | structured_llm

    text = "Марія Коваленко, 28 років, UX дизайнер зі знанням Figma, Adobe XD та user research"

    result = chain.invoke({"text": text})

    print(f"Витягнута інформація:")
    print(f"  Ім'я: {result.name}")
    print(f"  Вік: {result.age}")
    print(f"  Професія: {result.occupation}")
    print(f"  Навички: {', '.join(result.skills)}")
    print()

    return result


if __name__ == "__main__":
    print("=" * 60)
    print("LangChain v1.0 - Structured Output Demonstration")
    print("=" * 60 + "\n")

    try:
        demo_structured_person_extraction()
        demo_structured_book_info()
        demo_text_analysis()
        demo_batch_structured_output()
        demo_with_function_calling()

        print("\n" + "=" * 60)
        print("Всі демонстрації структурованого виводу завершені!")
        print("=" * 60)

    except Exception as e:
        print(f"Помилка: {e}")
        print("Переконайтесь, що ви налаштували .env файл з OPENAI_API_KEY")
