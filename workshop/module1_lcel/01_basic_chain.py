"""
MODULE 1: LCEL - Основи
СЛАЙД 6: Базовий приклад LCEL

Цей приклад демонструє:
- Створення простого ланцюга з LCEL
- Використання pipe оператора |
- Базові компоненти: Prompt, Model, Parser
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# Перевірка API ключа
if not os.getenv("OPENAI_API_KEY"):
    print("❌ ПОМИЛКА: OPENAI_API_KEY не знайдено!")
    print("Створіть .env файл і додайте: OPENAI_API_KEY=your_key_here")
    exit(1)


def demo_basic_chain():
    """Найпростіший LCEL ланцюг"""
    print("=" * 60)
    print("🔗 БАЗОВИЙ LCEL ЛАНЦЮГ")
    print("=" * 60 + "\n")

    # 1. Створюємо промпт
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Ти експерт в {domain}. Відповідай коротко та зрозуміло."),
        ("user", "{question}")
    ])

    # 2. Створюємо модель
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7
    )

    # 3. Створюємо парсер
    output_parser = StrOutputParser()

    # 4. Компонуємо через pipe оператор |
    chain = prompt | model | output_parser

    print("✅ Ланцюг створено: prompt | model | parser\n")

    # 5. Викликаємо ланцюг
    result = chain.invoke({
        "domain": "Python",
        "question": "Що таке декоратори?"
    })

    print(f"💬 Питання: Що таке декоратори?")
    print(f"🤖 Відповідь: {result}\n")

    return chain


def demo_different_inputs():
    """Той самий ланцюг з різними входами"""
    print("=" * 60)
    print("🔄 РІЗНІ ВХОДИ - ТОЙ САМИЙ ЛАНЦЮГ")
    print("=" * 60 + "\n")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Ти експерт в {domain}. Відповідай коротко та зрозуміло."),
        ("user", "{question}")
    ])
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    chain = prompt | model | StrOutputParser()

    # Різні запити
    queries = [
        {"domain": "JavaScript", "question": "Що таке closure?"},
        {"domain": "DevOps", "question": "Чому використовувати Docker?"},
        {"domain": "ML", "question": "Різниця між supervised та unsupervised?"},
    ]

    for i, query in enumerate(queries, 1):
        print(f"📌 Запит {i}: {query['domain']} - {query['question']}")
        result = chain.invoke(query)
        print(f"🤖 Відповідь: {result}\n")


def demo_chain_inspection():
    """Дебаг: що відбувається всередині ланцюга"""
    print("=" * 60)
    print("🔍 ІНСПЕКЦІЯ ЛАНЦЮГА")
    print("=" * 60 + "\n")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Ти експерт в {domain}."),
        ("user", "{question}")
    ])
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Без парсера - бачимо сирий output
    chain_raw = prompt | model

    print("🔹 Без парсера (AIMessage об'єкт):")
    result_raw = chain_raw.invoke({
        "domain": "Python",
        "question": "Hello"
    })
    print(f"Type: {type(result_raw)}")
    print(f"Content: {result_raw.content}")
    print(f"Metadata: {result_raw.response_metadata}\n")

    # З парсером - чистий string
    chain_parsed = prompt | model | StrOutputParser()

    print("🔹 З парсером (string):")
    result_parsed = chain_parsed.invoke({
        "domain": "Python",
        "question": "Hello"
    })
    print(f"Type: {type(result_parsed)}")
    print(f"Content: {result_parsed}\n")


# ============================================================================
# ІНТЕРАКТИВНА ЧАСТИНА ДЛЯ ВОРКШОПУ
# ============================================================================

def workshop_exercise():
    """
    ВПРАВА ДЛЯ УЧАСНИКІВ:
    Створіть ланцюг який:
    1. Приймає назву технології
    2. Генерує 3 переваги та 3 недоліки
    3. Форматує у списки
    """
    print("=" * 60)
    print("🎯 ВПРАВА: Створіть свій ланцюг")
    print("=" * 60 + "\n")

    print("Завдання:")
    print("1. Створіть промпт який приймає {technology}")
    print("2. Попросіть LLM згенерувати переваги та недоліки")
    print("3. Використайте StrOutputParser")
    print("4. Викличте з technology='Docker'\n")

    print("Шаблон:")
    print("""
prompt = ChatPromptTemplate.from_messages([
    ("system", "Ти технічний експерт."),
    ("user", "Назви 3 переваги та 3 недоліки {technology}")
])
# Ваш код тут...
""")

    # Розкоментуйте коли учасники готові побачити рішення
    input("\n⏸️  Натисніть Enter щоб побачити рішення...")
    show_solution()


def show_solution():
    """Рішення вправи"""
    print("\n" + "=" * 60)
    print("✅ РІШЕННЯ")
    print("=" * 60 + "\n")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Ти технічний експерт. Відповідай структуровано."),
        ("user", "Назви 3 переваги та 3 недоліки {technology}. Форматуй як:\nПереваги:\n- ...\nНедоліки:\n- ...")
    ])
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    chain = prompt | model | StrOutputParser()

    result = chain.invoke({"technology": "Docker"})
    print("🐳 Docker:")
    print(result)


# ============================================================================
# MAIN - Запуск демо
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("🎓 MODULE 1: LCEL - Базові ланцюги")
    print("=" * 60 + "\n")

    try:
        # Demo 1: Базовий ланцюг
        demo_basic_chain()
        input("⏸️  Натисніть Enter для продовження...")

        # Demo 2: Різні входи
        demo_different_inputs()
        input("⏸️  Натисніть Enter для продовження...")

        # Demo 3: Інспекція
        demo_chain_inspection()
        input("⏸️  Натисніть Enter для вправи...")

        # Workshop exercise
        workshop_exercise()

        print("\n" + "=" * 60)
        print("✅ MODULE 1 ЗАВЕРШЕНО!")
        print("=" * 60)
        print("\n📝 Key Takeaways:")
        print("  1. LCEL використовує pipe оператор |")
        print("  2. Три компоненти: Prompt, Model, Parser")
        print("  3. Runnable інтерфейс: .invoke() для виклику")
        print("  4. Той самий ланцюг працює з різними входами")

    except Exception as e:
        print(f"\n❌ Помилка: {e}")
        print("\nПеревірте:")
        print("  1. .env файл існує")
        print("  2. OPENAI_API_KEY правильний")
        print("  3. Інтернет з'єднання")
