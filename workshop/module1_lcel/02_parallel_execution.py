"""
MODULE 1: LCEL - Паралельне виконання
СЛАЙД 7: Parallel Execution

Цей приклад демонструє:
- RunnableParallel для одночасного виконання
- Порівняння швидкості: послідовно vs паралельно
- Практичне застосування
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv
import time

load_dotenv()


def demo_sequential_vs_parallel():
    """Порівняння швидкості: послідовно vs паралельно"""
    print("=" * 60)
    print("⚡ ПОСЛІДОВНО VS ПАРАЛЕЛЬНО")
    print("=" * 60 + "\n")

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # Три різні ланцюги
    summary_chain = (
        ChatPromptTemplate.from_template("Напиши короткий summary (2-3 речення) для: {text}")
        | model | StrOutputParser()
    )

    sentiment_chain = (
        ChatPromptTemplate.from_template("Визнач sentiment (позитивний/негативний/нейтральний): {text}")
        | model | StrOutputParser()
    )

    keywords_chain = (
        ChatPromptTemplate.from_template("Витягни 3-5 ключових слів з: {text}")
        | model | StrOutputParser()
    )

    text = """
    LangChain v1.0 представляє революційні зміни в розробці AI застосунків.
    Нова архітектура LCEL робить код простішим та зрозумілішим, а LangGraph
    дозволяє будувати складних stateful агентів. Розробники в захваті!
    """

    # ПОСЛІДОВНЕ виконання
    print("🐌 ПОСЛІДОВНЕ ВИКОНАННЯ:")
    start = time.time()

    summary = summary_chain.invoke({"text": text})
    sentiment = sentiment_chain.invoke({"text": text})
    keywords = keywords_chain.invoke({"text": text})

    sequential_time = time.time() - start

    print(f"⏱️  Час: {sequential_time:.2f}s")
    print(f"📝 Summary: {summary}")
    print(f"😊 Sentiment: {sentiment}")
    print(f"🔑 Keywords: {keywords}\n")

    # ПАРАЛЕЛЬНЕ виконання
    print("🚀 ПАРАЛЕЛЬНЕ ВИКОНАННЯ:")
    start = time.time()

    parallel_chain = RunnableParallel(
        summary=summary_chain,
        sentiment=sentiment_chain,
        keywords=keywords_chain
    )

    results = parallel_chain.invoke({"text": text})
    parallel_time = time.time() - start

    print(f"⏱️  Час: {parallel_time:.2f}s")
    print(f"📝 Summary: {results['summary']}")
    print(f"😊 Sentiment: {results['sentiment']}")
    print(f"🔑 Keywords: {results['keywords']}\n")

    # Порівняння
    speedup = sequential_time / parallel_time
    print("=" * 60)
    print(f"📊 РЕЗУЛЬТАТИ:")
    print(f"  Послідовно: {sequential_time:.2f}s")
    print(f"  Паралельно: {parallel_time:.2f}s")
    print(f"  Прискорення: {speedup:.2f}x")
    print(f"  Економія: {sequential_time - parallel_time:.2f}s")
    print("=" * 60 + "\n")


def demo_document_analysis():
    """Практичний кейс: аналіз документа з різних perspectives"""
    print("=" * 60)
    print("📄 АНАЛІЗ ДОКУМЕНТА")
    print("=" * 60 + "\n")

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    document = """
    Наш стартап розробив нову платформу для онлайн навчання.
    За перший місяць ми залучили 1000 користувачів і отримали
    overwhelmingly positive feedback. Однак, технічні виклики
    з масштабуванням бази даних створюють певні проблеми.
    Інвестори зацікавлені, але чекають на proof of market fit.
    """

    # Паралельний аналіз з різних перспектив
    analysis_chain = RunnableParallel(
        business=ChatPromptTemplate.from_template(
            "Бізнес аналіз (тільки факти): {doc}"
        ) | model | StrOutputParser(),

        technical=ChatPromptTemplate.from_template(
            "Технічні виклики (що треба вирішити): {doc}"
        ) | model | StrOutputParser(),

        risks=ChatPromptTemplate.from_template(
            "Потенційні ризики (3 найбільших): {doc}"
        ) | model | StrOutputParser(),

        opportunities=ChatPromptTemplate.from_template(
            "Можливості для зростання (3 найкращих): {doc}"
        ) | model | StrOutputParser()
    )

    print("🔄 Аналізуємо документ з 4 perspectives...")
    start = time.time()

    results = analysis_chain.invoke({"doc": document})

    print(f"✅ Готово за {time.time() - start:.2f}s\n")

    print("💼 БІЗНЕС АНАЛІЗ:")
    print(results['business'] + "\n")

    print("⚙️  ТЕХНІЧНІ ВИКЛИКИ:")
    print(results['technical'] + "\n")

    print("⚠️  РИЗИКИ:")
    print(results['risks'] + "\n")

    print("🎯 МОЖЛИВОСТІ:")
    print(results['opportunities'] + "\n")


def demo_nested_parallel():
    """Вкладений паралелізм: паралельні ланцюги всередині паралельних"""
    print("=" * 60)
    print("🎭 ВКЛАДЕНИЙ ПАРАЛЕЛІЗМ")
    print("=" * 60 + "\n")

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # Внутрішні паралельні ланцюги
    pros_cons = RunnableParallel(
        pros=ChatPromptTemplate.from_template("3 переваги {topic}") | model | StrOutputParser(),
        cons=ChatPromptTemplate.from_template("3 недоліки {topic}") | model | StrOutputParser()
    )

    alternatives = RunnableParallel(
        similar=ChatPromptTemplate.from_template("3 альтернативи {topic}") | model | StrOutputParser(),
        comparison=ChatPromptTemplate.from_template("Порівняй {topic} з конкурентами") | model | StrOutputParser()
    )

    # Зовнішній паралельний ланцюг
    full_analysis = RunnableParallel(
        analysis=pros_cons,
        market=alternatives
    )

    print("🔍 Комплексний аналіз...")
    results = full_analysis.invoke({"topic": "Docker"})

    print("\n📊 АНАЛІЗ:")
    print(f"✅ Переваги: {results['analysis']['pros']}")
    print(f"❌ Недоліки: {results['analysis']['cons']}\n")

    print("🔄 РИНОК:")
    print(f"🔹 Альтернативи: {results['market']['similar']}")
    print(f"🔹 Порівняння: {results['market']['comparison']}\n")


# ============================================================================
# ІНТЕРАКТИВНА ВПРАВА
# ============================================================================

def workshop_exercise():
    """
    ВПРАВА: Створіть систему аналізу код-ревью

    Паралельно перевіряйте:
    1. Code quality (чистота коду)
    2. Security issues (безпека)
    3. Performance concerns (продуктивність)
    4. Best practices (best practices)
    """
    print("=" * 60)
    print("🎯 ВПРАВА: Система код-ревью")
    print("=" * 60 + "\n")

    print("Завдання:")
    print("Створіть паралельний ланцюг який аналізує код за 4 критеріями\n")

    code_sample = """
def process_users(users):
    result = []
    for user in users:
        if user['age'] > 18:
            result.append(user['name'])
    return result
"""

    print("Код для аналізу:")
    print(code_sample)
    print("\nСтворіть RunnableParallel з 4 ланцюгами:")
    print("  - code_quality: оцінка чистоти")
    print("  - security: перевірка безпеки")
    print("  - performance: оптимізація")
    print("  - best_practices: рекомендації")

    input("\n⏸️  Натисніть Enter щоб побачити рішення...")
    show_solution(code_sample)


def show_solution(code):
    """Рішення вправи"""
    print("\n" + "=" * 60)
    print("✅ РІШЕННЯ")
    print("=" * 60 + "\n")

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    code_review = RunnableParallel(
        code_quality=(
            ChatPromptTemplate.from_template(
                "Code quality review (1-2 речення): {code}"
            ) | model | StrOutputParser()
        ),

        security=(
            ChatPromptTemplate.from_template(
                "Security check (потенційні проблеми): {code}"
            ) | model | StrOutputParser()
        ),

        performance=(
            ChatPromptTemplate.from_template(
                "Performance analysis (оптимізації): {code}"
            ) | model | StrOutputParser()
        ),

        best_practices=(
            ChatPromptTemplate.from_template(
                "Python best practices (рекомендації): {code}"
            ) | model | StrOutputParser()
        )
    )

    results = code_review.invoke({"code": code})

    print("📋 CODE REVIEW RESULTS:\n")
    print(f"✨ Code Quality:\n{results['code_quality']}\n")
    print(f"🔒 Security:\n{results['security']}\n")
    print(f"⚡ Performance:\n{results['performance']}\n")
    print(f"📚 Best Practices:\n{results['best_practices']}\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("🎓 MODULE 1: LCEL - Паралельне виконання")
    print("=" * 60 + "\n")

    try:
        # Demo 1: Порівняння швидкості
        demo_sequential_vs_parallel()
        input("⏸️  Натисніть Enter для продовження...")

        # Demo 2: Аналіз документа
        demo_document_analysis()
        input("⏸️  Натисніть Enter для продовження...")

        # Demo 3: Вкладений паралелізм
        demo_nested_parallel()
        input("⏸️  Натисніть Enter для вправи...")

        # Workshop exercise
        workshop_exercise()

        print("\n" + "=" * 60)
        print("✅ PARALLEL EXECUTION ЗАВЕРШЕНО!")
        print("=" * 60)
        print("\n📝 Key Takeaways:")
        print("  1. RunnableParallel запускає ланцюги одночасно")
        print("  2. Прискорення до 3-4x для незалежних операцій")
        print("  3. Результат повертається як dict")
        print("  4. Можна вкладати паралелізм для складних workflow")

    except Exception as e:
        print(f"\n❌ Помилка: {e}")
