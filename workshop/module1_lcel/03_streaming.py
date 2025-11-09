"""
MODULE 1: LCEL - Streaming
СЛАЙД 8: Streaming Responses

Цей приклад демонструє:
- Streaming відповідей в реальному часі
- Async streaming для production
- Порівняння UX: blocking vs streaming
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import asyncio
import time

load_dotenv()


def demo_streaming_vs_blocking():
    """Порівняння: блокуюча відповідь vs streaming"""
    print("=" * 60)
    print("🔄 BLOCKING VS STREAMING")
    print("=" * 60 + "\n")

    prompt = ChatPromptTemplate.from_template(
        "Напиши короткий параграф (5-7 речень) про {topic}"
    )
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    chain = prompt | model | StrOutputParser()

    topic = "переваги штучного інтелекту в медицині"

    # BLOCKING - чекаємо всю відповідь
    print("⏳ BLOCKING MODE:")
    print(f"Генеруємо текст про: {topic}")
    print("Чекаємо...\n")

    start = time.time()
    result = chain.invoke({"topic": topic})
    blocking_time = time.time() - start

    print(f"✅ Готово за {blocking_time:.2f}s")
    print(f"📝 {result}\n")

    # STREAMING - бачимо по мірі генерації
    print("⚡ STREAMING MODE:")
    print(f"Генеруємо текст про: {topic}\n")

    start = time.time()
    print("📝 ", end="", flush=True)

    for chunk in chain.stream({"topic": topic}):
        print(chunk, end="", flush=True)
        time.sleep(0.02)  # Симуляція читання користувачем

    streaming_time = time.time() - start

    print(f"\n\n✅ Завершено за {streaming_time:.2f}s")

    print("\n" + "=" * 60)
    print("📊 ПОРІВНЯННЯ:")
    print(f"  Blocking: {blocking_time:.2f}s (користувач чекає)")
    print(f"  Streaming: {streaming_time:.2f}s (користувач читає)")
    print(f"  💡 UX: Streaming відчувається швидше!")
    print("=" * 60 + "\n")


async def demo_async_streaming():
    """Async streaming для production"""
    print("=" * 60)
    print("🚀 ASYNC STREAMING (Production)")
    print("=" * 60 + "\n")

    prompt = ChatPromptTemplate.from_template(
        "Створи список з 5 порад для {topic}. Форматуй з номерами."
    )
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    chain = prompt | model | StrOutputParser()

    print("📋 Генеруємо поради про Python best practices\n")

    # Async streaming
    async for chunk in chain.astream({"topic": "Python best practices"}):
        print(chunk, end="", flush=True)
        await asyncio.sleep(0.02)

    print("\n")


def demo_multiple_concurrent_streams():
    """Кілька streaming операцій одночасно"""
    print("=" * 60)
    print("🎭 МНОЖИННИЙ CONCURRENT STREAMING")
    print("=" * 60 + "\n")

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    chains = {
        "Python": ChatPromptTemplate.from_template(
            "Одне речення про {lang}"
        ) | model | StrOutputParser(),

        "JavaScript": ChatPromptTemplate.from_template(
            "Одне речення про {lang}"
        ) | model | StrOutputParser(),

        "Rust": ChatPromptTemplate.from_template(
            "Одне речення про {lang}"
        ) | model | StrOutputParser(),
    }

    print("🔄 Streaming 3 responses паралельно:\n")

    for lang, chain in chains.items():
        print(f"\n💬 {lang}: ", end="", flush=True)
        for chunk in chain.stream({"lang": lang}):
            print(chunk, end="", flush=True)

    print("\n")


def demo_streaming_with_callbacks():
    """Streaming з callbacks для моніторингу"""
    print("=" * 60)
    print("📊 STREAMING З CALLBACKS")
    print("=" * 60 + "\n")

    from langchain_core.callbacks import StreamingStdOutCallbackHandler

    # Custom callback для tracking
    class TokenCounterCallback(StreamingStdOutCallbackHandler):
        def __init__(self):
            super().__init__()
            self.token_count = 0

        def on_llm_new_token(self, token: str, **kwargs):
            self.token_count += 1
            # Не друкуємо, тільки рахуємо

    callback = TokenCounterCallback()

    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        streaming=True,
        callbacks=[callback]
    )

    prompt = ChatPromptTemplate.from_template(
        "Напиши 3 речення про {topic}"
    )
    chain = prompt | model | StrOutputParser()

    print("🔢 Рахуємо tokens під час streaming...\n")
    print("📝 ", end="", flush=True)

    result = ""
    for chunk in chain.stream({"topic": "квантові комп'ютери"}):
        result += chunk
        print(chunk, end="", flush=True)

    print(f"\n\n✅ Tokens згенеровано: ~{callback.token_count}")
    print(f"📏 Довжина тексту: {len(result)} символів\n")


def demo_streaming_to_file():
    """Streaming безпосередньо у файл (logs, reports)"""
    print("=" * 60)
    print("💾 STREAMING ДО ФАЙЛУ")
    print("=" * 60 + "\n")

    prompt = ChatPromptTemplate.from_template(
        "Створи технічний звіт про {topic}. Включи introduction, 3 key points, conclusion."
    )
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    chain = prompt | model | StrOutputParser()

    output_file = "/tmp/streaming_report.txt"

    print(f"📄 Генеруємо звіт та зберігаємо в {output_file}...\n")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== ТЕХНІЧНИЙ ЗВІТ ===\n\n")

        for chunk in chain.stream({"topic": "Kubernetes adoption in enterprise"}):
            f.write(chunk)
            print(chunk, end="", flush=True)

        f.write("\n\n=== КІНЕЦЬ ЗВІТУ ===\n")

    print(f"\n\n✅ Звіт збережено в {output_file}")
    print(f"📊 Розмір файлу: {open(output_file).read().__sizeof__()} bytes\n")


# ============================================================================
# ІНТЕРАКТИВНА ВПРАВА
# ============================================================================

def workshop_exercise():
    """
    ВПРАВА: Створіть streaming чат-бот
    """
    print("=" * 60)
    print("🎯 ВПРАВА: Streaming Chatbot")
    print("=" * 60 + "\n")

    print("Завдання:")
    print("Створіть chatbot який:")
    print("  1. Приймає повідомлення від користувача")
    print("  2. Streamує відповідь в реальному часі")
    print("  3. Показує індикатор 'typing...' перед streaming")
    print("  4. Рахує скільки символів згенеровано\n")

    input("⏸️  Натисніть Enter щоб побачити рішення...")
    show_solution()


def show_solution():
    """Рішення вправи"""
    print("\n" + "=" * 60)
    print("✅ РІШЕННЯ: Streaming Chatbot")
    print("=" * 60 + "\n")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Ти дружній AI асистент. Відповідай корисно та ввічливо."),
        ("user", "{message}")
    ])
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    chain = prompt | model | StrOutputParser()

    def chat(user_message: str):
        """Chatbot function з streaming"""
        print(f"\n👤 Ви: {user_message}")
        print("🤖 AI: ", end="", flush=True)
        print("typing...", end="\r", flush=True)  # Typing indicator

        char_count = 0
        print("🤖 AI: ", end="", flush=True)  # Clear typing indicator

        for chunk in chain.stream({"message": user_message}):
            print(chunk, end="", flush=True)
            char_count += len(chunk)

        print(f"\n   (згенеровано {char_count} символів)")

    # Test chatbot
    chat("Привіт! Як ти працюєш?")
    chat("Розкажи цікавий факт про космос")

    print("\n💡 Цей підхід можна використати для:")
    print("  - Web чатів (через WebSocket)")
    print("  - CLI інтерфейсів")
    print("  - API endpoints з SSE (Server-Sent Events)")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("🎓 MODULE 1: LCEL - Streaming")
    print("=" * 60 + "\n")

    try:
        # Demo 1: Blocking vs Streaming
        demo_streaming_vs_blocking()
        input("⏸️  Натисніть Enter для продовження...")

        # Demo 2: Async streaming
        asyncio.run(demo_async_streaming())
        input("⏸️  Натисніть Enter для продовження...")

        # Demo 3: Multiple concurrent streams
        demo_multiple_concurrent_streams()
        input("⏸️  Натисніть Enter для продовження...")

        # Demo 4: Callbacks
        demo_streaming_with_callbacks()
        input("⏸️  Натисніть Enter для продовження...")

        # Demo 5: File streaming
        demo_streaming_to_file()
        input("⏸️  Натисніть Enter для вправи...")

        # Workshop exercise
        workshop_exercise()

        print("\n" + "=" * 60)
        print("✅ STREAMING MODULE ЗАВЕРШЕНО!")
        print("=" * 60)
        print("\n📝 Key Takeaways:")
        print("  1. .stream() для sync, .astream() для async")
        print("  2. Streaming покращує UX - користувач бачить прогрес")
        print("  3. Callbacks для моніторингу та metrics")
        print("  4. Можна streamити у файл, WebSocket, SSE")

    except Exception as e:
        print(f"\n❌ Помилка: {e}")
