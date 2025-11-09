"""
ПОРІВНЯННЯ: LangChain v0.x → v1.0
Показує що змінилось і чому це важливо

ПРОБЛЕМА: В v0.x було складно композувати ланцюги, багато boilerplate коду
РІШЕННЯ: v1.0 вводить LCEL - простий і зрозумілий спосіб композиції
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv

load_dotenv()


print("=" * 80)
print("МІГРАЦІЯ: Побудова ланцюгів LangChain v0.x → v1.0")
print("=" * 80 + "\n")


# ============================================================================
# ПРИКЛАД 1: Простий ланцюг
# ============================================================================

print("\n" + "=" * 80)
print("1. ПРОСТИЙ ЛАНЦЮГ: Prompt → Model → Output")
print("=" * 80 + "\n")

print("❌ СТАРИЙ СПОСІБ (v0.x) - Verbose, незрозуміло")
print("-" * 80)
print("""
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Багато boilerplate коду
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Розкажи про {topic}"
)
llm = OpenAI(temperature=0.7)
chain = LLMChain(llm=llm, prompt=prompt)

# Викликається через .run() або .predict()
result = chain.run(topic="Python")  # Deprecated!
# АБО
result = chain.predict(topic="Python")  # Не інтуїтивно!
""")

print("\n✅ НОВИЙ СПОСІБ (v1.0) - LCEL з оператором |")
print("-" * 80)
print("""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Просто і зрозуміло - як Unix pipes!
prompt = ChatPromptTemplate.from_template("Розкажи про {topic}")
model = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

# Єдиний інтерфейс для всього
result = chain.invoke({"topic": "Python"})
""")

print("\n🎯 ЩО ПОКРАЩИЛОСЬ:")
print("  • Оператор | робить композицію інтуїтивною (як bash pipes)")
print("  • Єдиний метод .invoke() замість .run(), .predict(), __call__()")
print("  • Менше коду, простіше читати")
print("  • Runnable інтерфейс: invoke, stream, batch - все працює однаково\n")

# Демонстрація нового підходу
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
prompt = ChatPromptTemplate.from_template("Одне речення про {topic}")
chain = prompt | model | StrOutputParser()

result = chain.invoke({"topic": "переваги LCEL"})
print(f"📝 Результат: {result}\n")


# ============================================================================
# ПРИКЛАД 2: Послідовні ланцюги
# ============================================================================

print("\n" + "=" * 80)
print("2. ПОСЛІДОВНІ ЛАНЦЮГИ: Один ланцюг → інший ланцюг")
print("=" * 80 + "\n")

print("❌ СТАРИЙ СПОСІБ (v0.x) - Потрібні спеціальні класи")
print("-" * 80)
print("""
from langchain.chains import SimpleSequentialChain, LLMChain

# Створюємо окремі ланцюги
chain1 = LLMChain(llm=llm, prompt=prompt1)
chain2 = LLMChain(llm=llm, prompt=prompt2)

# Використовуємо спеціальний клас для об'єднання
overall_chain = SimpleSequentialChain(
    chains=[chain1, chain2],
    verbose=True
)

result = overall_chain.run(input_text)
""")

print("\n✅ НОВИЙ СПОСІБ (v1.0) - Просто додаємо | між компонентами")
print("-" * 80)
print("""
# Просто з'єднуємо pipe оператором
prompt1 = ChatPromptTemplate.from_template("Генеруй ідею для: {topic}")
prompt2 = ChatPromptTemplate.from_template("Покращи цю ідею: {idea}")

chain = (
    prompt1
    | model
    | StrOutputParser()
    | (lambda idea: {"idea": idea})  # Перетворюємо для наступного промпта
    | prompt2
    | model
    | StrOutputParser()
)

result = chain.invoke({"topic": "стартап"})
""")

print("\n🎯 ЩО ПОКРАЩИЛОСЬ:")
print("  • Не потрібні спеціальні класи (SimpleSequentialChain, SequentialChain)")
print("  • Візуально видно потік даних зверху вниз")
print("  • Легко додавати/видаляти кроки")
print("  • Можна вставляти lambda функції для трансформації даних\n")


# ============================================================================
# ПРИКЛАД 3: Паралельне виконання
# ============================================================================

print("\n" + "=" * 80)
print("3. ПАРАЛЕЛЬНЕ ВИКОНАННЯ: Кілька операцій одночасно")
print("=" * 80 + "\n")

print("❌ СТАРИЙ СПОСІБ (v0.x) - Складно і не очевидно")
print("-" * 80)
print("""
import asyncio

# Потрібно вручну керувати async викликами
async def run_parallel():
    results = await asyncio.gather(
        chain1.arun(input1),
        chain2.arun(input2),
        chain3.arun(input3)
    )
    return results

# Або використовувати router chains з додатковою логікою
""")

print("\n✅ НОВИЙ СПОСІБ (v1.0) - RunnableParallel")
print("-" * 80)
print("""
from langchain_core.runnables import RunnableParallel

prompt1 = ChatPromptTemplate.from_template("Переваги {topic}")
prompt2 = ChatPromptTemplate.from_template("Недоліки {topic}")
prompt3 = ChatPromptTemplate.from_template("Альтернативи {topic}")

# Автоматично виконується паралельно!
parallel_chain = RunnableParallel(
    pros=prompt1 | model | StrOutputParser(),
    cons=prompt2 | model | StrOutputParser(),
    alternatives=prompt3 | model | StrOutputParser()
)

result = parallel_chain.invoke({"topic": "microservices"})
# result = {"pros": "...", "cons": "...", "alternatives": "..."}
""")

print("\n🎯 ЩО ПОКРАЩИЛОСЬ:")
print("  • Не потрібно вручну керувати async")
print("  • Автоматична паралелізація")
print("  • Результат у зручному dict форматі")
print("  • Працює і в sync, і в async режимах\n")

# Демонстрація
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
parallel = RunnableParallel(
    short=ChatPromptTemplate.from_template("Одне слово про {topic}") | model | StrOutputParser(),
    emoji=ChatPromptTemplate.from_template("Один емодзі для {topic}") | model | StrOutputParser(),
)

result = parallel.invoke({"topic": "Python"})
print(f"📝 Результат паралельного виконання:")
print(f"   Коротко: {result['short']}")
print(f"   Емодзі: {result['emoji']}\n")


# ============================================================================
# ПРИКЛАД 4: Streaming
# ============================================================================

print("\n" + "=" * 80)
print("4. STREAMING: Отримання відповіді частинами")
print("=" * 80 + "\n")

print("❌ СТАРИЙ СПОСІБ (v0.x) - Різні API для різних компонентів")
print("-" * 80)
print("""
# Для LLM
for chunk in llm.stream("prompt"):
    print(chunk, end="")

# Для Chain - інший спосіб
chain = LLMChain(...)
async for chunk in chain.astream(inputs):
    print(chunk, end="")

# Не всі компоненти підтримували streaming
""")

print("\n✅ НОВИЙ СПОСІБ (v1.0) - Єдиний інтерфейс .stream()")
print("-" * 80)
print("""
# Все що має Runnable інтерфейс підтримує .stream()
chain = prompt | model | StrOutputParser()

# Просто викликаємо .stream()
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)

# Працює для будь-якого ланцюга, незалежно від складності!
""")

print("\n🎯 ЩО ПОКРАЩИЛОСЬ:")
print("  • Єдиний .stream() метод для всього")
print("  • Весь ланцюг підтримує streaming, а не лише LLM")
print("  • Можна стрімити результат паралельних операцій")
print("  • Async варіант: .astream()\n")


# ============================================================================
# ПРИКЛАД 5: Batch обробка
# ============================================================================

print("\n" + "=" * 80)
print("5. BATCH ОБРОБКА: Обробка багатьох входів одночасно")
print("=" * 80 + "\n")

print("❌ СТАРИЙ СПОСІБ (v0.x) - Вручну в циклі або apply")
print("-" * 80)
print("""
# Вручну в циклі
results = []
for input_data in inputs:
    result = chain.run(input_data)
    results.append(result)

# Або через apply (не завжди доступний)
results = chain.apply(inputs)
""")

print("\n✅ НОВИЙ СПОСІБ (v1.0) - Вбудований .batch()")
print("-" * 80)
print("""
chain = prompt | model | StrOutputParser()

inputs = [
    {"topic": "Python"},
    {"topic": "JavaScript"},
    {"topic": "Rust"}
]

# Автоматично оптимізується для batch обробки!
results = chain.batch(inputs)
# ['Про Python...', 'Про JavaScript...', 'Про Rust...']
""")

print("\n🎯 ЩО ПОКРАЩИЛОСЬ:")
print("  • Автоматична batch оптимізація")
print("  • Менше API викликів → швидше і дешевше")
print("  • Єдиний .batch() для всіх компонентів")
print("  • Async варіант: .abatch()\n")

# Демонстрація
chain = (
    ChatPromptTemplate.from_template("Одне слово про {lang}")
    | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    | StrOutputParser()
)

inputs = [{"lang": "Python"}, {"lang": "JavaScript"}, {"lang": "Go"}]
results = chain.batch(inputs)

print("📝 Batch результати:")
for inp, res in zip(inputs, results):
    print(f"   {inp['lang']}: {res}")


# ============================================================================
# ПІДСУМОК
# ============================================================================

print("\n\n" + "=" * 80)
print("📊 ПІДСУМОК ЗМІН v0.x → v1.0")
print("=" * 80 + "\n")

print("┌─────────────────────┬──────────────────────────┬──────────────────────────┐")
print("│ Що робимо           │ v0.x (Старе)             │ v1.0 (Нове - LCEL)       │")
print("├─────────────────────┼──────────────────────────┼──────────────────────────┤")
print("│ Композиція          │ LLMChain, спец. класи    │ Оператор |              │")
print("│ Виклик              │ .run(), .predict()       │ .invoke()                │")
print("│ Streaming           │ Різні API                │ .stream()                │")
print("│ Batch               │ .apply() або цикл        │ .batch()                 │")
print("│ Async               │ .arun(), .apredict()     │ .ainvoke(), .astream()   │")
print("│ Паралельність       │ asyncio.gather()         │ RunnableParallel         │")
print("│ Читабельність       │ ⭐⭐ Багато boilerplate   │ ⭐⭐⭐⭐⭐ Дуже чисто      │")
print("│ Простота            │ ⭐⭐ Потрібно знати класи │ ⭐⭐⭐⭐⭐ Інтуїтивно      │")
print("└─────────────────────┴──────────────────────────┴──────────────────────────┘")

print("\n💡 КЛЮЧОВІ ПЕРЕВАГИ LCEL:")
print("  1. Єдиний Runnable інтерфейс для всього")
print("  2. Композиція через | оператор (як Unix pipes)")
print("  3. Автоматична підтримка sync/async/streaming/batch")
print("  4. Менше коду, більше ясності")
print("  5. Легше тестувати та дебажити")
print("  6. Кращі можливості для трейсингу (LangSmith)")

print("\n" + "=" * 80)
