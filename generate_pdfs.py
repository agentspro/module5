#!/usr/bin/env python3
"""
Generate two educational PDF documents for agents_v1 and agents_v2.
Uses fpdf2 with Arial Unicode for Ukrainian text support.
"""

from fpdf import FPDF
import os

FONT_PATH = "/Library/Fonts/Arial Unicode.ttf"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


class UkrainianPDF(FPDF):
    """PDF with Ukrainian text support and consistent styling."""

    def __init__(self):
        super().__init__()
        self.add_font("ArialUnicode", "", FONT_PATH, uni=True)
        self.add_font("ArialUnicode", "B", FONT_PATH, uni=True)
        self.add_font("ArialUnicode", "I", FONT_PATH, uni=True)
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self.page_no() > 1:
            self.set_font("ArialUnicode", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 8, self._header_text, align="R")
            self.ln(4)
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(6)
            self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-15)
        self.set_font("ArialUnicode", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Сторінка {self.page_no()}/{{nb}}", align="C")

    def title_page(self, title, subtitle, description):
        self.add_page()
        self.ln(50)
        self.set_font("ArialUnicode", "B", 28)
        self.set_text_color(30, 60, 120)
        self.multi_cell(0, 14, title, align="C")
        self.ln(8)
        self.set_font("ArialUnicode", "", 16)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 10, subtitle, align="C")
        self.ln(20)
        self.set_font("ArialUnicode", "", 11)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 7, description, align="C")
        self.ln(30)
        self.set_font("ArialUnicode", "I", 10)
        self.set_text_color(128, 128, 128)
        self.cell(0, 8, "agents.pro / module5 - Навчальний курс", align="C")
        self.ln(6)
        self.cell(0, 8, "2025", align="C")

    def chapter_title(self, num, title):
        self.add_page()
        self.set_font("ArialUnicode", "B", 22)
        self.set_text_color(30, 60, 120)
        self.cell(0, 14, f"{num}. {title}")
        self.ln(10)
        self.set_draw_color(30, 60, 120)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(10)
        self.set_text_color(0, 0, 0)

    def section_title(self, title):
        self.ln(4)
        self.set_font("ArialUnicode", "B", 14)
        self.set_text_color(50, 80, 140)
        self.cell(0, 10, title)
        self.ln(10)
        self.set_text_color(0, 0, 0)

    def subsection_title(self, title):
        self.ln(2)
        self.set_font("ArialUnicode", "B", 12)
        self.set_text_color(70, 70, 70)
        self.cell(0, 8, title)
        self.ln(8)
        self.set_text_color(0, 0, 0)

    def body_text(self, text):
        self.set_font("ArialUnicode", "", 10)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def bullet_list(self, items):
        self.set_font("ArialUnicode", "", 10)
        for item in items:
            self.cell(8)
            self.cell(5, 6, "\u2022")
            self.multi_cell(0, 6, item)
            self.ln(1)
        self.ln(2)

    def code_block(self, code, title=None):
        if title:
            self.set_font("ArialUnicode", "I", 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 6, title)
            self.ln(5)
        self.set_fill_color(240, 240, 245)
        self.set_draw_color(200, 200, 210)
        self.set_font("ArialUnicode", "", 8)
        self.set_text_color(40, 40, 40)
        x = self.get_x()
        y = self.get_y()
        lines = code.strip().split("\n")
        block_h = len(lines) * 5 + 6
        if y + block_h > 270:
            self.add_page()
            y = self.get_y()
        self.rect(10, y, 190, block_h, style="DF")
        self.set_xy(13, y + 3)
        for line in lines:
            self.cell(0, 5, line)
            self.ln(5)
            self.set_x(13)
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def info_box(self, title, text):
        self.set_fill_color(230, 240, 255)
        self.set_draw_color(100, 140, 200)
        y = self.get_y()
        lines = text.split("\n")
        box_h = len(lines) * 6 + 16
        if y + box_h > 270:
            self.add_page()
            y = self.get_y()
        self.rect(10, y, 190, box_h, style="DF")
        self.set_xy(14, y + 3)
        self.set_font("ArialUnicode", "B", 10)
        self.set_text_color(30, 60, 120)
        self.cell(0, 6, title)
        self.ln(7)
        self.set_x(14)
        self.set_font("ArialUnicode", "", 9)
        self.set_text_color(40, 40, 40)
        for line in lines:
            self.cell(0, 6, line)
            self.ln(6)
            self.set_x(14)
        self.set_y(y + box_h + 4)
        self.set_text_color(0, 0, 0)

    def table(self, headers, rows):
        col_w = 190 / len(headers)
        self.set_font("ArialUnicode", "B", 9)
        self.set_fill_color(30, 60, 120)
        self.set_text_color(255, 255, 255)
        for h in headers:
            self.cell(col_w, 7, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("ArialUnicode", "", 9)
        self.set_text_color(0, 0, 0)
        fill = False
        for row in rows:
            if fill:
                self.set_fill_color(245, 245, 250)
            else:
                self.set_fill_color(255, 255, 255)
            for cell in row:
                self.cell(col_w, 7, cell, border=1, fill=True)
            self.ln()
            fill = not fill
        self.ln(4)


# ============================================================================
# DOCUMENT 1: agents_v1 guide
# ============================================================================

def generate_v1_guide():
    pdf = UkrainianPDF()
    pdf._header_text = "LangChain 1.0 & LangGraph 1.0 - Навчальний посібник"
    pdf.alias_nb_pages()

    # --- Title page ---
    pdf.title_page(
        "LangChain 1.0 &\nLangGraph 1.0",
        "Навчальний посібник з мультиагентних систем",
        "Від базового агента до Supervisor Pattern.\n"
        "Покрокові пояснення, архітектурні рішення, приклади коду.\n"
        "Модуль 5 курсу agents.pro"
    )

    # === Chapter 1: Introduction ===
    pdf.chapter_title(1, "Вступ")
    pdf.body_text(
        "Цей документ є навчальним посібником до модуля agents_v1, який демонструє "
        "побудову мультиагентних AI-систем за допомогою двох ключових фреймворків від LangChain:"
    )
    pdf.section_title("Що таке LangChain 1.0?")
    pdf.body_text(
        "LangChain 1.0 (реліз жовтень 2025) - це стабільний фреймворк для розробки "
        "додатків на базі великих мовних моделей (LLM). Ключові особливості:\n"
        "- Стабільний API з гарантією зворотної сумісності до версії 2.0\n"
        "- Функція create_agent() для швидкого створення агентів\n"
        "- Callback-система для розширення поведінки агентів\n"
        "- Інтеграція з LangSmith для моніторингу та трейсингу"
    )
    pdf.section_title("Що таке LangGraph 1.0?")
    pdf.body_text(
        "LangGraph 1.0 - це бібліотека для побудови складних робочих процесів (workflows) "
        "з використанням графів станів (state graphs). Основні можливості:\n"
        "- StateGraph для оркестрації мультиагентних систем\n"
        "- Checkpointing (MemorySaver) для збереження стану\n"
        "- Conditional edges для динамічної маршрутизації\n"
        "- Підтримка Supervisor Pattern для координації агентів"
    )
    pdf.section_title("Чому це важливо?")
    pdf.body_text(
        "Мультиагентні системи дозволяють вирішувати складні задачі, розділяючи їх між "
        "спеціалізованими агентами. Кожен агент має свою роль, інструменти та контекст. "
        "Це забезпечує:\n"
        "- Модульність: кожен агент відповідає за свою частину\n"
        "- Масштабованість: легко додавати нових агентів\n"
        "- Надійність: помилка одного агента не зупиняє систему\n"
        "- Спостережуваність: кожен крок можна відстежити"
    )

    # === Chapter 2: Architecture ===
    pdf.chapter_title(2, "Архітектура")
    pdf.body_text(
        "Модуль agents_v1 побудований як прогресія від простого до складного. "
        "Кожен наступний скрипт базується на концепціях попереднього:"
    )
    pdf.info_box("Прогресія прикладів", (
        "01_basic_agent.py      -> create_agent API, реальні tools\n"
        "02_agent_middleware.py  -> Callback handlers для моніторингу\n"
        "03_rag_agent.py        -> StateGraph, Agentic RAG pattern\n"
        "04_multiagent.py       -> Supervisor Pattern, 4 агенти"
    ))
    pdf.section_title("Загальна схема архітектури")
    pdf.body_text(
        "Всі приклади використовують модель gpt-4o-mini через langchain-openai. "
        "State описується як TypedDict з Annotated[List, operator.add] для акумулювання "
        "повідомлень. Структуровані LLM-відповіді використовують llm.with_structured_output(PydanticModel). "
        "Маршрутизація графу реалізована через add_conditional_edges з routing-функцією, "
        "що повертає строкові ключі."
    )
    pdf.table(
        ["Компонент", "Бібліотека", "Призначення"],
        [
            ["LLM", "langchain-openai", "Виклик gpt-4o-mini"],
            ["Агент", "langchain.agents", "create_agent()"],
            ["Граф", "langgraph", "StateGraph, workflow"],
            ["Векторне сховище", "faiss-cpu", "Similarity search"],
            ["Embeddings", "OpenAIEmbeddings", "text-embedding-3-small"],
            ["Checkpointing", "MemorySaver", "Збереження стану"],
        ]
    )

    # === Chapter 3: Example 01 ===
    pdf.chapter_title(3, "Приклад 01: Базовий агент")
    pdf.body_text(
        "Файл: agents_v1/01_basic_agent.py\n\n"
        "Перший приклад демонструє створення агента з реальними інструментами "
        "через LangChain 1.0 API. Агент здатний отримувати погоду, шукати в інтернеті "
        "та виконувати математичні обчислення."
    )
    pdf.section_title("Реальні інструменти (Tools)")
    pdf.body_text(
        "На відміну від багатьох навчальних прикладів, тут використовуються реальні API. "
        "Кожен tool створюється за допомогою декоратора @tool з langchain_core.tools:"
    )
    pdf.subsection_title("1. get_weather - OpenWeatherMap API")
    pdf.code_block("""@tool
def get_weather(location: str) -> str:
    \"\"\"Get current weather for a specific location.\"\"\"
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": location, "appid": api_key, "units": "metric"}
    response = requests.get(url, params=params, timeout=10)
    data = response.json()
    return f"Weather in {location}: {data['main']['temp']}C" """)

    pdf.subsection_title("2. web_search - DuckDuckGo")
    pdf.code_block("""@tool
def web_search(query: str) -> str:
    \"\"\"Search the web using DuckDuckGo.\"\"\"
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(f"{i}. {r['title']}\\n   {r['body'][:200]}")
    return "\\n".join(formatted)""")

    pdf.subsection_title("3. calculate - numexpr")
    pdf.code_block("""@tool
def calculate(expression: str) -> str:
    \"\"\"Safe mathematical calculations using numexpr.\"\"\"
    result = ne.evaluate(expression)
    return f"Result: {result}" """)

    pdf.section_title("Створення агента")
    pdf.body_text(
        "Агент створюється через create_agent() - новий API LangChain 1.0, що замінює "
        "застарілий AgentExecutor. Виклик агента - через agent.invoke():"
    )
    pdf.code_block("""from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_weather, calculate, web_search],
    system_prompt="You are a helpful AI assistant..."
)

# Виклик агента
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Kyiv?"}]
})""", "Створення та виклик агента:")

    pdf.section_title("Invoke Pattern")
    pdf.body_text(
        "В LangChain 1.0 агент використовує dict з ключем 'messages'. "
        "Результат - також dict з 'messages', де останнє повідомлення містить відповідь:\n\n"
        "result = agent.invoke({\"messages\": [...]})\n"
        "output = result[\"messages\"][-1].content"
    )

    # === Chapter 4: Example 02 ===
    pdf.chapter_title(4, "Приклад 02: Агент з Callbacks")
    pdf.body_text(
        "Файл: agents_v1/02_agent_with_middleware.py\n\n"
        "Цей приклад показує як розширити поведінку агента через систему "
        "callbacks - офіційний механізм LangChain 1.0 для перехоплення подій."
    )
    pdf.section_title("BaseCallbackHandler API")
    pdf.body_text(
        "Кожен callback handler наслідує BaseCallbackHandler і перевизначає потрібні методи:\n"
        "- on_llm_start / on_llm_end - перед/після виклику LLM\n"
        "- on_tool_start / on_tool_end - перед/після виклику tool\n"
        "- on_agent_action - при дії агента (вибір tool)"
    )
    pdf.section_title("4 типи Callback Handlers")

    pdf.subsection_title("1. LoggingCallback")
    pdf.body_text(
        "Детальне логування всіх операцій: виклики LLM, використання tools, "
        "таймстемпи та статистика."
    )
    pdf.code_block("""class LoggingCallback(BaseCallbackHandler):
    def __init__(self):
        self.llm_calls = 0
        self.tool_calls = 0
        self.logs = []

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.llm_calls += 1
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "llm_start",
            "prompt_length": len(prompts[0]) if prompts else 0
        }
        self.logs.append(log_entry)

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.tool_calls += 1
        tool_name = serialized.get("name", "unknown")
        print(f"TOOL CALL: {tool_name}, Input: {input_str}")""")

    pdf.subsection_title("2. SecurityCallback")
    pdf.body_text(
        "Перехоплює та логує спроби використання небезпечних інструментів "
        "(execute_trade, send_notification). В продакшені може блокувати виклики."
    )
    pdf.code_block("""class SecurityCallback(BaseCallbackHandler):
    def __init__(self):
        self.high_risk_tools = ["execute_trade", "send_notification"]
        self.blocked_calls = 0

    def on_agent_action(self, action: AgentAction, **kwargs):
        if action.tool in self.high_risk_tools:
            self.blocked_calls += 1
            print(f"HIGH-RISK ACTION: {action.tool}")""")

    pdf.subsection_title("3. TokenCountCallback")
    pdf.body_text(
        "Підраховує приблизну кількість використаних токенів та попереджає "
        "при наближенні до ліміту. Використовує формулу: ~1 токен = 4 символи."
    )

    pdf.subsection_title("4. PerformanceCallback")
    pdf.body_text(
        "Вимірює час виконання кожного LLM-виклику та tool-виклику. "
        "Використовує on_llm_start/on_llm_end та on_tool_start/on_tool_end."
    )

    pdf.section_title("Підключення Callbacks")
    pdf.code_block("""# Callbacks передаються через config при виклику
result = agent.invoke(
    {"messages": [{"role": "user", "content": query}]},
    config={"callbacks": [logging_cb, security_cb, token_cb, performance_cb]}
)""", "Передача callbacks в agent.invoke():")

    pdf.section_title("Phoenix Tracing")
    pdf.body_text(
        "Приклад також інтегрує Arize Phoenix для трейсингу через OpenTelemetry. "
        "Якщо Phoenix сервер доступний на localhost:4317, трейсинг вмикається автоматично:"
    )
    pdf.code_block("""from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register
import phoenix as px

tracer_provider = register()
LangChainInstrumentor(tracer_provider=tracer_provider).instrument()""")

    # === Chapter 5: Example 03 ===
    pdf.chapter_title(5, "Приклад 03: Agentic RAG")
    pdf.body_text(
        "Файл: agents_v1/03_rag_agent_langgraph.py\n\n"
        "Цей приклад реалізує Agentic RAG (Retrieval-Augmented Generation) з використанням "
        "LangGraph StateGraph. На відміну від базового RAG, тут агент динамічно керує "
        "стратегією пошуку: перевіряє релевантність знайдених документів і за потреби "
        "переписує запит."
    )
    pdf.section_title("Архітектура потоку")
    pdf.info_box("RAG Flow", (
        "User Query\n"
        "    |-> Retrieve Docs (FAISS)\n"
        "        |-> Grade Relevance (LLM + structured output)\n"
        "            |-> relevant?  -> Generate Answer -> END\n"
        "            |-> irrelevant? -> Rewrite Query -> Retrieve Again"
    ))

    pdf.section_title("RAGState - Визначення стану")
    pdf.code_block("""class RAGState(TypedDict):
    question: str                          # Поточне питання
    retrieved_docs: List[Document]         # Знайдені документи
    relevance_grade: str                   # "relevant" або "irrelevant"
    rewrite_count: int                     # Лічильник перезаписів
    answer: str                            # Фінальна відповідь
    reasoning: Annotated[List[str], add]   # Кроки міркування""",
                   "RAGState TypedDict:")

    pdf.section_title("Knowledge Base (FAISS)")
    pdf.body_text(
        "Векторна база знань створюється з Document об'єктів. Кожен документ має "
        "page_content та metadata. Для embeddings використовується text-embedding-3-small:"
    )
    pdf.code_block("""embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}  # Топ-2 найрелевантніші
)""")

    pdf.section_title("4 ноди графу")
    pdf.subsection_title("1. retrieve_documents")
    pdf.body_text("Використовує FAISS retriever для пошуку релевантних документів за запитом.")

    pdf.subsection_title("2. grade_documents")
    pdf.body_text(
        "LLM з structured output оцінює релевантність знайдених документів. "
        "Використовує Pydantic модель GradeOutput з полями relevance та reasoning:"
    )
    pdf.code_block("""class GradeOutput(BaseModel):
    relevance: str = Field(description="'relevant' or 'irrelevant'")
    reasoning: str = Field(description="Why this grade")

structured_llm = llm.with_structured_output(GradeOutput)
grade_result = chain.invoke({"question": q, "documents": docs_text})""")

    pdf.subsection_title("3. rewrite_query")
    pdf.body_text("Переписує запит для кращого retrieval якщо документи нерелевантні. Максимум 2 спроби.")

    pdf.subsection_title("4. generate_answer")
    pdf.body_text("Генерує фінальну відповідь на основі знайдених документів та контексту.")

    pdf.section_title("Побудова графу")
    pdf.code_block("""workflow = StateGraph(RAGState)

# Ноди
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("grade", grade_documents)
workflow.add_node("rewrite", rewrite_query)
workflow.add_node("generate", generate_answer)

# Ребра
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", decide_next_step, {
    "generate": "generate",
    "rewrite": "rewrite"
})
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

# Компіляція з checkpointing
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)""", "Побудова StateGraph:")

    pdf.section_title("Checkpointing з MemorySaver")
    pdf.body_text(
        "MemorySaver зберігає стан виконання графу. Кожна сесія ідентифікується "
        "через thread_id, що дозволяє вести окремі розмови:\n\n"
        "config = {\"configurable\": {\"thread_id\": \"session_1\"}}\n"
        "result = agent.invoke(initial_state, config)"
    )

    # === Chapter 6: Example 04 ===
    pdf.chapter_title(6, "Приклад 04: Мультиагентна система")
    pdf.body_text(
        "Файл: agents_v1/04_multiagent_langgraph.py\n\n"
        "Найскладніший приклад реалізує Supervisor Pattern - архітектуру де центральний "
        "агент (Supervisor) координує команду спеціалізованих агентів."
    )
    pdf.section_title("Supervisor Pattern")
    pdf.body_text(
        "Supervisor отримує запит користувача і послідовно делегує роботу спеціалістам:\n"
        "1. Supervisor оцінює поточний стан\n"
        "2. Вирішує який агент має працювати далі\n"
        "3. Спеціаліст виконує роботу\n"
        "4. Результат повертається до Supervisor\n"
        "5. Цикл повторюється поки не буде FINISH"
    )

    pdf.section_title("MultiAgentState")
    pdf.code_block("""class MultiAgentState(TypedDict):
    messages: Annotated[List, operator.add]  # Комунікація
    question: str                            # Питання користувача
    current_agent: str                       # Активний агент
    retrieved_docs: List[Document]           # Документи (Researcher)
    analysis: str                            # Аналіз (Analyzer)
    final_answer: str                        # Відповідь (Synthesizer)
    supervisor_decision: str                 # Рішення supervisor
    iteration_count: int                     # Лічильник ітерацій""",
                   "Спільний state для всіх агентів:")

    pdf.section_title("4 спеціалізовані агенти")
    pdf.table(
        ["Агент", "Роль", "Операції"],
        [
            ["Supervisor", "Координатор", "Делегування, контроль"],
            ["Researcher", "Пошук", "RAG retrieval, оцінка якості"],
            ["Analyzer", "Аналіз", "Витяг insights, структуризація"],
            ["Synthesizer", "Синтез", "Фінальна відповідь"],
        ]
    )

    pdf.section_title("Structured Output для рішень")
    pdf.code_block("""class SupervisorDecision(BaseModel):
    next_agent: Literal["researcher","analyzer","synthesizer","FINISH"]
    reasoning: str

structured_llm = llm.with_structured_output(SupervisorDecision)
decision = structured_llm.invoke(messages)""",
                   "Supervisor використовує Pydantic для гарантованого формату рішень:")

    pdf.section_title("Побудова мультиагентного графу")
    pdf.code_block("""workflow = StateGraph(MultiAgentState)

# Ноди
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyzer", analyzer_node)
workflow.add_node("synthesizer", synthesizer_node)

# Entry point
workflow.set_entry_point("supervisor")

# Conditional routing від supervisor
workflow.add_conditional_edges("supervisor", route_after_supervisor, {
    "researcher": "researcher",
    "analyzer": "analyzer",
    "synthesizer": "synthesizer",
    "end": END
})

# Кожен агент повертається до supervisor
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("analyzer", "supervisor")
workflow.add_edge("synthesizer", "supervisor")""",
                   "Граф з циклом через supervisor:")

    # === Chapter 7: Key patterns ===
    pdf.chapter_title(7, "Ключові патерни та порівняння")
    pdf.section_title("Коли використовувати який підхід?")
    pdf.table(
        ["Підхід", "Коли використовувати", "Складність"],
        [
            ["create_agent", "Прості задачі з tools", "Низька"],
            ["Callbacks", "Моніторинг, безпека, логування", "Середня"],
            ["StateGraph RAG", "Адаптивний пошук документів", "Середня"],
            ["Supervisor", "Складні мультиагентні задачі", "Висока"],
        ]
    )
    pdf.section_title("Ключові патерни LangChain 1.0")
    pdf.bullet_list([
        "Всі LLM виклики через gpt-4o-mini (langchain-openai)",
        "State як TypedDict з Annotated[List, operator.add] для акумулювання",
        "Structured output через llm.with_structured_output(PydanticModel)",
        "Routing через add_conditional_edges з функцією що повертає str",
        "Checkpointing через MemorySaver (dev) або PostgresSaver (prod)",
        "Callbacks для cross-cutting concerns (logging, security, metrics)",
    ])
    pdf.section_title("Типові помилки")
    pdf.bullet_list([
        "Використання AgentExecutor замість create_agent (deprecated)",
        "Забути thread_id при checkpointing - стани змішуються",
        "Не обмежувати кількість rewrites/iterations - нескінченний цикл",
        "Ігнорування structured output - нестабільний парсинг відповідей",
    ])

    # === Chapter 8: Setup ===
    pdf.chapter_title(8, "Налаштування та запуск")
    pdf.section_title("Швидкий старт")
    pdf.code_block("""cd agents_v1
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # заповніть API ключі
python 01_basic_agent.py""")

    pdf.section_title("Необхідні змінні оточення")
    pdf.table(
        ["Змінна", "Обов'язкова", "Джерело"],
        [
            ["OPENAI_API_KEY", "Так", "platform.openai.com"],
            ["OPENWEATHERMAP_API_KEY", "Для 01_*", "openweathermap.org/api"],
            ["LANGCHAIN_TRACING_V2", "Ні", "true для LangSmith"],
            ["LANGCHAIN_API_KEY", "Ні", "smith.langchain.com"],
        ]
    )
    pdf.section_title("Ключові залежності")
    pdf.code_block("""langchain>=1.0
langgraph>=1.0
langchain-openai
faiss-cpu
yfinance
duckduckgo-search
numexpr
arize-phoenix
openinference-instrumentation-langchain""", "requirements.txt (основні):")

    pdf.section_title("Python версія")
    pdf.body_text(
        "Рекомендовано Python 3.10-3.13. Уникайте Python 3.14 через проблеми "
        "сумісності з Pydantic."
    )

    out_path = os.path.join(OUTPUT_DIR, "agents_v1_guide.pdf")
    pdf.output(out_path)
    print(f"Generated: {out_path}")
    return out_path


# ============================================================================
# DOCUMENT 2: agents_v2 guide
# ============================================================================

def generate_v2_guide():
    pdf = UkrainianPDF()
    pdf._header_text = "CrewAI Framework - Навчальний посібник"
    pdf.alias_nb_pages()

    # --- Title page ---
    pdf.title_page(
        "CrewAI Framework",
        "Рольова модель мультиагентних систем",
        "Від Sequential процесу до Memory-enabled Crew.\n"
        "Покрокові пояснення, інструменти, приклади коду.\n"
        "Модуль 5 курсу agents.pro"
    )

    # === Chapter 1: Introduction ===
    pdf.chapter_title(1, "Вступ")
    pdf.body_text(
        "Цей документ є навчальним посібником до модуля agents_v2, що демонструє "
        "побудову мультиагентних систем за допомогою CrewAI - фреймворку, заснованого "
        "на рольовій моделі взаємодії агентів."
    )
    pdf.section_title("Що таке CrewAI?")
    pdf.body_text(
        "CrewAI (версія 1.4.0+) - це Python-фреймворк для створення команд AI-агентів, "
        "де кожен агент має визначену роль, мету та історію. Ключова метафора - це 'екіпаж' "
        "(crew), де агенти працюють разом для досягнення спільної мети.\n\n"
        "Основні переваги CrewAI:\n"
        "- Рольова модель: агенти визначаються через role/goal/backstory\n"
        "- Два типи процесів: Sequential та Hierarchical\n"
        "- Параметризація задач через {variable} placeholders\n"
        "- Вбудована система памяті (memory)\n"
        "- Інтеграція з LangChain tools та власна бібліотека crewai_tools"
    )
    pdf.section_title("Відмінність від LangChain/LangGraph")
    pdf.table(
        ["Аспект", "LangChain/LangGraph", "CrewAI"],
        [
            ["Підхід", "Граф станів", "Рольова модель"],
            ["Оркестрація", "StateGraph + edges", "Crew + Process"],
            ["Агенти", "Функції-ноди", "Role/Goal/Backstory"],
            ["Координація", "Routing functions", "Manager або Sequential"],
            ["Складність", "Більше контролю", "Простіший API"],
            ["Налаштування", "Гранулярне", "Декларативне"],
        ]
    )

    # === Chapter 2: Key Concepts ===
    pdf.chapter_title(2, "Ключові концепції")
    pdf.section_title("Agent - Агент")
    pdf.body_text(
        "Agent - це основна одиниця CrewAI. Кожен агент визначається через:\n"
        "- role: роль агента (напр., 'Senior Research Analyst')\n"
        "- goal: мета, яку агент намагається досягти\n"
        "- backstory: контекст та досвід агента (покращує якість відповідей)\n"
        "- llm: модель для використання (напр., 'gpt-4o-mini')\n"
        "- tools: список інструментів доступних агенту\n"
        "- allow_delegation: чи може агент делегувати задачі іншим"
    )
    pdf.code_block("""researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments about {topic}",
    backstory="You are a seasoned research analyst with 10 years...",
    verbose=True,
    allow_delegation=False,
    llm="gpt-4o-mini"
)""", "Приклад створення агента:")

    pdf.section_title("Task - Задача")
    pdf.body_text(
        "Task визначає конкретну роботу для агента:\n"
        "- description: що потрібно зробити (підтримує {variable} placeholders)\n"
        "- expected_output: очікуваний результат\n"
        "- agent: відповідальний агент"
    )
    pdf.code_block("""research_task = Task(
    description="Conduct comprehensive research on {topic}...",
    expected_output="A detailed research report with 10-15 bullet points...",
    agent=researcher
)""", "Приклад створення задачі:")

    pdf.section_title("Crew - Екіпаж")
    pdf.body_text(
        "Crew об'єднує агентів та задачі. Конфігурується через:\n"
        "- agents: список агентів\n"
        "- tasks: список задач (порядок важливий для sequential)\n"
        "- process: Process.sequential або Process.hierarchical\n"
        "- verbose: детальне логування\n"
        "- memory: включення системи памяті"
    )

    pdf.section_title("Kickoff - Запуск")
    pdf.body_text(
        "Метод kickoff() запускає виконання crew. Параметри передаються через "
        "inputs dict і підставляються в {variable} placeholders:"
    )
    pdf.code_block("""crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff(inputs={
    "topic": "Multi-Agent AI Systems with LangChain and CrewAI"
})""", "Створення та запуск crew:")

    # === Chapter 3: Example 01 ===
    pdf.chapter_title(3, "Приклад 01: Базовий Crew")
    pdf.body_text(
        "Файл: agents_v2/01_basic_crew.py\n\n"
        "Перший приклад демонструє Sequential процес - найпростішу модель "
        "оркестрації, де задачі виконуються послідовно."
    )
    pdf.section_title("Структура: Researcher -> Writer -> Editor")
    pdf.body_text(
        "Три агенти працюють послідовно над створенням контенту:\n"
        "1. Senior Research Analyst - досліджує тему\n"
        "2. Tech Content Writer - пише статтю на основі дослідження\n"
        "3. Senior Content Editor - редагує та полірує результат"
    )
    pdf.info_box("Sequential Process", (
        "Researcher -> Результат дослідження\n"
        "    -> Writer -> Чернетка статті\n"
        "        -> Editor -> Фінальний текст"
    ))

    pdf.section_title("Параметризація через {topic}")
    pdf.body_text(
        "Всі задачі використовують {topic} placeholder. При виклику kickoff() "
        "передається конкретне значення, яке підставляється в усі описи задач та цілі агентів:"
    )
    pdf.code_block("""# В описі задачі:
description="Conduct comprehensive research on {topic}..."

# При запуску:
result = crew.kickoff(inputs={
    "topic": "Multi-Agent AI Systems"
})""")

    pdf.section_title("Ключові патерни")
    pdf.bullet_list([
        "allow_delegation=False - кожен агент виконує тільки свою задачу",
        "verbose=True - детальне логування для відлагодження",
        "Вихід кожної задачі автоматично передається наступній",
        "Crew можна перевикористати з іншим topic через kickoff()",
    ])

    # === Chapter 4: Example 02 ===
    pdf.chapter_title(4, "Приклад 02: Ієрархічний Crew")
    pdf.body_text(
        "Файл: agents_v2/02_hierarchical_crew.py\n\n"
        "Цей приклад демонструє Hierarchical процес з автоматично створеним "
        "менеджером, який координує команду з 6 спеціалістів."
    )
    pdf.section_title("Hierarchical Process")
    pdf.body_text(
        "В ієрархічному режимі CrewAI автоматично створює Manager-агента, який:\n"
        "1. Планує стратегію виконання\n"
        "2. Делегує задачі відповідним спеціалістам\n"
        "3. Перевіряє якість результатів\n"
        "4. Координує передачу даних між агентами"
    )
    pdf.info_box("Ієрархічна структура", (
        "MANAGER (auto-created)\n"
        "  |-- Requirements Analyst\n"
        "  |-- Software Architect\n"
        "  |-- Backend Developer\n"
        "  |-- Frontend Developer\n"
        "  |-- QA Engineer\n"
        "  |-- Documentation Specialist"
    ))

    pdf.section_title("6 спеціалізованих агентів")
    pdf.table(
        ["Агент", "Роль", "Інструменти"],
        [
            ["Requirements Analyst", "Аналіз вимог", "-"],
            ["Software Architect", "Дизайн архітектури", "-"],
            ["Backend Developer", "Backend розробка", "-"],
            ["Frontend Developer", "Frontend розробка", "-"],
            ["QA Engineer", "Тестування", "-"],
            ["Documentation Spec.", "Збереження у файл", "FileWriterTool"],
        ]
    )

    pdf.section_title("FileWriterTool")
    pdf.body_text(
        "Documentation Specialist використовує FileWriterTool для збереження "
        "результатів у файл project_deliverables.md:"
    )
    pdf.code_block("""from crewai_tools import FileWriterTool

documentation_specialist = Agent(
    role="Documentation Specialist",
    goal="Compile and save all project deliverables...",
    tools=[FileWriterTool()],
    llm="gpt-4o-mini"
)""")

    pdf.section_title("Конфігурація manager_llm")
    pdf.body_text(
        "Параметр manager_llm обов'язковий для ієрархічного процесу. "
        "Manager не входить в список agents - він створюється автоматично:"
    )
    pdf.code_block("""crew = Crew(
    agents=[analyst, architect, backend_dev, frontend_dev, qa, docs],
    tasks=[req_task, arch_task, back_task, front_task, test_task, save_task],
    process=Process.hierarchical,
    verbose=True,
    manager_llm="gpt-4o-mini"
)""")

    # === Chapter 5: Example 03 ===
    pdf.chapter_title(5, "Приклад 03: Crew з інструментами")
    pdf.body_text(
        "Файл: agents_v2/03_research_crew_with_tools.py\n\n"
        "Цей приклад показує інтеграцію різних типів інструментів: "
        "custom tools через @tool декоратор, crewai_tools бібліотеку "
        "та LangChain tools."
    )

    pdf.section_title("Типи інструментів")
    pdf.subsection_title("1. Custom tools (@tool декоратор)")
    pdf.body_text(
        "Власні функції, обгорнуті декоратором @tool з langchain_core.tools. "
        "Кожен tool має назву, опис та параметри:"
    )
    pdf.code_block("""from langchain_core.tools import tool

@tool
def analyze_data(data_json: str) -> str:
    \"\"\"Analyze JSON data and return statistical insights.\"\"\"
    data = json.loads(data_json)
    if isinstance(data, list):
        return f"Dataset contains {len(data)} items."
    return "Data analyzed successfully"

@tool
def calculate_metrics(expression: str) -> str:
    \"\"\"Safely evaluate mathematical expressions.\"\"\"
    allowed_names = {"abs": abs, "min": min, "max": max, "sum": sum}
    result = eval(expression, {"__builtins__": {}}, allowed_names)
    return f"Result: {result}" """)

    pdf.subsection_title("2. crewai_tools бібліотека")
    pdf.code_block("""from crewai_tools import FileReadTool, DirectoryReadTool

file_read_tool = FileReadTool()
directory_read_tool = DirectoryReadTool()""", "Готові інструменти:")

    pdf.subsection_title("3. LangChain tools")
    pdf.code_block("""from langchain_community.tools import DuckDuckGoSearchResults

search_tool = DuckDuckGoSearchResults(num_results=5)""", "Пошук через DuckDuckGo:")

    pdf.section_title("Призначення tools агентам")
    pdf.body_text(
        "Інструменти призначаються конкретним агентам через параметр tools. "
        "Кожен агент отримує тільки ті інструменти, які потрібні для його ролі:"
    )
    pdf.table(
        ["Агент", "Tools", "Призначення"],
        [
            ["Researcher", "FileRead, DirRead, DDG", "Збір інформації"],
            ["Analyst", "analyze_data, calculate", "Аналіз даних"],
            ["Writer", "(немає)", "Написання звіту"],
        ]
    )

    pdf.section_title("Множинна параметризація")
    pdf.body_text(
        "На відміну від Приклад 01, тут використовуються кілька параметрів:"
    )
    pdf.code_block("""inputs = {
    "research_topic": "Multi-Agent AI Frameworks Comparison",
    "focus_areas": "architecture, ease of use, tool integration",
    "target_audience": "Technical leaders and AI engineers"
}
result = crew.kickoff(inputs=inputs)""")

    # === Chapter 6: Example 04 ===
    pdf.chapter_title(6, "Приклад 04: Memory-enabled Crew")
    pdf.body_text(
        "Файл: agents_v2/04_memory_enabled_crew.py\n\n"
        "Цей приклад демонструє систему памяті CrewAI - здатність зберігати "
        "контекст між викликами kickoff() для персоналізованих відповідей."
    )
    pdf.section_title("Увімкнення памяті")
    pdf.code_block("""crew = Crew(
    agents=[context_analyzer, knowledge_curator, assistant],
    tasks=[context_task, knowledge_task, response_task],
    process=Process.sequential,
    verbose=True,
    memory=True,              # Увімкнути пам'ять
    embedder={
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small"
        }
    }
)""", "Конфігурація memory:")

    pdf.section_title("4 типи памяті")
    pdf.table(
        ["Тип", "Опис", "Збереження"],
        [
            ["Short-term", "Контекст поточної розмови", "В сесії"],
            ["Long-term", "Довготривала інформація", "Між сесіями"],
            ["Entity", "Дані про конкретні сутності", "Між сесіями"],
            ["Contextual", "Контекст задач та агентів", "В сесії"],
        ]
    )

    pdf.section_title("Multi-turn conversations")
    pdf.body_text(
        "Приклад симулює багатокрокову розмову, де кожен наступний виклик "
        "kickoff() враховує попередній контекст:"
    )
    pdf.code_block("""# Turn 1: Користувач ділиться контекстом
result1 = crew.kickoff(inputs={
    "user_request": "I'm working on a multi-agent AI project using LangChain...",
    "tone": "professional and helpful"
})

# Turn 2: Crew пам'ятає про LangChain проект
result2 = crew.kickoff(inputs={
    "user_request": "What's the best way to structure my agent code?",
    "tone": "professional and helpful"
})

# Turn 3: Crew пам'ятає весь контекст
result3 = crew.kickoff(inputs={
    "user_request": "Can you recommend testing strategies for this?",
    "tone": "professional and helpful"
})""")

    pdf.section_title("3 агенти для роботи з пам'яттю")
    pdf.bullet_list([
        "Context Analyzer - аналізує запит в контексті попередніх розмов",
        "Knowledge Curator - організовує та витягує релевантну інформацію з памяті",
        "Personal Assistant - створює персоналізовану відповідь",
    ])

    pdf.section_title("Embedder конфігурація")
    pdf.body_text(
        "Для роботи памяті потрібен embedder. CrewAI підтримує OpenAI, Cohere "
        "та HuggingFace embeddings. Рекомендовано text-embedding-3-small від OpenAI."
    )

    # === Chapter 7: Sequential vs Hierarchical ===
    pdf.chapter_title(7, "Sequential vs Hierarchical")
    pdf.section_title("Порівняння процесів")
    pdf.table(
        ["Аспект", "Sequential", "Hierarchical"],
        [
            ["Потік", "Лінійний A->B->C", "Manager делегує"],
            ["Гнучкість", "Низька", "Висока"],
            ["Передбачуваність", "Висока", "Середня"],
            ["Вартість (tokens)", "Нижча", "Вища (+manager)"],
            ["Складність debug", "Проста", "Складніша"],
            ["Кількість агентів", "2-4 оптимально", "5+ оптимально"],
        ]
    )
    pdf.section_title("Коли використовувати Sequential?")
    pdf.bullet_list([
        "Прості лінійні workflows (дослідження -> написання -> редагування)",
        "Передбачувана послідовність кроків",
        "Невелика кількість агентів (2-4)",
        "Обмежений бюджет на API виклики",
        "Потрібна простота відлагодження",
    ])
    pdf.section_title("Коли використовувати Hierarchical?")
    pdf.bullet_list([
        "Складні проекти з багатьма спеціалістами (5+)",
        "Потрібна валідація якості на кожному етапі",
        "Динамічне визначення порядку виконання",
        "Можливість перепризначення задач при невдачі",
        "Великі команди з чітким розподілом відповідальності",
    ])

    # === Chapter 8: Setup ===
    pdf.chapter_title(8, "Налаштування та запуск")
    pdf.section_title("Швидкий старт")
    pdf.code_block("""cd agents_v2
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # заповніть API ключі
python 01_basic_crew.py""")

    pdf.section_title("Необхідні змінні оточення")
    pdf.table(
        ["Змінна", "Обов'язкова", "Джерело"],
        [
            ["OPENAI_API_KEY", "Так", "platform.openai.com"],
            ["CREWAI_TELEMETRY", "Ні (false)", "Вимкнути телеметрію"],
        ]
    )
    pdf.section_title("Ключові залежності")
    pdf.code_block("""crewai>=1.4.0
crewai-tools>=0.38
langmem>=0.0.20
langchain>=1.0
duckduckgo-search""", "requirements.txt (основні):")

    pdf.section_title("Python версія")
    pdf.body_text(
        "Рекомендовано Python 3.10-3.13. Уникайте Python 3.14 через проблеми "
        "сумісності з Pydantic."
    )

    pdf.section_title("Важливі зауваження")
    pdf.bullet_list([
        "Кожен скрипт запускається окремо (не пакет)",
        "Скрипти використовують input() для пауз між тестами",
        "verbose=True рекомендовано для навчання і відлагодження",
        "Telemetry CrewAI краще вимкнути: CREWAI_TELEMETRY=false",
        "FileWriterTool в 02_* створює файли в поточній директорії",
    ])

    out_path = os.path.join(OUTPUT_DIR, "agents_v2_guide.pdf")
    pdf.output(out_path)
    print(f"Generated: {out_path}")
    return out_path


if __name__ == "__main__":
    print("Generating PDF guides...")
    v1 = generate_v1_guide()
    v2 = generate_v2_guide()
    print(f"\nDone! Files:\n  {v1}\n  {v2}")
