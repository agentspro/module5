# Multi-Agent AI Systems: LangChain 1.0, LangGraph 1.0 та CrewAI

Цей репозиторій містить практичні приклади мультиагентних систем, побудованих на найсучасніших фреймворках для AI-агентів.

## Структура репозиторію

```
module5/
├── agents_v1/          # LangChain 1.0 + LangGraph 1.0
│   ├── 01_basic_agent.py
│   ├── 02_agent_with_middleware.py
│   ├── 03_rag_agent_langgraph.py
│   ├── 04_multiagent_langgraph.py
│   ├── requirements.txt
│   ├── .env.example
│   └── README.md
│
└── agents_v2/          # CrewAI Framework
    ├── 01_basic_crew.py
    ├── 02_hierarchical_crew.py
    ├── 03_research_crew_with_tools.py
    ├── 04_memory_enabled_crew.py
    ├── requirements.txt
    ├── .env.example
    └── README.md
```

## agents_v1: LangChain 1.0 & LangGraph 1.0

Приклади, що демонструють найновіші можливості LangChain 1.0 та LangGraph 1.0:

### Ключові фічі
- **create_agent API** - новий спрощений API для створення агентів
- **StateGraph** - потужна оркестрація через графи станів
- **MemorySaver checkpointing** - персистентність між викликами
- **Middleware Architecture** - pre/post-processing hooks
- **Agentic RAG** - динамічний пошук з оцінкою релевантності
- **Supervisor Pattern** - ієрархічна координація агентів

### Агенти
1. **Basic Agent** - базовий агент з create_agent API
2. **Middleware Agent** - агент з logging, security, token limit middleware
3. **RAG Agent** - Agentic RAG з LangGraph StateGraph
4. **Multi-Agent System** - Supervisor Pattern з 4 спеціалізованими агентами

### Технології
- LangChain >= 1.0.0
- LangGraph >= 1.0.0
- OpenAI GPT-4o-mini
- FAISS vector store
- LangSmith tracing

[Детальна документація в agents_v1/README.md](agents_v1/README.md)

---

## agents_v2: CrewAI Framework

Приклади, що демонструють CrewAI - фреймворк для оркестрації колаборативних AI-агентів з рольовою моделлю.

### Ключові фічі
- **Role-playing agents** - агенти з чіткими ролями та цілями
- **Sequential & Hierarchical processes** - різні моделі виконання
- **Crew kickoff** - запуск команди агентів
- **Memory integration** - персистентна пам'ять між сесіями
- **Conversational Crew** - природна взаємодія з користувачем
- **Tools ecosystem** - інтеграція з LangChain tools та CrewAI toolkit

### Команди (Crews)
1. **Basic Crew** - проста команда з послідовним виконанням
2. **Hierarchical Crew** - ієрархічна структура з менеджером
3. **Research Crew with Tools** - команда дослідників з інструментами
4. **Memory-enabled Crew** - команда з персистентною пам'яттю

### Технології
- CrewAI >= 0.98.0
- OpenAI GPT-4o-mini
- LangChain tools integration
- LangMem для персистентної пам'яті
- Multimodal support

[Детальна документація в agents_v2/README.md](agents_v2/README.md)

---

## Швидкий старт

### Налаштування середовища

1. Клонуйте репозиторій:
```bash
git clone https://github.com/agentspro/module5.git
cd module5
```

2. Оберіть фреймворк (agents_v1 або agents_v2):
```bash
cd agents_v1  # або cd agents_v2
```

3. Встановіть залежності:
```bash
pip install -r requirements.txt
```

4. Налаштуйте змінні середовища:
```bash
cp .env.example .env
# Відредагуйте .env та додайте ваші API ключі
```

### Конфігурація API ключів

Створіть файл `.env` у відповідній папці:

```bash
# OpenAI API
OPENAI_API_KEY=sk-your-openai-api-key-here

# LangSmith (опціонально, для трейсингу)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your-langsmith-api-key-here
LANGCHAIN_PROJECT=your-project-name
```

### Запуск прикладів

**agents_v1 (LangChain/LangGraph):**
```bash
cd agents_v1
python 01_basic_agent.py
python 02_agent_with_middleware.py
python 03_rag_agent_langgraph.py
python 04_multiagent_langgraph.py
```

**agents_v2 (CrewAI):**
```bash
cd agents_v2
python 01_basic_crew.py
python 02_hierarchical_crew.py
python 03_research_crew_with_tools.py
python 04_memory_enabled_crew.py
```

---

## Порівняння фреймворків

### LangChain/LangGraph (agents_v1)

**Переваги:**
- Повний контроль над потоком виконання
- Гнучкість в архітектурі
- Потужна система middleware
- Детальний контроль над state management
- Відмінна інтеграція з LangSmith

**Використовуйте коли:**
- Потрібен детальний контроль над логікою
- Складні графи станів з умовними переходами
- Критична важливість observability
- Кастомні pattern'и оркестрації

### CrewAI (agents_v2)

**Переваги:**
- Швидка розробка з простим API
- Вбудована рольова модель
- Автоматична делегація задач
- Природна колаборація між агентами
- Hierarchical process out-of-the-box

**Використовуйте коли:**
- Потрібна швидка розробка multi-agent систем
- Чітка рольова структура команди
- Автоматична координація між агентами
- Focus на бізнес-логіку, а не на інфраструктуру

---

## Вимоги

- Python >= 3.10, < 3.14
- OpenAI API ключ
- (Опціонально) LangSmith API ключ для трейсингу

---

## Ресурси

### LangChain/LangGraph
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangSmith](https://smith.langchain.com/)

### CrewAI
- [CrewAI Documentation](https://docs.crewai.com/)
- [CrewAI GitHub](https://github.com/crewAIInc/crewAI)
- [CrewAI Examples](https://github.com/crewAIInc/crewAI-examples)

---

## Ліцензія

MIT License

---

## Автор

sanyaden <alex.denysyuk@gmail.com>

---

## Changelog

### 2025-11-09
- Створено agents_v1 з 4 агентами на LangChain 1.0 / LangGraph 1.0
- Додано agents_v2 з 4 прикладами на CrewAI
- Додано порівняльний аналіз фреймворків
