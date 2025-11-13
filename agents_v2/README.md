# CrewAI Examples: Multi-Agent Collaboration Framework

Практичні приклади використання CrewAI - фреймворку для створення колаборативних AI-агентів з рольовою моделлю.

## Огляд

CrewAI - це Python-фреймворк для оркестрації автономних AI-агентів, що працюють разом як команда. На відміну від одноагентних систем, CrewAI дозволяє створювати команди спеціалізованих агентів з чіткими ролями, цілями та процесами координації.

### Ключові переваги CrewAI

- **Швидка розробка**: Простий API для створення складних multi-agent систем
- **Рольова модель**: Кожен агент має чітку роль, ціль та backstory
- **Автоматична координація**: Вбудовані процеси для sequential та hierarchical виконання
- **Інтеграція з інструментами**: Підтримка LangChain tools та власної бібліотеки інструментів
- **Memory capabilities**: Персистентна пам'ять між сесіями
- **Conversational crews**: Природна взаємодія з користувачем

## Структура прикладів

```
agents_v2/
├── 01_basic_crew.py                # Базовий sequential crew
├── 02_hierarchical_crew.py         # Hierarchical process з менеджером
├── 03_research_crew_with_tools.py  # Інтеграція з tools
├── 04_memory_enabled_crew.py       # Memory-enabled conversational crew
├── requirements.txt                # Залежності
├── .env.example                    # Приклад конфігурації
└── README.md                       # Ця документація
```

## Основні концепції

### 1. Agent (Агент)

Агент - це AI-асистент з конкретною роллю та експертизою:

```python
from crewai import Agent

researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments about {topic}",
    backstory=(
        "You are a seasoned research analyst with a keen eye for detail. "
        "Your expertise lies in diving deep into topics..."
    ),
    tools=[search_tool, file_tool],  # Опціонально
    verbose=True,
    allow_delegation=False,
    llm="gpt-4o-mini"
)
```

**Параметри:**
- `role`: Роль агента в команді
- `goal`: Що агент намагається досягти (може містити {змінні})
- `backstory`: Контекст та експертиза агента
- `tools`: Список інструментів, доступних агенту
- `verbose`: Детальне логування
- `allow_delegation`: Чи може делегувати задачі іншим
- `llm`: Модель для використання

### 2. Task (Задача)

Задача визначає конкретну роботу для агента:

```python
from crewai import Task

research_task = Task(
    description=(
        "Conduct comprehensive research on {topic}. "
        "Your analysis should include key features, use cases, "
        "and best practices."
    ),
    expected_output=(
        "A detailed research report with 10-15 bullet points "
        "covering all aspects of {topic}"
    ),
    agent=researcher
)
```

**Параметри:**
- `description`: Детальний опис що потрібно зробити
- `expected_output`: Формат очікуваного результату
- `agent`: Хто відповідальний за виконання
- Змінні в `{дужках}` підставляються при kickoff()

### 3. Crew (Команда)

Crew об'єднує агентів та задачі в координовану систему:

```python
from crewai import Crew, Process

crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,  # або Process.hierarchical
    verbose=True,
    memory=True  # Опціонально: включає пам'ять
)

# Запуск команди
result = crew.kickoff(inputs={"topic": "AI Agents"})
```

**Параметри:**
- `agents`: Список агентів у команді
- `tasks`: Список задач для виконання
- `process`: Sequential (послідовно) або Hierarchical (з менеджером)
- `verbose`: Детальне логування
- `memory`: Включити персистентну пам'ять
- `manager_llm`: Модель для менеджера (тільки hierarchical)

## Процеси виконання

### Sequential Process (Послідовний)

Задачі виконуються одна за одною в порядку визначення:

```
Task 1 → Task 2 → Task 3 → Result
```

**Переваги:**
- Простий та передбачуваний
- Кожна задача отримує результат попередньої
- Легко дебажити

**Використовуйте коли:**
- Лінійний workflow
- Чіткі залежності між задачами
- Невелика команда (2-4 агенти)

### Hierarchical Process (Ієрархічний)

Менеджер автоматично координує виконання задач:

```
         Manager
            |
    +-------+-------+
    |       |       |
 Agent1  Agent2  Agent3
```

**Переваги:**
- Інтелектуальна делегація задач
- Валідація результатів менеджером
- Адаптивне виконання

**Використовуйте коли:**
- Складний проект з багатьма спеціалістами
- Потрібна валідація якості
- Динамічні залежності між задачами
- Велика команда (5+ агентів)

## Інструменти (Tools)

CrewAI підтримує інструменти з різних джерел:

### CrewAI Tools

```python
from crewai_tools import (
    FileReadTool,
    DirectoryReadTool,
    WebsiteSearchTool,
    ScrapeWebsiteTool
)

file_tool = FileReadTool()
agent = Agent(role="...", tools=[file_tool])
```

**Доступні інструменти:**
- `FileReadTool` - читання файлів
- `DirectoryReadTool` - перегляд директорій
- `WebsiteSearchTool` - пошук на веб-сайтах
- `ScrapeWebsiteTool` - скрапінг контенту
- `CodeDocsSearchTool` - пошук у документації коду
- І більше 20 інших...

### LangChain Tools

```python
from langchain.tools import Tool

def my_function(input: str) -> str:
    return f"Processed: {input}"

custom_tool = Tool(
    name="MyTool",
    func=my_function,
    description="What this tool does"
)

agent = Agent(role="...", tools=[custom_tool])
```

### Створення власних інструментів

```python
from crewai_tools import BaseTool

class MyCustomTool(BaseTool):
    name: str = "Custom Tool"
    description: str = "Does something specific"

    def _run(self, argument: str) -> str:
        # Your implementation
        return result
```

## Memory (Пам'ять)

CrewAI підтримує різні типи пам'яті:

```python
crew = Crew(
    agents=[...],
    tasks=[...],
    memory=True,
    embedding_model={
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small"
        }
    }
)
```

**Типи пам'яті:**
- **Short-term**: Контекст поточної розмови
- **Long-term**: Персистентна між сесіями
- **Entity**: Інформація про конкретні сутності
- **Contextual**: Контекст задач та агентів

**Переваги:**
- Персоналізовані відповіді
- Навчання з часом
- Безперервні багатокрокові розмови
- Запам'ятовування переваг користувача

## Приклади використання

### 01_basic_crew.py - Базовий Sequential Crew

**Що демонструє:**
- Створення агентів з ролями та цілями
- Визначення задач з параметрами
- Sequential процес виконання
- Параметризація через inputs

**Команда:**
- Researcher: збір інформації
- Writer: створення контенту
- Editor: редагування та поліпшення

**Запуск:**
```bash
python 01_basic_crew.py
```

### 02_hierarchical_crew.py - Ієрархічна команда

**Що демонструє:**
- Hierarchical процес з автоматичним менеджером
- Координація великої команди (5 агентів)
- Делегація задач та валідація
- Складний проектний workflow

**Команда:**
- Manager (авто-створений): координація
- Requirements Analyst: аналіз вимог
- Software Architect: дизайн архітектури
- Backend Developer: backend розробка
- Frontend Developer: frontend розробка
- QA Engineer: тестування

**Запуск:**
```bash
python 02_hierarchical_crew.py
```

### 03_research_crew_with_tools.py - Інтеграція з інструментами

**Що демонструє:**
- Використання CrewAI tools (FileRead, DirectoryRead)
- Створення custom tools (DataAnalyzer, Calculator)
- Інтеграція LangChain tools
- Розподіл інструментів між агентами

**Команда:**
- Data Researcher: збір інформації з інструментами
- Data Analyst: аналіз з Calculator та DataAnalyzer
- Report Writer: компіляція звіту

**Інструменти:**
- FileReadTool
- DirectoryReadTool
- Custom DataAnalyzer
- Custom Calculator

**Запуск:**
```bash
python 03_research_crew_with_tools.py
```

### 04_memory_enabled_crew.py - Пам'ять та навчання

**Що демонструє:**
- Включення персистентної пам'яті
- Multi-turn conversations
- Навчання з попередніх взаємодій
- Персоналізовані відповіді

**Команда:**
- Personal Assistant: персоналізована допомога
- Knowledge Curator: організація знань
- Context Analyzer: аналіз контексту

**Можливості:**
- Запам'ятовування переваг
- Контекст з попередніх розмов
- Застосування вивченого
- Безперервна розмова

**Запуск:**
```bash
python 04_memory_enabled_crew.py
```

## Налаштування та встановлення

### Вимоги

- Python >= 3.10, < 3.14
- OpenAI API ключ

### Встановлення

1. Створіть віртуальне середовище (рекомендовано):
```bash
# Створити venv
python3 -m venv venv

# Активувати
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
```

**ВАЖЛИВО:** Використовуйте Python 3.11 (не 3.14, є проблеми сумісності).

2. Встановіть залежності:
```bash
pip install -r requirements.txt
```

3. Створіть `.env` файл:
```bash
cp .env.example .env
```

3. Відредагуйте `.env`:
```bash
# OpenAI API Key
OPENAI_API_KEY=sk-your-openai-api-key-here

# LangChain Tracing (Optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your-langsmith-api-key-here
LANGCHAIN_PROJECT=crewai-agents

# CrewAI Configuration
CREWAI_TELEMETRY=false
```

### Залежності

```
crewai>=1.4.0           # Основний фреймворк
crewai-tools>=0.38.0    # Бібліотека інструментів
langchain>=0.3.0        # LangChain integration
langchain-openai        # OpenAI інтеграція
openai>=1.50.0          # OpenAI SDK
pydantic>=2.0.0         # Валідація даних
python-dotenv           # Змінні середовища
langmem>=0.1.0          # Advanced memory (опціонально)
```

## Best Practices

### Дизайн агентів

1. **Чіткі ролі**: Кожен агент повинен мати специфічну експертизу
2. **Детальні backstory**: Контекст покращує якість відповідей
3. **Фокусовані цілі**: Уникайте занадто широких відповідальностей
4. **Правильні інструменти**: Давайте агентам тільки потрібні інструменти

### Дизайн задач

1. **Специфічні описи**: Детально що потрібно зробити
2. **Чіткі очікування**: Формат та зміст результату
3. **Параметризація**: Використовуйте {змінні} для гнучкості
4. **Критерії якості**: Вкажіть стандарти

### Вибір процесу

**Sequential коли:**
- Прямий лінійний workflow
- Менше 5 агентів
- Передбачувані залежності
- Важлива простота

**Hierarchical коли:**
- Складний проект
- 5+ агентів
- Потрібна валідація
- Динамічні рішення

### Продуктивність

1. **Вибір моделі**: gpt-4o-mini для швидкості, gpt-4 для якості
2. **verbose=False** у продакшені для швидкості
3. **Обмеження інструментів**: Тільки необхідні tools
4. **Моніторинг токенів**: Стежте за використанням

## Порівняння з іншими фреймворками

### CrewAI vs LangChain

**CrewAI:**
- Фокус на командній роботі агентів
- Вбудована рольова модель
- Простіший для multi-agent систем
- Автоматична координація

**LangChain:**
- Більше контролю над деталями
- Ширший набір інструментів
- Краща інтеграція з екосистемою
- Більш гнучка архітектура

### CrewAI vs LangGraph

**CrewAI:**
- Вищий рівень абстракції
- Швидша розробка
- Менше коду для простих кейсів

**LangGraph:**
- Детальний контроль над flow
- Складні графи станів
- Краще для нестандартних паттернів

## Troubleshooting

### Агент не використовує інструменти

**Проблема:** Агент не викликає доступні tools

**Рішення:**
1. Переконайтеся, що `tools` передані агенту
2. Додайте чіткий опис інструменту
3. Згадайте інструмент у task description
4. Перевірте, чи потрібен інструмент для задачі

### Hierarchical process не працює

**Проблема:** Помилки при hierarchical process

**Рішення:**
1. Встановіть `manager_llm` параметр
2. Переконайтеся, що агенти мають `allow_delegation=False`
3. Менеджер НЕ повинен бути в списку agents
4. Використовуйте достатньо потужну модель для менеджера

### Memory не зберігається

**Проблема:** Контекст не переноситься між викликами

**Рішення:**
1. Перевірте `memory=True` в Crew
2. Налаштуйте `embedding_model` конфігурацію
3. Переконайтеся, що OPENAI_API_KEY доступний
4. Перевірте версію CrewAI (>=1.4.0)

### Високе використання токенів

**Проблема:** Швидке витрачання токенів

**Рішення:**
1. Використовуйте gpt-4o-mini замість gpt-4
2. Скоротіть backstory агентів
3. Спростіть task descriptions
4. Вимкніть `verbose` у продакшені
5. Обмежте кількість інструментів

## Приклади реальних use cases

### 1. Content Creation Pipeline
```python
# Researcher → Writer → Editor → SEO Specialist
# Sequential process для створення blog posts
```

### 2. Software Development Team
```python
# Manager координує:
# Analyst → Architect → Developer → Tester
# Hierarchical process для розробки фіч
```

### 3. Customer Support System
```python
# Classifier → Specialist → QA
# З memory для персоналізації
# Sequential process з інструментами
```

### 4. Research & Analysis
```python
# Data Collector → Analyst → Report Writer
# З tools для web scraping та аналізу
# Sequential process
```

## Ресурси

### Офіційна документація
- [CrewAI Docs](https://docs.crewai.com/)
- [CrewAI GitHub](https://github.com/crewAIInc/crewAI)
- [CrewAI Examples](https://github.com/crewAIInc/crewAI-examples)

### Спільнота
- [Discord](https://discord.com/invite/X4JWnZnxPb)
- [Twitter](https://twitter.com/crewAIInc)

### Додаткові матеріали
- [LangChain Tools](https://python.langchain.com/docs/integrations/tools/)
- [OpenAI API](https://platform.openai.com/docs/api-reference)

## Ліцензія

MIT License

## Автор

sanyaden <alex.denysyuk@gmail.com>
