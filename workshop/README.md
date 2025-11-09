# 🎓 LangChain/LangGraph Workshop - Complete Package

> **Production-ready materials for 3-4 hour workshop**

Цей пакет містить ВСЕ необхідне для проведення професійного воркшопу з LangChain v1.0 та LangGraph.

---

## 📦 Що включено?

### 1. 📂 Робочий код (Modules)
- **module1_lcel/** - LCEL (LangChain Expression Language)
  - `01_basic_chain.py` - Базові ланцюги
  - `02_parallel_execution.py` - Паралельне виконання
  - `03_streaming.py` - Streaming відповідей

- **module2_agents/** - Агенти та інструменти
  - `01_basic_agent.py` - Створення агентів з tools

- **module3_langgraph/** - LangGraph (TODO)
- **module4_multi_agent/** - Multi-agent системи (TODO)

### 2. 📋 Матеріали для спікера
- `IMPROVED_SPEAKER_NOTES.md` - Покращений гід з порадами
- `handouts/WORKSHOP_CHECKLIST.md` - Чек-лист підготовки

### 3. 📄 Матеріали для учасників
- `handouts/CHEAT_SHEET.md` - Швидка довідка (1-2 сторінки)
- `exercises/` - Практичні вправи
- `solutions/` - Рішення вправ

---

## 🚀 Швидкий старт для спікера

### 1. Підготовка (за тиждень)

```bash
# Клонуйте репозиторій
git clone <repo-url>
cd workshop

# Встановіть залежності
pip install -r requirements.txt

# Створіть .env файл
cp .env.example .env
# Додайте ваші API keys

# Тестуйте всі скрипти
python module1_lcel/01_basic_chain.py
python module1_lcel/02_parallel_execution.py
python module1_lcel/03_streaming.py
python module2_agents/01_basic_agent.py
```

### 2. День воркшопу

1. Прийдіть за 30 хвилин
2. Перевірте проектор і інтернет
3. Налаштуйте IDE (font size 18-20)
4. Відкрийте `WORKSHOP_CHECKLIST.md`
5. Let's go! 🎉

---

## 📚 Структура воркшопу

### Timing (Total: 3-4 години)

```
📍 Intro (15 хв)
   └─ Презентація + Motivation

📦 Module 1: LCEL (45 хв)
   ├─ Базові ланцюги (15 хв)
   ├─ Паралельне виконання (15 хв)
   └─ Streaming (15 хв)

☕ Break (10 хв)

🤖 Module 2: Агенти (45 хв)
   ├─ Створення агентів (20 хв)
   ├─ Custom tools (15 хв)
   └─ Вправа (10 хв)

☕ Break (10 хв)

🔄 Module 3: LangGraph (45 хв)
   ├─ StateGraph (20 хв)
   ├─ Checkpointing (15 хв)
   └─ Cycles (10 хв)

☕ Break (10 хв)

🎭 Module 4: Multi-Agent (45 хв) - OPTIONAL
   ├─ Supervisor pattern (20 хв)
   ├─ Demo (15 хв)
   └─ Discussion (10 хв)

📊 Production Tips (20 хв)
   └─ Performance, Monitoring, Deployment

💻 Practice Time (30 хв)
   └─ Hands-on exercise

❓ Q&A + Wrap-up (15 хв)
```

---

## 🎯 Для учасників

### Prerequisites

**Обов'язково:**
- Python 3.9+
- Базові знання Python
- IDE (VS Code recommended)
- OpenAI API key

**Рекомендовано:**
- Git basics
- Terminal experience
- REST API розуміння

### Setup інструкції

**1. Clone repository:**
```bash
git clone <repo-url>
cd workshop
```

**2. Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# або
venv\Scripts\activate  # Windows
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Setup API keys:**
```bash
cp .env.example .env
# Edit .env and add your keys:
# OPENAI_API_KEY=sk-...
```

**5. Test setup:**
```bash
python module1_lcel/01_basic_chain.py
```

Якщо працює - ви готові! 🎉

---

## 📖 Ресурси для навчання

### Під час воркшопу:
- `CHEAT_SHEET.md` - швидка довідка
- Приклади коду в кожному модулі
- Live coding з спікером

### Після воркшопу:
- [LangChain Docs](https://python.langchain.com/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangChain Academy](https://academy.langchain.com/)
- [Discord Community](https://discord.gg/langchain)

---

## 🛠️ Для спікера: Customization

### Адаптація під вашу аудиторію:

**Junior developers:**
- Більше часу на основи
- Детальніше про Python concepts
- Пропустіть Module 4

**Senior developers:**
- Швидше через basics
- Більше advanced topics
- Deep dive в production concerns

**Short workshop (2 год):**
- Тільки Module 1 + Module 2
- Більше demo, менше exercises
- Дайте homework для Module 3-4

**Long workshop (full day):**
- Додайте real project build
- Більше practice time
- Code review session
- Deployment workshop

---

## 📂 Файлова структура

```
workshop/
├── README.md                          # Цей файл
├── IMPROVED_SPEAKER_NOTES.md          # Детальний гід для спікера
├── requirements.txt                   # Python залежності
├── .env.example                       # Приклад environment variables
│
├── module1_lcel/                      # Module 1: LCEL
│   ├── 01_basic_chain.py             # ✅ Готово
│   ├── 02_parallel_execution.py      # ✅ Готово
│   └── 03_streaming.py               # ✅ Готово
│
├── module2_agents/                    # Module 2: Agents
│   └── 01_basic_agent.py             # ✅ Готово
│
├── module3_langgraph/                 # Module 3: LangGraph
│   └── (TODO)
│
├── module4_multi_agent/               # Module 4: Multi-Agent
│   └── (TODO)
│
├── exercises/                         # Вправи для учасників
│   ├── exercise1_build_chatbot.md
│   ├── exercise2_rag_system.md
│   └── exercise3_agent_team.md
│
├── solutions/                         # Рішення вправ
│   ├── solution1.py
│   ├── solution2.py
│   └── solution3.py
│
└── handouts/                          # Матеріали для роздачі
    ├── CHEAT_SHEET.md                # ✅ Готово
    └── WORKSHOP_CHECKLIST.md         # ✅ Готово
```

---

## ⚡ Quick Commands

```bash
# Запуск конкретного модуля
python module1_lcel/01_basic_chain.py

# Запуск з verbose logging
LANGCHAIN_VERBOSE=true python module1_lcel/01_basic_chain.py

# Перевірка версій
pip list | grep langchain

# Update залежностей
pip install -U langchain langchain-openai langgraph
```

---

## 🐛 Troubleshooting

### "No API key found"
```bash
# Check .env file exists
ls -la .env

# Check it has correct variable
cat .env | grep OPENAI_API_KEY
```

### "Module not found"
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Verify installation
python -c "import langchain; print(langchain.__version__)"
```

### "Rate limit exceeded"
- Використайте backup API key
- Або показуйте pre-recorded demos
- Або додайте time.sleep() між викликами

---

## 📊 Feedback & Improvements

### Після воркшопу:

1. **Collect feedback:**
   - Google Forms survey
   - Quick verbal feedback
   - LinkedIn endorsements

2. **Iterate:**
   - Update materials based on feedback
   - Fix code issues
   - Add more examples

3. **Share:**
   - Recording (if allowed)
   - Updated materials
   - Blog post summary

---

## 📝 License

MIT License - feel free to use and adapt!

---

## 🤝 Contributing

Знайшли баг? Маєте пропозиції?

1. Open issue на GitHub
2. Create pull request
3. Або напишіть [your-email]

---

## ✨ Credits

**Created by:** [Your Name]
**Workshop date:** [Date]
**Version:** 1.0 (LangChain v1.0 compatible)

**Based on:**
- LangChain official docs
- LangGraph documentation
- Real production experience

---

## 🎯 Success Metrics

**Good workshop:**
- 80%+ participants complete exercises
- Positive energy in room
- 5+ questions during session
- 4/5+ average rating

**Great workshop:**
- Participants build something new
- Active discussion
- Follow-up questions after
- People share on social media

---

**Questions? Issues? Improvements?**

Open an issue or reach out at [contact info]

**Good luck with your workshop! 🚀**

---

_Last updated: 2024 | Compatible with LangChain v1.0+_
