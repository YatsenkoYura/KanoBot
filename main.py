import sqlite3
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import json
from datetime import datetime
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_openai import ChatOpenAI
from langchain_classic.chains import LLMChain
from langchain_core.prompts.prompt import PromptTemplate
from langchain_experimental.agents import create_pandas_dataframe_agent
import chromadb
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import re
import statistics
import numpy as np
import uuid
import os

# Конфигурация
API_KEY = "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"
DB_PATH = "database_novec.db"
CHROMA_DB_PATH = "./chroma_db"
MODEL_NAME = "Qwen/Qwen3-Next-80B-A3B-Instruct"
EMBEDDING_MODEL = "BAAI/bge-m3"

# Глобальные состояния системы
_SYSTEM_COMPONENTS: Optional[Dict[str, Any]] = None

# ==================== МЕНЕДЖЕР ИСТОРИИ ====================

class ChatHistoryManager:
    """Менеджер истории чата: последние 5 сообщений от пользователя и 5 от ассистента"""
    
    def __init__(self, max_user_messages: int = 5, max_assistant_messages: int = 5):
        self.max_user_messages = max_user_messages
        self.max_assistant_messages = max_assistant_messages
        self.histories = {}  # chat_id -> {"user": [], "assistant": []}
    
    def get_history(self, chat_id: str) -> List[Dict]:
        """Возвращает объединённую историю (чередование сообщений)"""
        if chat_id not in self.histories:
            return []
        
        user_msgs = self.histories[chat_id]["user"]
        assistant_msgs = self.histories[chat_id]["assistant"]
        combined = []
        for u, a in zip(user_msgs, assistant_msgs):
            combined.append(u)
            combined.append(a)
        return combined
    
    def add_message(self, chat_id: str, role: str, content: str):
        """Добавляет сообщение в истории"""
        if chat_id not in self.histories:
            self.histories[chat_id] = {"user": [], "assistant": []}
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        if role == "user":
            self.histories[chat_id]["user"].append(message)
            if len(self.histories[chat_id]["user"]) > self.max_user_messages:
                self.histories[chat_id]["user"] = self.histories[chat_id]["user"][-self.max_user_messages:]
        else:
            self.histories[chat_id]["assistant"].append(message)
            if len(self.histories[chat_id]["assistant"]) > self.max_assistant_messages:
                self.histories[chat_id]["assistant"] = self.histories[chat_id]["assistant"][-self.max_assistant_messages:]
    
    def format_history_for_prompt(self, chat_id: str) -> str:
        """Форматирует историю для промпта"""
        combined = self.get_history(chat_id)
        if not combined:
            return ""
        formatted = []
        for msg in combined:
            role = "Пользователь" if msg["role"] == "user" else "Ассистент"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)
    
    def clear_history(self, chat_id: str):
        """Очистка истории чата"""
        if chat_id in self.histories:
            self.histories[chat_id] = {"user": [], "assistant": []}

# Инициализируем менеджер истории
history_manager = ChatHistoryManager()

# ==================== ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ ====================

def initialize_system_once() -> Dict[str, Any]:
    global _SYSTEM_COMPONENTS
    if _SYSTEM_COMPONENTS is not None:
        return _SYSTEM_COMPONENTS
    _SYSTEM_COMPONENTS = initialize_system()
    return _SYSTEM_COMPONENTS

def initialize_system() -> Dict[str, Any]:
    print("🔄 Инициализация системы...")
    df = load_data()
    llm_agent = init_llm_agent()
    # pd_agent больше не нужен
    collection = init_chromadb()
    embedding_model = init_embedding_model()
    print("✅ Система готова к работе")
    return {
        'df': df,
        'llm_agent': llm_agent,
        'collection': collection,
        'embedding_model': embedding_model
    }

def load_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        query = "SELECT * FROM main_data_last"
        df = pd.read_sql(query, conn)
        print(f"📊 Загружено {len(df)} записей")
        return df
    finally:
        conn.close()

def init_llm_agent() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0
    )

def init_pandas_agent(llm_agent: ChatOpenAI, df: pd.DataFrame) -> Any:
    # 1. Собираем "паспорт" данных: типы + примеры значений
    # Это критически важно, чтобы агент понимал, что лежит внутри (например, "Москва" или "г. Москва")
    data_samples = []
    for col in df.columns:
        dtype = df[col].dtype
        # Если значений мало (категория) - показываем все, иначе - первые 3
        if df[col].nunique() < 10:
            vals = df[col].unique().tolist()
        else:
            vals = df[col].dropna().head(3).tolist()
        data_samples.append(f"- Колонка '{col}' (тип {dtype}), примеры: {vals}")
    
    data_info_str = "\n".join(data_samples)

    # 2. Новый системный промпт
    # Мы убираем требование "отвечать одним словом" и разрешаем агенту использовать Python
    prefix = f"""
Ты — аналитик данных на Python. В твоем распоряжении pandas dataframe `df`.

ОПИСАНИЕ ДАННЫХ:
{data_info_str}

ТВОЯ ЗАДАЧА:
1. Проанализировать запрос пользователя.
2. Написать и выполнить код Python (используя `df`), чтобы получить точный ответ.
3. Вернуть ответ на русском языке.

ВАЖНО:
- Если возникает ошибка парсинга (OutputParserException), это значит, ты не соблюдаешь формат ReAct.
- Всегда используй формат:
  Thought: думаю, что делать
  Action: python_repl_ast
  Action Input: df['col'].mean()
  Observation: результат
  Final Answer: ответ текстом
"""

    # 3. Создание агента
    return create_pandas_dataframe_agent(
        llm_agent,
        df,
        verbose=True,  # Полезно видеть в консоли, какой код пишет агент
        allow_dangerous_code=True,
        handle_parsing_errors=True, # Теперь при ошибке агент попытается сам себя исправить
        max_iterations=5, # Ограничиваем число попыток, чтобы не завис
        prefix=prefix
    )

def init_chromadb() -> Any:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_collection("data")

def init_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True, device='cpu')

# ==================== ОБРАБОТКА ЗАПРОСОВ С УЧЕТОМ ИСТОРИИ ====================
def ensemble_with_rag_and_pandas(user_input: str, chat_history: str, 
                               rag_context: Optional[str], 
                               pandas_output: Optional[str]) -> str:
    """
    Ансамбль с Судьей: генерирует варианты и выбирает тот, который лучше всего отражает ФАКТЫ из таблицы.
    """
    # --- 1. Подготовка контекста ---
    sources_text = ""
    # Блок данных (Приоритет 1)
    if pandas_output and "Не удалось" not in pandas_output and "Нет данных" not in pandas_output:
        sources_text += f"### 📊 ДАННЫЕ ИЗ ТАБЛИЦЫ (ФАКТЫ):\n{pandas_output}\n\n"
    else:
        sources_text += "### 📊 ДАННЫЕ ИЗ ТАБЛИЦЫ: [Нет данных]\n\n"
        
    # Блок RAG (Приоритет 2)
    if rag_context:
        sources_text += f"### 📚 СПРАВОЧНИК (КОНТЕКСТ):\n{rag_context}\n"

    # --- 2. Промпт для Генераторов ---
    gen_template = """Ты — медицинский аналитик.
Твоя задача: ответить на вопрос, опираясь ГЛАВНЫМ ОБРАЗОМ на "ДАННЫЕ ИЗ ТАБЛИЦЫ".

ВХОДНЫЕ ДАННЫЕ:
{sources_text}

ИСТОРИЯ: {chat_history}
ВОПРОС: {user_input}

ИНСТРУКЦИЯ:
1. Если в "ДАННЫХ ИЗ ТАБЛИЦЫ" есть ответ — используй его. Это истина.
2. RAG (Справочник) используй только для пояснений. Не позволяй RAG'у противоречить таблице.
3. Будь краток и профессионален.

Ответ:"""

    gen_prompt = PromptTemplate(
        template=gen_template, 
        input_variables=["user_input", "chat_history", "sources_text"]
    )

    # Используем две разные модели, чтобы получить разные формулировки
    generator_models = [
        "google/gemma-3-27b-it",
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    ]
    
    answers = []
    print(f"🧠 Запуск ансамбля ({len(generator_models)} модели)...")
    
    for model_name in generator_models:
        try:
            llm = build_llm(model_name, API_KEY)
            chain = LLMChain(llm=llm, prompt=gen_prompt)
            # Запускаем генерацию
            ans = chain.run(user_input=user_input, chat_history=chat_history, sources_text=sources_text)
            answers.append(ans)
        except Exception as e:
            print(f"⚠️ Ошибка модели {model_name}: {e}")
            continue
    
    if not answers:
        return "Извините, не удалось сформировать ответ на основе данных."

    # Если ответ всего один — возвращаем его (нет смысла судить одного)
    if len(answers) == 1:
        return answers[0]

    # --- 3. СУДЬЯ (JUDGE) ---
    # Судья выбирает лучший ответ, сверяясь с сырыми данными
    
    judge_llm = ChatOpenAI(
        model="Qwen/Qwen3-Next-80B-A3B-Instruct", # Мощная модель для судейства
        openai_api_key=API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0
    )
    
    answers_block = "\n\n".join([f"=== КАНДИДАТ {i+1} ===\n{a}" for i, a in enumerate(answers)])
    
    # Передаем судье сырой вывод pandas, чтобы он проверил галлюцинации
    raw_data_snippet = pandas_output if pandas_output else "Нет данных"

    judge_template = """
Ты — Главный Врач-Аналитик. Твоя задача — выбрать наиболее точный и профессиональный ответ из предложенных кандидатов.

ВОПРОС ПОЛЬЗОВАТЕЛЯ: {user_input}

ИСТИННЫЕ ДАННЫЕ (из таблицы):
{raw_data_snippet}

КАНДИДАТЫ ОТВЕТОВ:
{answers_block}

КРИТЕРИИ ВЫБОРА (по важности):
1. **ТОЧНОСТЬ ДАННЫХ**: Кандидат ОБЯЗАН использовать цифры/факты из "ИСТИННЫХ ДАННЫХ". Если кандидат пишет "нет информации", а в данных она есть — это ПЛОХОЙ кандидат.
2. **ПРИОРИТЕТ ТАБЛИЦЫ**: Если данные таблицы противоречат общим знаниям, побеждает таблица.
3. **ПРОФЕССИОНАЛИЗМ**: Ответ должен звучать врачебно и сухо.

ВЕРНИ ТОЛЬКО ТЕКСТ ЛУЧШЕГО ОТВЕТА (без "Кандидат 1" и комментариев).
"""
    
    judge_prompt = PromptTemplate(
        template=judge_template,
        input_variables=["user_input", "answers_block", "raw_data_snippet"]
    )
    
    print("⚖️ Судья выбирает лучший ответ...")
    final_verdict = LLMChain(llm=judge_llm, prompt=judge_prompt).run(
        user_input=user_input, 
        answers_block=answers_block,
        raw_data_snippet=raw_data_snippet
    )
    
    return final_verdict

def process_user_query(user_input: str, chat_id: str = "default") -> str:
    print(f"\n🎯 Запрос от {chat_id}: {user_input}")
    history_manager.add_message(chat_id, "user", user_input)
    
    # 1. Инициализация
    components = initialize_system_once()
    chat_history = history_manager.format_history_for_prompt(chat_id)
    
    # 2. Маршрутизация (Развилка)
    destination = route_query_with_history(user_input, chat_history, components['llm_agent'])
    print(f"📍 Маршрут выбран: {destination}")
    
    response = ""
    
    # 3. Выполнение логики
    if destination == "plot":
        # Ветка графиков
        response = process_plot_with_pd_agent_analysis_exec(user_input, chat_history, components)
        
    elif destination == "chat":
        # Ветка простой болтовни (LLM без инструментов)
        response = process_llm_with_history(user_input, chat_history, components['llm_agent'])
        
    else: # destination == "analysis"
        # Ветка "Комбайн": RAG + Pandas Agent
        # Это та самая новая функция, которую мы обсудили шагом ранее
        response = process_data_query_with_history(user_input, chat_history, components)
    
    # 4. Сохранение и возврат
    history_manager.add_message(chat_id, "assistant", response)
    return response

# --- Router with 'plot' destination ---
def route_query_with_history(user_input: str, chat_history: str, llm_agent: ChatOpenAI) -> str:
    response_schemas = [
        ResponseSchema(name="destination", description="chat, analysis или plot", type="string")
    ]
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = parser.get_format_instructions()
    
    router_prompt = PromptTemplate(
        template="""
        Твоя задача — классифицировать намерение пользователя для медицинского ассистента.
        
        ИСТОРИЯ:
        {chat_history}
        
        ЗАПРОС: {user_input}

        КАТЕГОРИИ (выбери одну):
        
        1. "plot" — ЕСЛИ пользователь явно просит визуализацию: "построй график", "нарисуй гистограмму", "визуализируй", "plot", "chart".
        
        2. "analysis" — ЕСЛИ пользователь задает ЛЮБОЙ содержательный вопрос по медицине или данным.
           - Вопросы "Сколько...", "Какое среднее...", "Найди записи..." (это статистика).
           - Вопросы "Что такое...", "Какие симптомы...", "Как лечить..." (это база знаний).
           - Если сомневаешься между чатом и анализом — выбирай "analysis".
           
        3. "chat" — ЕСЛИ это просто общение без запроса информации.
           - Приветствия ("Привет", "Здравствуйте").
           - Благодарности ("Спасибо").
           - Вопросы о личности бота ("Кто ты?").
           - Явный оффтоп.

        {format_instructions}
        """,
        input_variables=["user_input", "chat_history"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    router_chain = LLMChain(llm=llm_agent, prompt=router_prompt, output_parser=parser)
    try:
        result = router_chain.run(user_input=user_input, chat_history=chat_history)
        return result['destination']
    except Exception as e:
        print(f"Router error: {e}, fallback to analysis")
        return "analysis" # Если роутер сломался, лучше попытаться найти данные

def process_llm_with_history(user_input: str, chat_history: str, llm_agent: ChatOpenAI) -> str:
    # Мы добавляем "Persona" и "Guardrails" в промпт
    llm_prompt = PromptTemplate(
        template="""Ты — профессиональный медицинский ассистент.
Твоя задача — помогать пользователю только в вопросах медицины, здравоохранения, лекарственных препаратов и анализа медицинских данных.

ИСТОРИЯ РАЗГОВОРА:
{chat_history}

ТЕКУЩИЙ ЗАПРОС: {user_input}

ПРАВИЛА ОТВЕТА:
1. Если вопрос касается медицины, здоровья, препаратов или анализа данных — дай подробный и профессиональный ответ.
2. Если вопрос касается приветствия (привет, здравствуйте) — поздоровайся и предложи помощь по медицинским вопросам.
3. Если вопрос НЕ СВЯЗАН с медициной (политика, спорт, кино, погода, рецепты еды, программирование не по теме и т.д.) — вежливо откажись отвечать.
   Пример отказа: "Извините, но я специализируюсь только на медицинских вопросах. Я не могу поддержать разговор на эту тему."

Отвечай, строго следуя этим правилам.""",
        input_variables=["user_input", "chat_history"]
    )
    llm_chain = LLMChain(llm=llm_agent, prompt=llm_prompt)
    return llm_chain.run(user_input=user_input, chat_history=chat_history)

def process_data_query_with_history(user_input: str, chat_history: str, components: Dict[str, Any]) -> str:
    print("🔄 Запуск гибридного поиска (RAG + Pandas)...")
    
    # 1. Получаем контекст из RAG (База знаний)
    enhanced_query = f"{chat_history}\nТекущий запрос: {user_input}"
    rag_context, rag_docs = perform_rag_search(enhanced_query, components)
    
    # Проверяем полезность RAG (опционально, можно оставить просто текст)
    is_rag_useful = False
    if rag_context:
        is_rag_useful = is_rag_good_with_history(rag_context, user_input, chat_history, components['llm_agent'])
        if not is_rag_useful:
            rag_context = None # Отбрасываем, если мусор
            print("❌ RAG найден, но признан бесполезным.")
        else:
            print("✅ RAG контекст принят.")

    # 2. Получаем данные от Pandas Agent (Статистика из БД)
    # Нам нужно вызвать агента, но немного модифицировать вызов, чтобы он не падал с ошибкой,
    # а возвращал "Нет данных", если не справился.
    pandas_output = run_pandas_code_exec(user_input, chat_history, components)
    
    # 3. Отправляем всё в ансамбль для синтеза ответа
    return ensemble_with_rag_and_pandas(user_input, chat_history, rag_context, pandas_output)

def run_pandas_code_exec(user_input: str, chat_history: str, components: Dict[str, Any]) -> Optional[str]:
    """
    Аналог Plot-функции, но для получения ТЕКСТА/ЧИСЕЛ, а не картинки.
    Генерирует код -> Выполняет -> Возвращает значение переменной `final_answer`.
    """
    print("📊 Генерация кода для анализа данных...")
    
    df = components['df']
    llm_agent = components['llm_agent']
    
    # 1. Описание колонок (как в Plot)
    columns_info = []
    for col in df.columns:
        try:
            vals = df[col].dropna().head(3).tolist()
            dtype = df[col].dtype
            columns_info.append(f"- {col} ({dtype}), примеры: {vals}")
        except:
            columns_info.append(f"- {col} (unknown)")
    columns_str = "\n".join(columns_info)

    # 2. Промпт для генерации кода
    prompt_template = """
Ты — Python Data Analyst. Твоя задача — написать код для ответа на вопрос пользователя.

ДАННЫЕ (переменная `df`):
{columns_str}

ИСТОРИЯ ЧАТА:
{chat_history}

ВОПРОС: {user_input}

ТРЕБОВАНИЯ К КОДУ:
1. Напиши Python код, который анализирует `df`.
2. Сохрани итоговый ответ (строку или число) в переменную `final_answer`.
   - Если считаешь количество/среднее: `final_answer = str(df[...].mean())`
   - Если ищешь список: `final_answer = ", ".join(df[...].unique())`
   - Если данных нет (пустой результат): `final_answer = "В базе данных нет записей по этому запросу"`
3. НЕ используй `print()`. Только присвой значение переменной `final_answer`.
4. Оберни код в try-except, если нужно.

ПРИМЕР:
# Вопрос: Сколько записей с возрастом > 50?
count = df[df['age'] > 50].shape[0]
final_answer = f"Найдено {{count}} записей."

Пиши ТОЛЬКО код.
"""
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["columns_str", "chat_history", "user_input"]
    )

    try:
        # Генерация кода
        chain = LLMChain(llm=llm_agent, prompt=prompt)
        gen_code = chain.run(
            columns_str=columns_str, 
            chat_history=chat_history, 
            user_input=user_input
        )
        
        # Очистка кода от markdown
        cleaned_code = gen_code.replace("```python", "").replace("```", "").strip()
        
        # Выполнение
        local_scope = {"pd": pd, "np": np, "df": df, "final_answer": "Ошибка вычисления"}
        
        try:
            exec(cleaned_code, {}, local_scope)
            result = str(local_scope.get("final_answer", "Код выполнился, но переменная final_answer не найдена."))
            print(f"✅ Результат анализа: {result}")
            return result
        except Exception as e:
            print(f"⚠️ Ошибка выполнения кода анализа: {e}")
            return f"Ошибка при расчетах: {e}"
            
    except Exception as e:
        print(f"❌ Ошибка LLM: {e}")
        return None

def perform_rag_search(user_input: str, components: Dict[str, Any]) -> Tuple[str, List[str]]:
    query_embedding = components['embedding_model'].encode([user_input], normalize_embeddings=True)[0]
    results = components['collection'].query(
        query_embeddings=[query_embedding.tolist()],
        n_results=6,
        include=["documents", "distances"]
    )
    rag_docs = results["documents"][0]
    context = "\n\n".join(rag_docs)
    if rag_docs:
        print(f"📄 Найдено {len(rag_docs)} документов")
    return context, rag_docs

def is_rag_good_with_history(context: str, user_input: str, chat_history: str, llm_agent: ChatOpenAI) -> bool:
    check_prompt = PromptTemplate(
        template="""
Учитывай историю разговора:
{chat_history}
Контекст из базы знаний:
{context}
Текущий вопрос пользователя: {user_input}
Ответь одним словом:
- "yes" — если контекст содержит полезную информацию;
- "no" — если контекст бесполезен.""",
        input_variables=["context", "user_input", "chat_history"]
    )
    check_chain = LLMChain(llm=llm_agent, prompt=check_prompt, verbose=False)
    check_result = check_chain.run(context=context, user_input=user_input, chat_history=chat_history).strip().lower()
    return check_result == "yes"

def process_pandas_agent_with_history(user_input: str, chat_history: str, components: Dict[str, Any]) -> str:
    print("📊 Используем pandas agent для анализа данных с учетом истории...")
    try:
        df = components['df']
        columns_list = ", ".join(df.columns.tolist())
        dtypes_info = ", ".join([f"{col}: {dtype}" for col, dtype in df.dtypes.items()])
        
        pd_raw = components['pd_agent'].run({
            "input": user_input, 
            "chat_history": chat_history,
            "columns_list": columns_list,
            "dtypes_info": dtypes_info
        })
        if not pd_raw:
            return "Не удалось получить данные. Попробуйте уточнить запрос."
        return ensemble_with_history(pd_raw, chat_history)
    except Exception as e:
        print(f"Ошибка pandas agent: {e}")
        return process_llm_with_history(user_input, chat_history, components['llm_agent'])

# ==================== НОВАЯ ВЕТКА: plot ====================

def get_plot_config(df, user_input, chat_history, llm_agent) -> Dict[str, Any]:
    """
    Определяет конфигурацию графика. Поддерживает как простые (1 колонка), так и сложные (фильтры + время) запросы.
    """
    columns_list = ", ".join(df.columns.tolist())
    # Берем только первые 5 значений для примеров типов, чтобы не засорять промпт
    dtypes_info = []
    for col, dtype in df.dtypes.items():
        example_vals = df[col].dropna().head(3).tolist()
        dtypes_info.append(f"{col} ({dtype}): примеры {example_vals}")
    dtypes_str = "\n".join(dtypes_info)

    prompt = PromptTemplate(
        template="""
Ты - Data Scientist. Твоя задача - перевести запрос пользователя в настройки для matplotlib.

ДАННЫЕ:
{dtypes_str}

ИСТОРИЯ:
{chat_history}

ЗАПРОС: {user_input}

Верни JSON с настройками. Поля могут быть null, если не нужны.

СТРУКТУРА JSON:
1. "plot_type": "hist" (гистограмма), "bar" (столбцы), "line" (линия/время), "scatter" (точки), "box" (ящик).
2. "x_col": Главная колонка.
   - Для гистограммы/box: колонка, распределение которой смотрим (например, "стоимость").
   - Для временного графика: колонка с датой.
   - Для барчарта: колонка категорий.
3. "y_col": Вторая колонка (опционально).
   - Для гистограммы/box: ВСЕГДА null.
   - Для scatter: вторая числовая колонка.
   - Для временного графика/барчарта:
     * Если нужно просто посчитать количество строк (например, "динамика обращений"), пиши "COUNT".
     * Если нужно значение (например, "сумма продаж по датам"), пиши имя колонки значений.
4. "filter_condition": Pandas query string (например, "`диагноз` == 'ОРВИ'"). Если фильтра нет - null.

ПРИМЕРЫ:
- "Гистограмма стоимости": {{"plot_type": "hist", "x_col": "стоимость", "y_col": null, "filter_condition": null}}
- "График ОРВИ по дням": {{"plot_type": "line", "x_col": "дата", "y_col": "COUNT", "filter_condition": "`диагноз` == 'ОРВИ'"}}
- "Связь цены и количества": {{"plot_type": "scatter", "x_col": "цена", "y_col": "количество", "filter_condition": null}}

ВАЖНО: Верни ТОЛЬКО JSON.
""",
        input_variables=["user_input", "chat_history", "dtypes_str"]
    )

    chain = LLMChain(llm=llm_agent, prompt=prompt)
    
    try:
        response = chain.run({
            "user_input": user_input,
            "chat_history": chat_history,
            "dtypes_str": dtypes_str
        })
        cleaned_json = response.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_json)
    except Exception as e:
        print(f"JSON Error: {e}")
        # Fallback на безопасную гистограмму первой числовой колонки
        num_cols = df.select_dtypes(include='number').columns
        col = num_cols[0] if len(num_cols) > 0 else df.columns[0]
        return {"plot_type": "hist", "x_col": col, "y_col": None, "filter_condition": None}

def process_plot_with_pd_agent_analysis_exec(user_input: str, chat_history: str, components: Dict[str, Any]) -> str:
    import matplotlib
    matplotlib.use('Agg') 
    import seaborn as sns


    df = components['df']
    llm_agent = components['llm_agent']
    
    # 1. Подготовка
    plot_filename = f"plot.png"
    
    columns_info = []
    for col in df.columns:
        try:
            vals = df[col].dropna().head(3).tolist()
            dtype = df[col].dtype
            columns_info.append(f"- {col} ({dtype}), примеры: {vals}")
        except:
            columns_info.append(f"- {col} (unknown)")
    columns_str = "\n".join(columns_info)

    # 2. ИСПРАВЛЕННЫЙ ПРОМПТ
    code_prompt = """
Ты - Python Data Scientist. Напиши код для построения графика.

ДАННЫЕ (df):
{columns_str}

ЗАПРОС: {user_input}

АЛГОРИТМ РАБОТЫ (СТРОГО СЛЕДУЙ ЕМУ):

1. Подготовь данные:
   `plot_data = df['...']` (или groupby).
   `chosen_col = '...'`

2. Настрой флаг обрезки `is_truncated`:
   - Инициализируй `is_truncated = False`.
   - ВАЖНО: Если (и ТОЛЬКО если) ты собираешься строить **SCATTER PLOT** (точечную диаграмму) и `len(plot_data) > 20000`:
       Выполни: `plot_data = plot_data.sample(n=20000, random_state=42)`
       Выполни: `is_truncated = True`
   - Для ГИСТОГРАММ (hist), ЯЩИКОВ (boxplot), СТОЛБЦОВ (bar) обрезать данные ЗАПРЕЩЕНО. Оставь `is_truncated = False`.

3. Построй график:
   - Используй `plot_data`.
   - Добавь `plt.title(...)`, `plt.grid(True, alpha=0.3)`.

4. Сохрани результат:
   `plt.savefig('{plot_filename}', bbox_inches='tight')`

Пример правильной логики (ПСЕВДОКОД):
# Если гистограмма:
# is_truncated = False
# plt.hist(plot_data)

# Если scatter:
# if len(plot_data) > 20000:
#    plot_data = plot_data.sample(20000)
#    is_truncated = True
# else:
#    is_truncated = False
# plt.scatter(...)

Пиши ТОЛЬКО валидный Python код.
"""
    prompt = PromptTemplate(template=code_prompt, input_variables=["columns_str", "user_input", "plot_filename"])
    
    try:
        # --- ЭТАП 1: Генерация графика ---
        gen_code = LLMChain(llm=llm_agent, prompt=prompt).run(
            columns_str=columns_str, user_input=user_input, plot_filename=plot_filename
        )
        cleaned_code = gen_code.replace("```python", "").replace("```", "").strip()
        
        # is_truncated = False по умолчанию, чтобы код не падал
        local_scope = {"pd": pd, "np": np, "plt": plt, "sns": sns, "df": df.copy(), "is_truncated": False}
        plt.clf() 
        plt.figure(figsize=(10, 6))
        
        try:
            exec(cleaned_code, {}, local_scope)
        except Exception as exec_err:
            return f"❌ Ошибка графика:\n`{exec_err}`\nКод:\n{cleaned_code}"
        
        if not os.path.exists(plot_filename):
            return f"⚠️ График не создан. Код:\n{cleaned_code}"

        # --- ЭТАП 2: Сбор статистики ---
        analyze_data = local_scope.get("plot_data", df) 
        analyze_col = local_scope.get("chosen_col", None)
        is_truncated = local_scope.get("is_truncated", False)

        analysis_scope = {"df": analyze_data, "chosen_col": analyze_col, "pd": pd, "np": np}

        # Скрипт сбора сухих фактов
        stats_code = """
try:
    col_data = None
    stat_name = str(chosen_col) if chosen_col else "Unknown"

    if isinstance(df, pd.Series):
        col_data = df
    elif isinstance(df, pd.DataFrame):
        if chosen_col and chosen_col in df.columns:
            col_data = df[chosen_col]
        else:
            nums = df.select_dtypes(include='number')
            col_data = nums.iloc[:, 0] if not nums.empty else df.iloc[:, 0]
    elif isinstance(df, list) or np.isscalar(df):
        col_data = pd.Series(df)

    numeric_data = pd.to_numeric(col_data, errors='coerce').dropna()
    valid_len = col_data.count()
    
    stats = {}
    if not numeric_data.empty:
        desc = numeric_data.describe()
        stats = {
            'mean': float(desc['mean']),
            'median': float(desc['50%']),
            'min': float(desc['min']),
            'max': float(desc['max']),
            'std': float(desc['std']),
            'sum': float(numeric_data.sum())
        }

    top_vals = col_data.value_counts().head(5).to_dict()
    top_vals_clean = {str(k): int(v) for k, v in top_vals.items()}

    result = {
        'column': stat_name,
        'count': int(valid_len),
        'numeric_stats': stats,
        'top_values': top_vals_clean
    }
except Exception as e:
    result = {'error': str(e)}
"""
        exec(stats_code, {}, analysis_scope)
        stats_result = analysis_scope.get("result", {})

        # --- ЭТАП 3: ФИНАЛЬНЫЙ ОТВЕТ ---
        
        truncation_msg = "ДА (данные ограничены 20,000 точек для производительности)" if is_truncated else "НЕТ"

        final_prompt = f"""
Пользователь задал вопрос: "{user_input}".
График построен. 
Была ли произведена обрезка данных (is_truncated)? {truncation_msg}.

Статистика:
{json.dumps(stats_result, ensure_ascii=False, indent=2)}

Задача:
1. Опиши статистику (цифры).
2. Тон: сухой, аналитический.
3. ВАЖНО: Если обрезка была ("ДА") -> Добавь в конце предупреждение: "⚠️ Данные на графике были ограничены выборкой (20 000 точек) для оптимизации отображения."
4. Если обрезки не было ("НЕТ") -> Никаких предупреждений не пиши.
"""
        final_answer = llm_agent.invoke(final_prompt).content

        return f"{final_answer}\n\n📊 **График:** `{plot_filename}`"

    except Exception as e:
        return f"❌ Ошибка: {e}"
    finally:
        plt.close('all')
        
def ensemble_with_rag_and_history(user_input: str, chat_history: str, context: str) -> str:
    return ensemble_answer_with_context_and_history(API_KEY, user_input, chat_history, context)

def ensemble_with_history(user_input: str, chat_history: str) -> str:
    return ensemble_answer_with_context_and_history(API_KEY, user_input, chat_history, context=None)

def ensemble_answer_with_context_and_history(api_key: str, user_input: str, 
                                           chat_history: str, context: Optional[str] = None) -> str:
    generator_models = [
        "google/gemma-3-27b-it",
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "openai/gpt-oss-20b"
    ]
    
    # Общая инструкция для всех моделей ансамбля
    base_instruction = """Ты — строгий медицинский консультант. 
Ты ОБЯЗАН отвечать ТОЛЬКО на вопросы, связанные с медициной и здоровьем.
Если тема вопроса отвлеченная (игры, политика, развлечения) — откажись отвечать.
"""

    if context:
        template = base_instruction + """
Используй этот контекст из базы знаний (если он полезен):
{context}

История разговора:
{chat_history}

Вопрос пользователя: {user_input}
Ответ:"""
        input_vars = ["user_input", "chat_history", "context"]
    else:
        template = base_instruction + """
История разговора:
{chat_history}

Вопрос пользователя: {user_input}
Ответ:"""
        input_vars = ["user_input", "chat_history"]
        
    gen_prompt = PromptTemplate(template=template, input_variables=input_vars)
    
    answers = []
    # Запрос ко всем моделям
    for model_name in generator_models:
        # ВАЖНО: создаем новый LLM instance для каждого вызова, чтобы избежать конфликтов
        llm = build_llm(model_name, api_key)
        chain = LLMChain(llm=llm, prompt=gen_prompt)
        try:
            if context:
                ans = chain.run(user_input=user_input, chat_history=chat_history, context=context)
            else:
                ans = chain.run(user_input=user_input, chat_history=chat_history)
            answers.append(ans)
        except Exception as e:
            print(f"Ошибка модели {model_name}: {e}")
            # Если модель упала, продолжаем без нее
            continue
            
    if not answers:
        return "Извините, произошла ошибка генерации ответа."

    # Судья выбирает лучший ответ
    judge_llm = ChatOpenAI(
        model="Qwen/Qwen3-Next-80B-A3B-Instruct",
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0
    )
    
    answers_block = "\n\n".join([f"Ответ {i+1}:\n{a}" for i, a in enumerate(answers)])
    
    judge_prompt = PromptTemplate(
        template="""
Ты — главный врач, проверяющий ответы ассистентов.

Критерии отбора:
1. БЕЗОПАСНОСТЬ: Ответ должен быть только про медицину. Если ассистент начал отвечать про политику или игры — это плохой ответ.
2. ТОЧНОСТЬ: Медицинские факты должны быть верными.
3. КОНТЕКСТ: Ответ должен учитывать историю разговора.

Вопрос пользователя:
{user_input}

Кандидаты ответов:
{answers_block}

Выбери ОДИН лучший ответ, который соответствует роли медицинского ассистента. 
Верни ТОЛЬКО текст выбранного ответа без "Ответ 1:" и без своих комментариев.
""",
        input_variables=["user_input", "answers_block"]
    )
    judge_chain = LLMChain(llm=judge_llm, prompt=judge_prompt)
    
    # В судью историю можно не передавать целиком, достаточно вопроса и ответов, чтобы сэкономить токены,
    # но если контекст важен, можно добавить. Здесь упростим для надежности.
    return judge_chain.run(user_input=user_input, answers_block=answers_block)

def build_llm(model_name: str, api_key: str, temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=temperature
    )

# ==================== УТИЛИТЫ ДЛЯ ИСТОРИИ ====================

def clear_chat_history(chat_id: str = "default"):
    history_manager.clear_history(chat_id)
    return f"История чата {chat_id} очищена"

def get_chat_history(chat_id: str = "default") -> List[Dict]:
    return history_manager.get_history(chat_id)

def export_chat_history(chat_id: str = "default") -> str:
    history = history_manager.get_history(chat_id)
    return json.dumps(history, ensure_ascii=False, indent=2)

# ==================== ОБРАБОТКА TELEGRAM ====================

def handle_telegram_message(message_text: str, chat_id: str) -> str:
    try:
        if message_text.strip().lower() == "/clear":
            return clear_chat_history(chat_id)
        elif message_text.strip().lower() == "/history":
            history = get_chat_history(chat_id)
            if not history:
                return "История чата пуста"
            return f"История ({len(history)} сообщений):\n" + "\n".join(
                [f"{msg['role']}: {msg['content'][:100]}..." for msg in history]
            )
        response = process_user_query(message_text, chat_id)
        return response
    except Exception as e:
        print(f"❌ Ошибка обработки сообщения: {e}")
        return "Произошла ошибка при обработке вашего запроса. Попробуйте еще раз."

# ==================== ТЕСТ ====================

if __name__ == "__main__":
    test_chat_id = "test_user_123"
    queries = [
        "В каком районе больше всего рецептов?",

    ]
    for query in queries:
        print(f"\n{'='*60}")
        print(f"👤 {query}")
        response = handle_telegram_message(query, test_chat_id)
        # выводим часть ответа для проверки
        print(f"🤖 {response[:1000]}...")
    print(f"\n📜 История чата ({len(get_chat_history(test_chat_id))} сообщений)")
    for msg in get_chat_history(test_chat_id):
        role = "👤" if msg['role'] == 'user' else '🤖'
        print(f"{role} {msg['content'][:1000]}...")