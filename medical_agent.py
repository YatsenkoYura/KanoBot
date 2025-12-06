import sys
import os
import sqlite3
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Callable
import json
from datetime import datetime
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_openai import ChatOpenAI
from langchain_classic.chains import LLMChain
from langchain_core.prompts.prompt import PromptTemplate
from langchain_experimental.agents import create_pandas_dataframe_agent
import chromadb
from sentence_transformers import SentenceTransformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import uuid
import base64
from io import BytesIO

# Конфигурация
API_KEY = None
DB_PATH = "database_novec.db"
CHROMA_DB_PATH = "./chroma_db"
MODEL_NAME = "Qwen/Qwen3-Next-80B-A3B-Instruct"
EMBEDDING_MODEL = "BAAI/bge-m3"

# Глобальные состояния системы
_SYSTEM_COMPONENTS: Optional[Dict[str, Any]] = None


class ChatHistoryManager:
    def __init__(self, max_user_messages: int = 5, max_assistant_messages: int = 5):
        self.max_user_messages = max_user_messages
        self.max_assistant_messages = max_assistant_messages
        self.histories = {}
    
    def get_history(self, chat_id: str) -> List[Dict]:
        if chat_id not in self.histories: return []
        user_msgs = self.histories[chat_id]["user"]
        assistant_msgs = self.histories[chat_id]["assistant"]
        combined = []
        for u, a in zip(user_msgs, assistant_msgs):
            combined.append(u)
            combined.append(a)
        return combined
    
    def add_message(self, chat_id: str, role: str, content: str, graph_data: Optional[str] = None):
        if chat_id not in self.histories: self.histories[chat_id] = {"user": [], "assistant": []}
        message = {"role": role, "content": content, "timestamp": datetime.now().isoformat(), "graph_data": graph_data}
        if role == "user":
            self.histories[chat_id]["user"].append(message)
            if len(self.histories[chat_id]["user"]) > self.max_user_messages:
                self.histories[chat_id]["user"] = self.histories[chat_id]["user"][-self.max_user_messages:]
        else:
            self.histories[chat_id]["assistant"].append(message)
            if len(self.histories[chat_id]["assistant"]) > self.max_assistant_messages:
                self.histories[chat_id]["assistant"] = self.histories[chat_id]["assistant"][-self.max_assistant_messages:]
    
    def format_history_for_prompt(self, chat_id: str) -> str:
        combined = self.get_history(chat_id)
        if not combined: return ""
        formatted = []
        for msg in combined:
            role = "Пользователь" if msg["role"] == "user" else "Ассистент"
            content = re.sub(r'\[GRAPH_FILE:.*?\]', '', msg['content']).strip()
            if content: formatted.append(f"{role}: {content}")
        return "\n".join(formatted)
    
    def clear_history(self, chat_id: str):
        if chat_id in self.histories: self.histories[chat_id] = {"user": [], "assistant": []}

history_manager = ChatHistoryManager()


def initialize_system_once(api_key: str) -> Dict[str, Any]:
    global _SYSTEM_COMPONENTS, API_KEY
    API_KEY = api_key
    if _SYSTEM_COMPONENTS is not None: return _SYSTEM_COMPONENTS
    _SYSTEM_COMPONENTS = initialize_system(api_key)
    return _SYSTEM_COMPONENTS

def initialize_system(api_key: str) -> Dict[str, Any]:
    print("🔄 Инициализация медицинской системы...")
    df = load_data()
    llm_agent = init_llm_agent(api_key)
    pd_agent = init_pandas_agent(llm_agent, df) 
    collection = init_chromadb()
    embedding_model = init_embedding_model()
    print("✅ Медицинская система готова к работе")
    return {
        'df': df, 'llm_agent': llm_agent, 'pd_agent': pd_agent,
        'collection': collection, 'embedding_model': embedding_model
    }

def load_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        query = "SELECT * FROM main_data_last"
        df = pd.read_sql(query, conn)
        print(f"📊 Загружено {len(df)} записей")
        return df
    finally: conn.close()

def init_llm_agent(api_key: str) -> ChatOpenAI:
    return ChatOpenAI(model=MODEL_NAME, openai_api_key=api_key, openai_api_base="https://openrouter.ai/api/v1", temperature=0)

def init_pandas_agent(llm_agent: ChatOpenAI, df: pd.DataFrame) -> Any:
    """Legacy"""
    columns_list = ", ".join(df.columns.tolist())
    prefix = f"Ты - эксперт. Доступные колонки: {columns_list}"
    return create_pandas_dataframe_agent(llm_agent, df, verbose=False, allow_dangerous_code=True, handle_parsing_errors=True, prefix=prefix)

def init_chromadb() -> Any:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_collection("data")

def init_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True, device='cpu')



def process_user_query(user_input: str, chat_id: str = "default", api_key: str = None, on_status: Callable[[str], None] = None) -> Dict[str, Any]:
    if not api_key: raise ValueError("API ключ не указан")
    
    print(f"\n🎯 Запрос от {chat_id}: {user_input}")
    if on_status: on_status("Анализирую запрос...")
    
    history_manager.add_message(chat_id, "user", user_input)
    components = initialize_system_once(api_key)
    chat_history = history_manager.format_history_for_prompt(chat_id)
    
    destination = route_query_with_history(user_input, chat_history, components['llm_agent'])
    print(f"📍 Направление: {destination}")
    
    graph_data = None
    
    if destination == "plot":
        if on_status: on_status("Подготавливаю данные для графика...")
        response, graph_data = process_plot_with_pd_agent_analysis_exec(user_input, chat_history, components, on_status)
        
    elif destination == "chat":
        if on_status: on_status("Формирую ответ...")
        response = process_llm_with_history(user_input, chat_history, components['llm_agent'])
        
    else: 
        if on_status: on_status("Ищу информацию в базе знаний и таблицах...")
        response, graph_data = process_data_query_with_history(user_input, chat_history, components, on_status)
    
    history_manager.add_message(chat_id, "assistant", response, graph_data)
    
    return {"text": response, "graph_data": graph_data, "chat_id": chat_id}

def route_query_with_history(user_input: str, chat_history: str, llm_agent: ChatOpenAI) -> str:
    response_schemas = [ResponseSchema(name="destination", description="chat, analysis или plot", type="string")]
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = parser.get_format_instructions()
    
    router_prompt = PromptTemplate(
        template="""Классифицируй намерение пользователя.
        ИСТОРИЯ: {chat_history}
        ЗАПРОС: {user_input}
        КАТЕГОРИИ:
        1. "plot" — Графики, визуализация.
        2. "analysis" — Статистика, медицинские вопросы, поиск записей.
        3. "chat" — Приветствия, оффтоп, болтовня.
        {format_instructions}""",
        input_variables=["user_input", "chat_history"],
        partial_variables={"format_instructions": format_instructions}
    )
    try:
        chain = LLMChain(llm=llm_agent, prompt=router_prompt, output_parser=parser)
        result = chain.run(user_input=user_input, chat_history=chat_history)
        dest = result['destination']
        if dest == 'pd_agent' or dest == 'llm': return 'analysis'
        return dest
    except: return "analysis"

def process_llm_with_history(user_input: str, chat_history: str, llm_agent: ChatOpenAI) -> str:
    llm_prompt = PromptTemplate(
        template="""Ты — медицинский ассистент. ИСТОРИЯ: {chat_history}. ЗАПРОС: {user_input}. Отвечай профессионально.""",
        input_variables=["user_input", "chat_history"]
    )
    return LLMChain(llm=llm_agent, prompt=llm_prompt).run(user_input=user_input, chat_history=chat_history)

def process_data_query_with_history(user_input: str, chat_history: str, components: Dict[str, Any], on_status: Callable = None) -> Tuple[str, Optional[str]]:
    if on_status: on_status("🔍 Поиск в справочниках...")
    enhanced_query = f"{chat_history}\nТекущий запрос: {user_input}"
    rag_context, rag_docs = perform_rag_search(enhanced_query, components)
    
    if on_status: on_status("📊 Анализ табличных данных...")
    pandas_output = run_pandas_code_exec(user_input, chat_history, components)
    
    if on_status: on_status("✍️ Формирование заключения (Судья)...")
    response = ensemble_with_rag_and_pandas(user_input, chat_history, rag_context, pandas_output)
    
    return response, None

def run_pandas_code_exec(user_input: str, chat_history: str, components: Dict[str, Any]) -> Optional[str]:
    print("📊 Code Exec Generation...")
    df = components['df']
    llm_agent = components['llm_agent']
    
    columns_info = []
    for col in df.columns:
        try:
            vals = df[col].dropna().head(3).tolist()
            columns_info.append(f"- {col} ({df[col].dtype}), примеры: {vals}")
        except: pass
    columns_str = "\n".join(columns_info)

    prompt_template = """Ты — Python Data Analyst. 
        ДАННЫЕ (df): {columns_str}
        ИСТОРИЯ: {chat_history}
        ВОПРОС: {user_input}
        ТРЕБОВАНИЯ:
        1. Напиши Python код для анализа `df`.
        2. Сохрани ответ в `final_answer` (строка).
        3. НЕ используй print.
        Пиши ТОЛЬКО код.
        """
    prompt = PromptTemplate(template=prompt_template, input_variables=["columns_str", "chat_history", "user_input"])

    try:
        chain = LLMChain(llm=llm_agent, prompt=prompt)
        gen_code = chain.run(columns_str=columns_str, chat_history=chat_history, user_input=user_input)
        cleaned_code = gen_code.replace("```python", "").replace("```", "").strip()
        
        local_scope = {"pd": pd, "np": np, "df": df, "final_answer": "Не удалось вычислить"}
        try:
            exec(cleaned_code, {}, local_scope)
            return str(local_scope.get("final_answer", "Нет ответа"))
        except Exception as e:
            return None
    except: return None

def perform_rag_search(user_input: str, components: Dict[str, Any]) -> Tuple[str, List[str]]:
    query_embedding = components['embedding_model'].encode([user_input], normalize_embeddings=True)[0]
    results = components['collection'].query(query_embeddings=[query_embedding.tolist()], n_results=6, include=["documents"])
    rag_docs = results["documents"][0]
    context = "\n\n".join(rag_docs)
    return context, rag_docs

def ensemble_with_rag_and_pandas(user_input: str, chat_history: str, rag_context: Optional[str], pandas_output: Optional[str]) -> str:
    sources_text = ""
    if rag_context: sources_text += f"\n[Медицинская база знаний]:\n{rag_context}\n"
    if pandas_output: sources_text += f"\n[Статистика из таблицы (Факты)]:\n{pandas_output}\n"
    if not rag_context and not pandas_output: sources_text += "\n(Данных в источниках не найдено)\n"

    gen_prompt_template = """Ты — медицинский аналитик.
        ФАКТЫ: {sources_text}
        ИСТОРИЯ: {chat_history}
        ВОПРОС: {user_input}
        ОТВЕТ:"""
    
    gen_prompt = PromptTemplate(template=gen_prompt_template, input_variables=["user_input", "chat_history", "sources_text"])
    
    try:
        llm = build_llm("google/gemma-3-27b-it", API_KEY)
        draft_answer = LLMChain(llm=llm, prompt=gen_prompt).run(user_input=user_input, chat_history=chat_history, sources_text=sources_text)
        
        judge_llm = ChatOpenAI(
            model="Qwen/Qwen3-Next-80B-A3B-Instruct",
            openai_api_key=API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0
        )
        
        judge_prompt_template = """
        Ты — Главный Врач. Отредактируй ответ ассистента для пользователя.

        ВОПРОС ПОЛЬЗОВАТЕЛЯ: {user_input}
        ЧЕРНОВИК ОТВЕТА: {draft_answer}
        ИСТИННЫЕ ДАННЫЕ (из таблицы): {pandas_output}

        ТРЕБОВАНИЯ К ФИНАЛЬНОМУ ОТВЕТУ:
        1. Убери любые упоминания кода, python, df, pandas, "предположим".
        2. Оставь только медицинские факты и цифры.
        3. Если данных нет — напиши вежливо: "К сожалению, данных недостаточно".
        4. Стиль: профессиональный, четкий, врачебный.

        ФИНАЛЬНЫЙ ТЕКСТ:"""
        
        judge_prompt = PromptTemplate(
            template=judge_prompt_template,
            input_variables=["user_input", "draft_answer", "pandas_output"]
        )
        
        final_verdict = LLMChain(llm=judge_llm, prompt=judge_prompt).run(
            user_input=user_input, 
            draft_answer=draft_answer, 
            pandas_output=str(pandas_output)[:500]
        )
        return final_verdict

    except Exception as e:
        print(f"Ensemble error: {e}")
        return "Извините, произошла ошибка при формировании ответа."


def process_plot_with_pd_agent_analysis_exec(user_input: str, chat_history: str, components: Dict[str, Any], on_status: Callable = None) -> Tuple[str, Optional[str]]:
    if on_status: on_status("🖥️ Проектирую график...")
    
    df = components['df']
    llm_agent = components['llm_agent']
    
    try:
        columns_info = []
        for col in df.columns[:25]:
            sample_vals = df[col].dropna().sample(min(3, len(df))).tolist() if len(df) > 0 else []
            sample_str = ', '.join([str(v)[:30] for v in sample_vals]) 
            columns_info.append(f"- {col} ({df[col].dtype}), примеры: {sample_str}")
        columns_str = "\n".join(columns_info)

        code_prompt = """Ты - Python Data Scientist. 
        Твоя задача: написать код для графика (matplotlib).
        ДАННЫЕ (df): {columns_str}
        ЗАПРОС: {user_input}
        ИСТОРИЯ: {chat_history}

        ИНСТРУКЦИЯ:
        1. `plot_data = df.copy()`
        2. ДАТЫ: `plot_data['дата'] = pd.to_datetime(plot_data['дата'], dayfirst=True, errors='coerce')` (только конкретную колонку!).
        3. ОБРЕЗКА (ВАЖНО):
        - Инициализируй `is_truncated = False`.
        - Если `len(plot_data) > 20000` и это SCATTER PLOT:
            `plot_data = plot_data.sample(n=20000, random_state=42)`
            `is_truncated = True`
        4. ПОСТРОЕНИЕ: `plt.figure(figsize=(10, 6))`.
        Пиши ТОЛЬКО код.
        """
        prompt = PromptTemplate(template=code_prompt, input_variables=["columns_str", "user_input", "chat_history"])
        
        gen_code = LLMChain(llm=llm_agent, prompt=prompt).run(columns_str=columns_str, user_input=user_input, chat_history=chat_history)
        cleaned_code = gen_code.replace("```python", "").replace("```", "").strip()
        
        if on_status: on_status("📈 Отрисовываю график...")
        
        local_scope = {"pd": pd, "np": np, "plt": plt, "sns": sns, "df": df, "matplotlib": matplotlib, "is_truncated": False}
        plt.clf()
        plt.figure(figsize=(12, 8))
        
        try:
            exec(cleaned_code, {}, local_scope)
        except Exception as exec_err:
            # Fallback
            if "to_datetime" in str(exec_err) or "mappings" in str(exec_err):
                plt.clf()
                num_cols = df.select_dtypes(include='number').columns
                if len(num_cols) > 0:
                    df[num_cols[0]].hist()
                    plt.title("Fallback")
                    local_scope['plot_data'] = df
                else:
                    return f"Не удалось построить график: {exec_err}", None
            else:
                return f"Ошибка построения графика: {exec_err}", None

        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        if on_status: on_status("📝 Анализирую результаты визуализации...")
        
        final_data = local_scope.get("plot_data", df)
        is_truncated = local_scope.get("is_truncated", False)
        
        if final_data.empty:
            return "График пуст (нет данных).", img_str

        try:
            desc = final_data.describe(include='all').to_string()[:1500]
        except: desc = "Статистика недоступна"
            
        warning = " (Данные ограничены 20000 точек)" if is_truncated else ""

        final_prompt = f"""
        Пользователь просил: "{user_input}"
        Статистика графика{warning}: {desc}
        
        ТВОЯ ЗАДАЧА:
        Напиши краткий вывод (2-3 предложения).
        
        ЗАПРЕЩЕНО:
        - Писать код.
        - Писать "Предположим df".
        
        ПРОСТО ОПИШИ ТРЕНДЫ ИЛИ РАСПРЕДЕЛЕНИЕ.
        """
        final_answer = llm_agent.invoke(final_prompt).content
        
        return final_answer, img_str
        
    except Exception as e:
        return f"Ошибка: {e}", None
    finally: plt.close('all')

def build_llm(model_name: str, api_key: str, temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(model=model_name, openai_api_key=api_key, openai_api_base="https://openrouter.ai/api/v1", temperature=temperature)

def handle_telegram_message(message_text: str, chat_id: str, api_key: str, on_status: Callable[[str], None] = None) -> Dict[str, Any]:
    # 1. Ручная обработка /clear
    if message_text.strip().lower() == "/clear":
        history_manager.clear_history(chat_id)
        return {
            "text": "История медицинского диалога очищена 🧹",
            "graph_data": None,
            "chat_id": chat_id
        }
    
    elif message_text.strip().lower() == "/history":
        history = history_manager.get_history(chat_id)
        if not history:
            text = "История медицинского диалога пуста 📭"
        else:
            formatted = []
            for msg in history:
                role = "👤 Пациент" if msg["role"] == "user" else "🤖 Доктор НЯМ"
                content = msg['content']
                # Убираем метки графиков
                content = re.sub(r'\[GRAPH_FILE:.*?\]', '', content).strip()
                if content:
                    formatted.append(f"{role}: {content[:200]}")
            text = "История медицинского диалога:\n\n" + "\n\n".join(formatted)
            
        return {
            "text": text,
            "graph_data": None,
            "chat_id": chat_id
        }

    return process_user_query(message_text, chat_id, api_key, on_status)

def initialize_medical_agent(api_key: str):
    return initialize_system_once(api_key)


