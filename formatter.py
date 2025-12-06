# formatter.py
import re
from typing import Optional


class TelegramHTMLFormatter:
    """Класс для форматирования ответов в HTML, совместимый с Telegram"""
    
    @staticmethod
    def markdown_to_html(text: str) -> str:
        """
        Преобразует базовый Markdown в Telegram-совместимый HTML
        Telegram поддерживает: <b>, <i>, <u>, <s>, <code>, <pre>
        """
        if not text:
            return ""
        
        # Убираем возможные метки файлов графиков
        text = re.sub(r'\[GRAPH_FILE:.*?\]', '', text)
        
        # Сначала обрабатываем блоки кода, чтобы не преобразовывать Markdown внутри них
        # Блок кода: ```код```
        text = re.sub(r'```([\s\S]*?)```', r'<pre>\1</pre>', text)
        
        # Inline код: `код`
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        
        # Жирный текст: **текст** или __текст__
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'__(.*?)__', r'<b>\1</b>', text)
        
        # Курсив: *текст* или _текст_
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        text = re.sub(r'_(.*?)_', r'<i>\1</i>', text)
        
        # Зачеркнутый текст: ~~текст~~
        text = re.sub(r'~~(.*?)~~', r'<s>\1</s>', text)
        
        # Заголовки: # Заголовок -> <b>Заголовок</b>
        text = re.sub(r'^### (.*?)$', r'<b>\1</b>', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.*?)$', r'<b>\1</b>', text, flags=re.MULTILINE)
        text = re.sub(r'^# (.*?)$', r'<b>\1</b>', text, flags=re.MULTILINE)
        
        # Списки: - элемент -> • элемент
        text = re.sub(r'^\s*[-*]\s+(.*?)$', r'• \1', text, flags=re.MULTILINE)
        
        # Нумерованные списки: 1. элемент -> 1. элемент (просто оставляем как есть)
        
        # Горизонтальные разделители: --- или *** -> разделительная линия
        text = re.sub(r'^---\s*$', '─────────────────────', text, flags=re.MULTILINE)
        text = re.sub(r'^\*\*\*\s*$', '─────────────────────', text, flags=re.MULTILINE)
        
        # Обработка таблиц (простых)
        text = TelegramHTMLFormatter._format_tables(text)
        
        # Экранирование HTML символов (только для текста вне тегов)
        text = TelegramHTMLFormatter._escape_html_safe(text)
        
        return text.strip()
    
    @staticmethod
    def _escape_html_safe(text: str) -> str:
        """Экранирует HTML, но сохраняет разрешённые теги Telegram."""
        VALID_TAGS = {"b", "i", "u", "s", "code", "pre"}

        parts = re.split(r'(<[^>]+>)', text)
        result_parts = []

        for part in parts:
            if part.startswith("<") and part.endswith(">"):
                # Попытка определить настоящий HTML-тег
                tag = re.sub(r'[<\/> ]', '', part).lower()
                if tag in VALID_TAGS:
                    result_parts.append(part)  # настоящий тег — оставляем
                else:
                    # Неизвестный "тег" — экранируем
                    escaped = part.replace('<', '&lt;').replace('>', '&gt;')
                    result_parts.append(escaped)
            else:
                # Обычный текст — экранируем
                escaped = (
                    part.replace('&', '&amp;')
                        .replace('<', '&lt;')
                        .replace('>', '&gt;')
                )
                result_parts.append(escaped)

        return ''.join(result_parts)

    
    @staticmethod
    def _format_tables(text: str) -> str:
        """Форматирует простые Markdown таблицы в читаемый вид"""
        lines = text.split('\n')
        formatted_lines = []
        in_table = False
        table_rows = []
        
        for i, line in enumerate(lines):
            if '|' in line and ('---' in line or '===' in line):
                # Это разделитель таблицы, пропускаем или заменяем на что-то простое
                continue
            
            if '|' in line and len(line.split('|')) > 2:
                # Это строка таблицы
                if not in_table:
                    in_table = True
                
                # Очищаем строку таблицы
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                if cells:
                    # Форматируем с выравниванием
                    formatted_row = ' | '.join(cells)
                    table_rows.append(formatted_row)
            else:
                if in_table and table_rows:
                    # Завершаем таблицу
                    formatted_lines.extend(table_rows)
                    table_rows = []
                    in_table = False
                
                formatted_lines.append(line)
        
        # Если текст закончился таблицей
        if in_table and table_rows:
            formatted_lines.extend(table_rows)
        
        return '\n'.join(formatted_lines)
    
    @staticmethod
    def format_medical_response(response: str) -> str:
        """
        Специальное форматирование для медицинских ответов
        ТОЛЬКО преобразование Markdown в HTML, без добавления лишних элементов
        """
        if not response:
            return "Нет данных для отображения."
        
        # Просто преобразуем Markdown в HTML и возвращаем
        return TelegramHTMLFormatter.markdown_to_html(response)