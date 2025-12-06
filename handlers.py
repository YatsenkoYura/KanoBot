import logging
import asyncio
import base64
import re
from io import BytesIO
import time

from aiogram import Router, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message, BufferedInputFile
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramBadRequest

from medical_agent import handle_telegram_message
from lexicon import start_message, help_message
from formatter import TelegramHTMLFormatter

logger = logging.getLogger(__name__)

def create_router(api_key: str) -> Router:
    router = Router()

    @router.message(CommandStart())
    async def cmd_start(message: Message):
        await message.answer(start_message)

    @router.message(Command("help"))
    async def cmd_help(message: Message):
        await message.answer(help_message)

    @router.message(Command("chat"))
    async def cmd_chat(message: Message):
        await message.answer("Введите ваше сообщение!")

    @router.message(Command("clear"))
    async def cmd_clear(message: Message):
        # Вызываем напрямую, без callback'а
        response = await asyncio.to_thread(
            handle_telegram_message, "/clear", str(message.chat.id), api_key, None
        )
        await message.answer(response["text"])

    @router.message(Command("history"))
    async def cmd_history(message: Message):
        # Вызываем напрямую
        response = await asyncio.to_thread(
            handle_telegram_message, "/history", str(message.chat.id), api_key, None
        )
        # ВАЖНО: parse_mode=None, чтобы сырой текст истории не ломал HTML разметку
        if len(response["text"]) > 4096:
            for x in range(0, len(response["text"]), 4096):
                await message.answer(response["text"][x:x+4096], parse_mode=None)
        else:
            await message.answer(response["text"], parse_mode=None)

    @router.message()
    async def get_medical_response(message: Message):
        user_message = message.text
        
        if not user_message or user_message.strip() == "":
            await message.answer("Пожалуйста, введите ваш медицинский вопрос.")
            return
        
        # Статусное сообщение
        status_msg = await message.answer("⏳ Доктор НЯМ начал анализ...")
        await message.bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

        loop = asyncio.get_running_loop()
        last_status_update = [0] 

        # Callback для обновления статуса
        def status_callback(text: str):
            current_time = time.time()
            if current_time - last_status_update[0] > 1.5:
                last_status_update[0] = current_time
                async def _update_wrapper():
                    try:
                        await status_msg.edit_text(f"⏳ {text}")
                    except Exception:
                        pass 
                asyncio.run_coroutine_threadsafe(_update_wrapper(), loop)

        try:
            response = await asyncio.to_thread(
                handle_telegram_message,
                user_message, 
                str(message.chat.id), 
                api_key,
                status_callback 
            )
            
            # График
            if response.get("graph_data"):
                await status_msg.delete() 
                try:
                    graph_bytes = base64.b64decode(response["graph_data"])
                    try:
                        photo = BufferedInputFile(
                            file=graph_bytes,
                            filename=f"chart_{response['chat_id']}.png"
                        )
                        await message.bot.send_photo(
                            chat_id=message.chat.id,
                            photo=photo,
                            caption="📈 Визуализация медицинских данных"
                        )
                    except:
                        bio = BytesIO(graph_bytes)
                        bio.name = f"chart.png"
                        await message.bot.send_photo(
                            chat_id=message.chat.id,
                            photo=bio,
                            caption="📈 Визуализация медицинских данных"
                        )
                except Exception as graph_error:
                    logger.error(f"Ошибка графика: {graph_error}")

            # Текст
            formatted_response = TelegramHTMLFormatter.format_medical_response(response["text"])
            logger.info(f'Ответ (длина {len(formatted_response)}): {formatted_response[:100]}...')
            
            def clean_html_tags(text):
                return re.sub(r'<[^>]+>', '', text)

            if response.get("graph_data"):
                # Если был график, шлем текст новым сообщением
                if len(formatted_response) > 4096:
                    for x in range(0, len(formatted_response), 4096):
                        part = formatted_response[x:x+4096]
                        try:
                            await message.answer(part, parse_mode="HTML")
                        except TelegramBadRequest:
                            await message.answer(clean_html_tags(part), parse_mode=None)
                else:
                    try:
                        await message.answer(formatted_response, parse_mode="HTML")
                    except TelegramBadRequest:
                        await message.answer(clean_html_tags(formatted_response), parse_mode=None)
            else:
                # Если только текст, редактируем статус
                if len(formatted_response) > 4096:
                    await status_msg.delete()
                    for x in range(0, len(formatted_response), 4096):
                        part = formatted_response[x:x+4096]
                        try:
                            await message.answer(part, parse_mode="HTML")
                        except TelegramBadRequest:
                            await message.answer(clean_html_tags(part), parse_mode=None)
                else:
                    try:
                        await status_msg.edit_text(formatted_response, parse_mode="HTML")
                    except TelegramBadRequest:
                        await status_msg.edit_text(clean_html_tags(formatted_response), parse_mode=None)
            
        except Exception as e:
            logger.exception("Ошибка при обращении к агенту")
            try:
                await status_msg.edit_text("❌ Произошла ошибка. Попробуйте позже.")
            except:
                await message.answer("❌ Произошла ошибка.")

    return router