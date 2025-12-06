import logging
import asyncio
import sys
import os

from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand
from aiogram.fsm.storage.memory import MemoryStorage

import handlers
from config import Config, load_config
from medical_agent import initialize_medical_agent

async def set_main_menu(bot: Bot):

    main_menu_commands = [
        BotCommand(command='/start',
                   description='Поприветствовать доктора НЯМа'),
        BotCommand(command='/help',
                   description='Что умеет доктор НЯМ?'),
        BotCommand(command='/chat',
                   description='Начать общение с доктором НЯМ'),
        BotCommand(command='/clear',
                   description='Очистить историю диалога'),
        BotCommand(command='/history',
                   description='Показать историю диалога'),
    ]

    await bot.set_my_commands(main_menu_commands)

logger = logging.getLogger(__name__)


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(filename)s:%(lineno)d #%(levelname)-8s '
               '[%(asctime)s] - %(name)s - %(message)s'
    )

    config: Config = load_config('.env')

    # Инициализируем медицинского агента при запуске бота
    logger.info("🔄 Инициализация медицинского агента...")
    try:
        initialize_medical_agent(config.bot.openrouter_api_key)
        logger.info("✅ Медицинский агент инициализирован")
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации медицинского агента: {e}")
        return

    storage = MemoryStorage()

    bot = Bot(
        token=config.bot.token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
        timeout=300,
    )

    dp = Dispatcher(storage=storage)

    dp.startup.register(set_main_menu)

    # Передаем API ключ для медицинского агента
    dp.include_router(handlers.create_router(config.bot.openrouter_api_key))

    await bot.delete_webhook(drop_pending_updates=True)
    try:
        await dp.start_polling(bot)
    except Exception as e:
        logger.exception(e)

if sys.platform.startswith("win") or os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

asyncio.run(main())