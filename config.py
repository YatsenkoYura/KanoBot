# config.py
import logging
import os
from dataclasses import dataclass

from environs import Env

logger = logging.getLogger(__name__)


@dataclass
class BotSettings:
    token: str
    openrouter_api_key: str  # Заменили authorization_key


@dataclass
class LoggSettings:
    level: str
    format: str


@dataclass
class Config:
    bot: BotSettings
    log: LoggSettings


def load_config(path: str | None = None) -> Config:
    env = Env()

    if path:
        if not os.path.exists(path):
            logger.warning(".env file not found at '%s', skipping...", path)
        else:
            logger.info("Loading .env from '%s'", path)

    env.read_env(path)

    token = env("BOT_TOKEN")
    if not token:
        raise ValueError("BOT_TOKEN must not be empty")

    openrouter_api_key = env("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY must not be empty")

    logg_settings = LoggSettings(
        level=env("LOG_LEVEL", "INFO"),
        format=env("LOG_FORMAT", "%(filename)s:%(lineno)d #%(levelname)-8s [%(asctime)s] - %(name)s - %(message)s")
    )

    logger.info("Configuration loaded successfully")

    return Config(
        bot=BotSettings(token=token, openrouter_api_key=openrouter_api_key),
        log=logg_settings
    )