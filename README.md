## Установка

```bash
git clone https://github.com/YatsenkoYura/KanoBot.git
cd KanoBot
cp .env.example .env
```

### Необходимо заполнить .env файл, свой ключ бота, свой ключ на openrouter

```bash
docker build -t bot-ai .
docker run -it --rm   -v $(pwd)/.env:/app/.env   -e PYTHONUNBUFFERED=1   bot-ai
```

### Первый запуск будет долгим, установка всех зависимсотей и моделей долгий шаг
