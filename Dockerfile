# Используем официальный Python образ
FROM python:3.11-slim-bullseye

# Устанавливаем рабочую директорию
WORKDIR /home

# Устанавливаем Poetry
ENV POETRY_VERSION=1.8.5
RUN pip install poetry==$POETRY_VERSION

# Настраиваем Poetry
RUN poetry config virtualenvs.create false

# Копируем файлы Poetry для установки зависимостей
COPY pyproject.toml poetry.lock /home/

# Устанавливаем Python-зависимости через Poetry
RUN poetry install --no-interaction --no-ansi

# Копируем весь исходный код проекта
COPY . /home

# Указываем порт, который будет слушать приложение
EXPOSE 8000 5000

CMD poetry run python commands.py
