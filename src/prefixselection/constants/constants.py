from enum import Enum


class Lang(Enum):
    RU='русский 🇷🇺'
    EN='english 🇬🇧'

MESSAGES={'EN': {'download_model': 'Enter the name of the model you want to download from Hugging Face'},
          'RU': {'download_model': 'Введите название модели, которую хотите скачать с Hugging Face'}}