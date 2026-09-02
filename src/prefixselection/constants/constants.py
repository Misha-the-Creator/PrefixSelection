MESSAGES = {
    'EN': {
        'enter_model_to_download': 'Add model to the ensemble',
        'generate': 'Make generation in response',
        'model_prompt': 'Enter the model name from Hugging Face.',
        'wrong_input': 'Such a model does not exist on Hugging Face. Enter the correct name.',
        'downloading': 'Downloading...',
        'chosen': 'The model {model_name} is installed. The number of models in the ensemble: {number_of_models}',
        'menu': 'Choose command',
        'quit': 'Quit',
    },
    'RU': {
        'enter_model_to_download': 'Добавить модель к ансамблю',
        'generate': 'Выполнить генерацию в ответ на ввод',
        'model_prompt': 'Введите название модели с Hugging Face',
        'wrong_input': 'Такой модели на hugging face не существует. Введите корректное название',
        'downloading': 'Скачивание началось...',
        'chosen': 'Модель {model_name} установлена. Число моделей в ансамбле: {number_of_models}',
        'menu': 'Выберите команду',
        'quit': 'Выход',
    },
}

MENU = [
    ("enter_model_to_download", "load-model"),
    ("generate", "generate"),
]