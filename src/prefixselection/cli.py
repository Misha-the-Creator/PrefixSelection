import click
from constants.constants import MESSAGES


@click.group(invoke_without_command=True)
@click.option('--lang', 
              prompt="Choose language / Выберите язык")
@click.pass_context
def select_lang(ctx: click.Context, lang: str):
    ctx.ensure_object(dict)
    while lang not in list(MESSAGES):
        click.echo(f"Please select one of the following values / Пожалуйста, выберите одно из следующих значений — {list(MESSAGES)}", err=True)    
        lang = click.prompt("Choose language / Выберите язык")
    ctx.obj['lang'] = lang
    ctx.obj['messages'] = MESSAGES[lang]
    click.echo(f"You have chosen / Вы выбрали {lang}")


@click.command
@click.option('--model_name', 
              prompt="Enter model name / Введите имя модели для скачивания через запятую", 
              help='Дает возможность ввести имя модели для скачивания / Allows you to enter the model name for downloading')
def select_model_to_download(raw_models_name: str):
    try:
        list_of_models = raw_models_name.split(', ')
    except Exception as e:
        click.echo('')


# @click.command
# @click.argument("")
