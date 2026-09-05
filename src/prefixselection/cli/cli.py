from typing import cast

import click
import torch
from algorithm import Generation, PrefixDense
from constants import MENU, MESSAGES
from logic.parser import ModelHandler
from transformers import AutoModelForCausalLM, AutoTokenizer


@click.group(invoke_without_command=True)
@click.option("--lang", prompt="Choose language / Выберите язык",
              type=click.Choice(list(MESSAGES), case_sensitive=False))
@click.pass_context
def entry(ctx: click.Context, lang: str):
    ctx.ensure_object(dict)
    ctx.obj["lang"] = lang
    ctx.obj["t"] = MESSAGES[lang]
    ctx.obj["models"] = []
    click.echo(f"You have chosen / Вы выбрали {lang}")

    if ctx.invoked_subcommand is None:
        menu_loop(ctx)


@entry.command("load-model")
@click.option("--model", default=None)
@click.pass_obj
def load_model(obj, model):
    t = obj["t"]
    if model is None:
        model = click.prompt(t["model_prompt"])

    model_handler = ModelHandler()
    if not model_handler.validate_if_model_exists(model_str=model):
        click.echo(t["wrong_input"])
        return

    try:
        click.echo(t["downloading"])
        tokenizer = AutoTokenizer.from_pretrained(f'{model}')
        model = AutoModelForCausalLM.from_pretrained(model,
                                                       torch_dtype=torch.float16,
                                                       device_map="auto")  
    except Exception as e:
        click.echo(f'При скачивании произошла ошибка: {e}')
        return
    obj["models"].append((model, tokenizer))
    click.echo(t["chosen"].format(model_name=model.name_or_path, number_of_models=len(obj["models"])))


@entry.command("enter-generation-text")
@click.option("--text", default=None)
@click.pass_obj
def enter_generation_text(obj, text):
    t = obj["t"]
    if not obj["models"]:
        raise click.UsageError(t["empty"])

    input_text = click.prompt(t["model_prompt"])
    generation = Generation(llms=[elem[0] for elem in obj['models']], tokenizers=[elem[1] for elem in obj['models']], top_k=5)
    selection = PrefixDense(probs_generator=generation, 
                            input_str=input_text,
                            stop_token=['<|im_end|>', '<|endoftext|>', '</s>', '<end_of_turn>', '<|end_of_response|>', '<|end_of_inference|>'])
    answer = selection.runpipe()
    click.echo(f'Ответ модели: {answer}')


def menu_loop(ctx: click.Context):
    t = ctx.obj["t"]
    group = cast(click.Group, ctx.command)

    while True:
        click.echo()
        for i, (key, _) in enumerate(MENU, start=1):
            click.echo(f"  {i}. {t[key]}")
        click.echo(f"  0. {t['quit']}")

        choice = click.prompt(
            t["menu"],
            type=click.Choice([str(i) for i in range(len(MENU) + 1)]),
            show_choices=False,
        )
        if choice == "0":
            return

        cmd = group.get_command(ctx, MENU[int(choice) - 1][1])
        try:
            ctx.invoke(cmd)
        except click.UsageError as e:
            click.echo(f"Error: {e.format_message()}", err=True)