from huggingface_hub import repo_exists


class ModelHandler:
    def __init__(self):
        pass

    def validate_if_model_exists(self, model_str: str):
        return repo_exists(model_str)