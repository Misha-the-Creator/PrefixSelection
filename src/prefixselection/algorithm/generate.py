import torch
from transformers.generation import GenerationMixin
from transformers.tokenization_utils_sentencepiece import SentencePieceBackend
from transformers.tokenization_utils_tokenizers import TokenizersBackend

TokenizerType = TokenizersBackend | SentencePieceBackend

class Generation:
    def __init__(
        self,
        llms: list[GenerationMixin],
        tokenizers: list[TokenizerType],
        top_k: int,
    ):
        self.llms = llms
        self.tokenizers = tokenizers
        self.chat_prefixes = []
        self.top_k = top_k
        self.devices = [next(llm.parameters()).device for llm in llms]

    @staticmethod
    def _build_chat_prefix(
        tokenizer: TokenizerType,
        user_message: str,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": user_message,
            }
        ]

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @staticmethod
    def _tokenize_prompt(
        tokenizer: TokenizerType,
        prompt: str,
        device: torch.device,
    ):
        return tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)

    @staticmethod
    def _generate_log_probs(
        model: GenerationMixin,
        inputs,
    ) -> torch.Tensor:
        with torch.inference_mode():
            logits = model(**inputs).logits[:, -1, :]

        return torch.log_softmax(
            logits.float(),
            dim=-1,
        )

    def _get_top_distribution(
        self,
        log_probs: torch.Tensor,
        tokenizer: TokenizerType,
    ) -> dict[str, float]:
        k = min(self.top_k, log_probs.shape[-1])

        top_log_probs, top_token_ids = torch.topk(
            log_probs[0],
            k=k,
        )

        distribution: dict[str, float] = {}

        for _, (token_id, log_prob) in enumerate(
            zip(top_token_ids, top_log_probs, strict=True),
            start=1,
        ):
            token_id_int = token_id.item()

            token_text = tokenizer.decode(
                [token_id_int],
                skip_special_tokens=False,
            )

            probability = log_prob.exp().item()

            distribution[token_text] = (
                distribution.get(token_text, 0.0)
                + probability
            )

        return distribution

    def initialize_chat(self, user_message: str) -> None:
        self.chat_prefixes = []
        for tokenizer in self.tokenizers:
            chat_prefix = self._build_chat_prefix(tokenizer=tokenizer,
                                                  user_message=user_message,)
            self.chat_prefixes.append(chat_prefix)

    def generate_pipe(
        self,
        generated_text: str,
        model_to_run: str = '',
    ):
        if self.chat_prefixes is None:
            raise RuntimeError(
                "Сначала вызови initialize_chat(user_message)"
            )

        distributions: dict[str, dict[str, float]] = {}
        if model_to_run == "":
            for idx, (llm, tokenizer) in enumerate(zip(self.llms, self.tokenizers, strict=True)):
                prompt = self.chat_prefixes[idx] + generated_text
                inputs = self._tokenize_prompt(
                    tokenizer=tokenizer,
                    prompt=prompt,
                    device=self.devices[idx],
                )

                log_probs = self._generate_log_probs(model=llm,
                                                     inputs=inputs,)

                distributions[idx] = self._get_top_distribution(log_probs=log_probs,
                                                                tokenizer=tokenizer,)

        if model_to_run != "":
            model_to_run = int(model_to_run)
            prompt = self.chat_prefixes[model_to_run] + generated_text
            inputs = self._tokenize_prompt(tokenizer=self.tokenizers[model_to_run],
                                           prompt=prompt,
                                           device=self.devices[model_to_run])

            log_probs = self._generate_log_probs(model=self.llms[model_to_run],
                                                 inputs=inputs,)

            distributions[model_to_run] = self._get_top_distribution(log_probs=log_probs,
                                                                     tokenizer=self.tokenizers[model_to_run],)

        return distributions