import os
from typing import Any, Dict, List, Optional, Tuple, Union

from grazie.api.client.gateway import AuthType, GrazieApiGatewayClient, GrazieHeaders
from grazie.api.client.chat.prompt import ChatPrompt
from grazie.api.client.endpoints import GrazieApiGatewayUrls
from grazie.api.client.llm_parameters import LLMParameters
from grazie.api.client.parameters import Parameters
from grazie.api.client.profiles import Profile

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable warning
EOF_STRINGS = ["<|endoftext|>", "###"]


class EndOfFunctionCriteria(StoppingCriteria):
    def __init__(self, start_length, eos, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_length = start_length
        self.eos = eos
        self.tokenizer = tokenizer
        self.end_length = {}

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length:]
        )
        done = []
        for index, decoded_generation in enumerate(decoded_generations):
            finished = any(
                [stop_string in decoded_generation for stop_string in self.eos]
            )
            if (
                    finished and index not in self.end_length
            ):  # ensures first time we see it
                for stop_string in self.eos:
                    if stop_string in decoded_generation:
                        self.end_length[index] = len(
                            input_ids[
                            index,  # get length of actual generation
                            self.start_length: -len(
                                self.tokenizer.encode(
                                    stop_string,
                                    add_special_tokens=False,
                                    return_tensors="pt",
                                )[0]
                            ),
                            ]
                        )
            done.append(finished)
        return all(done)


class LLAMA:
    def __init__(
            self, model_name: str, device: str, eos: List, max_length: int
    ) -> None:
        self.device = device

        token = os.getenv("GRAZIE_JWT_TOKEN")
        # In a real application, you would have to supply the client's IP address
        # self.client_ip = "{}.{}.{}.{}".format(*[str(random.randint(0, 255)) for octet in range(4)])

        self.client = GrazieApiGatewayClient(
            url=GrazieApiGatewayUrls.STAGING,
            grazie_jwt_token=token,
            auth_type=AuthType.APPLICATION,
        )

        self.eos = EOF_STRINGS + eos
        self.max_length = max_length
        self.prefix_token = "<fim_prefix>"
        self.suffix_token = "<fim_suffix><fim_middle>"
        self.skip_special_tokens = False

    @torch.inference_mode()
    def generate(
            self, prompt, batch_size=3, temperature=1.0, max_length=512
    ) -> List[str]:
        input_str = self.prefix_token + prompt + self.suffix_token

        outputs = []

        for i in range(batch_size):
            response = self.client.chat(
                chat=(
                    ChatPrompt()
                    .add_user(input_str)
                ),
                profile=Profile.OPENAI_GPT_4_TURBO,
                prompt_id="Fuzz4All-try",
                parameters={
                    LLMParameters.Temperature: Parameters.FloatValue(max(temperature, 1e-2)),
                    LLMParameters.Length: Parameters.IntValue(min(self.max_length, len(input_str) + max_length)),
                }
            )
            output = response.content
            print("DIRECT OUTPUT: \n" + output)
            print("DIRECT OUTPUT FINISHED\n")
            # removes eos tokens.
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index])

        return outputs


class StarCoder:
    def __init__(
            self, model_name: str, device: str, eos: List, max_length: int
    ) -> None:
        checkpoint = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/Users/ravsemirnov/Desktop/F4All/bigcode",
            local_files_only=True,
            # checkpoint, token='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'
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                # "/Users/ravsemirnov/Desktop/F4All/bigcode",
                checkpoint, token='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',
                trust_remote_code=True,
            )
            .to(torch.bfloat16)
            .to(device)
        )
        self.eos = EOF_STRINGS + eos
        self.max_length = max_length
        self.prefix_token = "<fim_prefix>"
        self.suffix_token = "<fim_suffix><fim_middle>"
        self.skip_special_tokens = False

    @torch.inference_mode()
    def generate(
            self, prompt, batch_size=10, temperature=1.0, max_length=512
    ) -> List[str]:
        input_str = self.prefix_token + prompt + self.suffix_token
        input_tokens = self.tokenizer.encode(input_str, return_tensors="pt").to(
            self.device
        )

        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )

        raw_outputs = self.model.generate(
            input_tokens,
            max_length=min(self.max_length, len(input_tokens[0]) + max_length),
            do_sample=True,
            top_p=1.0,
            temperature=max(temperature, 1e-2),
            num_return_sequences=batch_size,
            stopping_criteria=scores,
            output_scores=True,
            return_dict_in_generate=True,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]):]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index])
        return outputs


def make_model(eos: List, model_name: str, device: str, max_length: int):
    """Returns a llm model instance (optional: using the configuration file)."""

    kwargs_for_model = {
        "model_name": model_name,
        "eos": eos,
        "device": device,
        "max_length": max_length,
    }

    # print the model config
    print("=== Model Config ===")
    print(f"model_name: {model_name}")
    for k, v in kwargs_for_model.items():
        print(f"{k}: {v}")

    if "starcoder" in model_name.lower():
        model_obj = StarCoder(**kwargs_for_model)
    else:
        model_obj = LLAMA(**kwargs_for_model)

    model_obj_class_name = model_obj.__class__.__name__

    print(f"model_obj (class name): {model_obj_class_name}")
    print("====================")

    return model_obj
