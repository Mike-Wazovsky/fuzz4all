import os
import random
import string
import time
from typing import List, Iterator

import torch
from grazie.api.client.chat.prompt import ChatPrompt
from grazie.api.client.chat.response import ChatResponseStream
from grazie.api.client.endpoints import GrazieApiGatewayUrls
from grazie.api.client.gateway import AuthType, GrazieApiGatewayClient
from grazie.api.client.llm_parameters import LLMParameters
from grazie.api.client.parameters import Parameters
from grazie.api.client.profiles import Profile

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable warning
EOF_STRINGS = ["<|endoftext|>", "###"]

class Grazie:
    def __init__(
            self, profile: Profile, device: str, eos: List, max_length: int, gpt_chance: float
    ) -> None:
        self.device = device

        token = os.getenv("GRAZIE_JWT_TOKEN")

        self.client = GrazieApiGatewayClient(
            url=GrazieApiGatewayUrls.STAGING,
            grazie_jwt_token=token,
            auth_type=AuthType.APPLICATION,
        )

        self.profile = profile
        self.max_length = max_length
        self.prefix_token = "<fim_prefix>"
        self.suffix_token = "<fim_suffix><fim_middle>"
        self.gpt_chance = gpt_chance

    @torch.inference_mode()
    def generate(
            self, prompt, batch_size=10, temperature=1.0, max_length=512
    ) -> List[str]:
        input_str = self.prefix_token + prompt + self.suffix_token

        outputs = []

        for i in range(batch_size):
            random_seed = self.generate_random_seed(128)
            response = self.request_grazie(input_str + random_seed, temperature)
            output = response.content
            outputs.append(output)

        return outputs

    def generate_random_seed(self, length):
        digits = string.digits
        return ''.join(random.choice(digits) for _ in range(length))

    def stream_to_str(self, response: Iterator[ChatResponseStream]):
        finished = False
        whole_response = ""
        chunk = None

        while not finished:
            try:
                chunk = next(response)
            except StopIteration:
                finished = True
            whole_response += chunk.chunk
        return whole_response

    def request_grazie(self, request: str, temperature: float):
        response = None


        while response is None:
            try:
                response = self.client.chat(
                    chat=(
                        ChatPrompt()
                        .add_user(request)
                    ),
                    profile=self.profile,
                    prompt_id="openai-gpt-4o-mini",
                    parameters={
                        LLMParameters.Temperature: Parameters.FloatValue(temperature),
                        LLMParameters.Length: Parameters.IntValue(16192),
                    }
                )
            except Exception as e:
                print(e)
                time.sleep(5)
        return response