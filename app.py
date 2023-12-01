import os
from vllm import SamplingParams
from vllm import LLM
from huggingface_hub import snapshot_download


class InferlessPythonModel:
    def initialize(self):
        snapshot_download(
            "teknium/OpenHermes-2.5-Mistral-7B",
            local_dir="/model",
            token="<<your_token>>",
        )
        self.llm = LLM("/model")
    
    def infer(self, inputs):
        prompts = inputs["questions"]
        sampling_params = SamplingParams(
            temperature=0.75,
            top_p=1,
            max_tokens=800,
            presence_penalty=1.15,
        )
        result = self.llm.generate(prompts, sampling_params)
        result_output = [output.outputs[0].text for output in result]

        return {"result": result_output[0]}

    def finalize(self, args):
        pass
