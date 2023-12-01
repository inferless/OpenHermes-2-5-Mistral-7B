import os
from vllm import SamplingParams
from vllm import LLM


class InferlessPythonModel:
    def initialize(self):
        self.llm = LLM(
          model="teknium/OpenHermes-2.5-Mistral-7B",
          quantization="awq",
          dtype="float16")

    def infer(self, inputs):
        prompts = inputs["questions"]
        sampling_params = SamplingParams(
            temperature=0.75,
            top_p=1,
            max_tokens=200
        )
        result = self.llm.generate(prompts, sampling_params)
        result_output = [output.outputs[0].text for output in result]

        return {"result": result_output[0]}

    def finalize(self, args):
        pass
