import os
from vllm import SamplingParams
from vllm import LLM
from huggingface_hub import snapshot_download


class InferlessPythonModel:
    def initialize(self):
        print("Inside Initialize Function", flush=True)
        self.template = """SYSTEM: You are a helpful assistant.
        USER: {}
        ASSISTANT: """
        snapshot_download(
            "meta-llama/Llama-2-7b-chat-hf",
            local_dir="/model",
            token="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",
        )
        self.llm = LLM("/model")
    
    def infer(self, inputs):
        print("Inside Infer Function", flush=True)
        prompts = [self.template.format(q) for q in inputs["questions"]]
        print("Sampling...", flush=True)
        sampling_params = SamplingParams(
            temperature=0.75,
            top_p=1,
            max_tokens=800,
            presence_penalty=1.15,
        )
        print("Generating...", flush=True)
        result = self.llm.generate(prompts, sampling_params)
        num_tokens = 0
        for output in result:
            num_tokens += len(output.outputs[0].token_ids)
            print(output.prompt, output.outputs[0].text, "\n\n", sep="")
        print(f"Generated {num_tokens} tokens")

    def finalize(self):
        pass
