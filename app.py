from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download
from pathlib import Path

class InferlessPythonModel:
    def initialize(self):
        model_id = "TheBloke/OpenHermes-2.5-Mistral-7B-AWQ"  # Specify the model repository ID
        # Define sampling parameters for model generation
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=128)
        # Initialize the LLM object
        self.llm = LLM(model=model_id, quantization="awq", dtype="float16")
        
    def infer(self,inputs):
        prompts = inputs["prompt"]  # Extract the prompt from the input
        result = self.llm.generate(prompts, self.sampling_params)
        # Extract the generated text from the result
        result_output = [output.outputs[0].text for output in result]

        # Return a dictionary containing the result
        return {'generated_result': result_output[0]}

    def finalize(self):
        pass
