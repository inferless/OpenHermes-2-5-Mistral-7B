from vllm import LLM, SamplingParams

class InferlessPythonModel:
    def initialize(self):
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95,max_tokens=128)
        self.llm = LLM(model="TheBloke/OpenHermes-2.5-Mistral-7B-AWQ", quantization="awq", dtype="float16")
    def infer(self, inputs):
        prompts = inputs["prompt"]
        result = self.llm.generate(prompts, self.sampling_params)
        result_output = [output.outputs[0].text for output in result]
        print(result_output)
        return {generated_result': result_output[0]}
    def finalize(self,*args):
        self.llm = None
