from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download
from pathlib import Path

class InferlessPythonModel:
    def initialize(self):
        repo_id = "TheBloke/OpenHermes-2.5-Mistral-7B-AWQ"  # Specify the model repository ID
        # HF_TOKEN = os.getenv("HF_TOKEN")  # Access Hugging Face token from environment variable
        volume_nfs = "/var/nfs-mount/common_llm"  # Define nfs volume for fast cold-start
        model_dir = f"{volume_nfs}/{repo_id}"  # Construct model directory path
        model_dir_path = Path(model_dir)  # Convert path to Path object

        # Create the model directory if it doesn't exist
        if not model_dir_path.exists():
            model_dir_path.mkdir(exist_ok=True, parents=True)

        # Download the model snapshot from Hugging Face Hub
        snapshot_download(
            repo_id,
            local_dir=model_dir
            # token=HF_TOKEN  # Provide token if necessary
        )

        # Define sampling parameters for model generation
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=128)

        # Initialize the LLM object
        self.llm = LLM(model=model_dir, quantization="awq", dtype="float16")
        
    def infer(self,inputs):
        prompts = inputs["prompt"]  # Extract the prompt from the input
        result = self.llm.generate(prompts, self.sampling_params)
        # Extract the generated text from the result
        result_output = [output.outputs[0].text for output in result]

        # Return a dictionary containing the result
        return {'result': result_output[0]}

    def finalize(self):
        pass
