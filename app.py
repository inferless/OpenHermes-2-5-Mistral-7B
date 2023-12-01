import json
import numpy as np
import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
import base64
from io import BytesIO


class InferlessPythonModel:
  def initialize(self):
      self.tokenizer = AutoTokenizer.from_pretrained("TheBloke/OpenHermes-2.5-Mistral-7B-GPTQ", use_fast=True)
      self.model = AutoGPTQForCausalLM.from_quantized(
        "TheBloke/OpenHermes-2.5-Mistral-7B-GPTQ",
        use_safetensors=True,
        device="cuda:0",
        quantize_config=None,
        inject_fused_attention=False
      )

  def infer(self, inputs):
    prompt = inputs["prompt"]
    input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    output = self.model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=200)
    result = self.tokenizer.decode(output[0])
    return {"generated_result": result}

  def finalize(self,args):
    self.tokenizer = None
    self.model = None