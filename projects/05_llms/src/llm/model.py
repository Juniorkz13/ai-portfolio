import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LocalLLM:
    def __init__(self):
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        self.model.eval()

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id
        )

        output = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()

        if "RESPOSTA" in output:
            output = output.split("RESPOSTA", 1)[-1].strip()
        elif "RESPOSTA:" in output:
            output = output.split("RESPOSTA:", 1)[-1].strip()
        
        output = output.lstrip(":").strip()

        if not output:
            return "Não sei responder com base nos documentos disponíveis."

        return output.split("\n")[0].strip()
