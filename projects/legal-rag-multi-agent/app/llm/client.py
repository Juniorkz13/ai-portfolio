class FakeLLMClient:
    def generate(self, prompt: str) -> str:
        return (
            "Com base nas informações analisadas, existem aspectos jurídicos "
            "que devem ser avaliados com cautela. O tema envolve requisitos "
            "legais específicos e possíveis interpretações divergentes, "
            "o que pode impactar a decisão final."
        )