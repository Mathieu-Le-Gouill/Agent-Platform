# adapters/llm/openai.py
from mistralai.client import Mistral

class MistralLLM:
    client: Mistral


    def __init__(
        self,
        api_key: str,
    ):
        self.client = Mistral(api_key=api_key)


    async def generate(
        self,
        prompt: str,
        model: str,
    ) -> str | None:

        response = await self.client.chat.complete_async(
            model=model,
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        )

        content = getattr(response.choices[0].message, "content", None)

        return content if isinstance(content, str) else None