from mistralai.client import Mistral
from models.message import Message

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

    
    async def stream(
        self,
        prompt: str,
        model: str,
    ) -> str | None: 
        ...


    async def chat(
        self,
        messages: list[Message],
        model: str,
    ) -> str | None:

        response = await self.client.chat.complete_async(
            model=model,
            messages=messages,
        )

        content = response.choices[0].message.content

        return content if isinstance(content, str) else None
    


