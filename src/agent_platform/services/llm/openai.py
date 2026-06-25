from openai import AsyncOpenAI

class OpenAILLM:  
    client: AsyncOpenAI


    def __init__(
        self,
        api_key: str,
    ):
        self.client = AsyncOpenAI(api_key=api_key)


    async def generate(
        self,
        prompt: str,
        model: str,
    ) -> str | None:

        chat_completion = await self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
        )

        return chat_completion.choices[0].message.content