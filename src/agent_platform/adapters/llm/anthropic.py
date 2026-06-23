from anthropic import AsyncAnthropic, HUMAN_PROMPT, AI_PROMPT


class AnthropicLLM:
    client: AsyncAnthropic


    def __init__(
        self,
        api_key: str,
    ):
        self.client = AsyncAnthropic(api_key=api_key)


    async def generate(
        self,
        prompt: str,
        max_tokens_to_sample: int,
        model: str,
    ) -> str | None:
    
        completion = await self.client.completions.create(
            model=model,
            max_tokens_to_sample=max_tokens_to_sample,
            prompt=f"{HUMAN_PROMPT}{prompt}{AI_PROMPT}",
        )

        return completion.completion