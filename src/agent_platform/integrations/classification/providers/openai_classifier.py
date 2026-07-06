from __future__ import annotations

from openai import AsyncOpenAI

from agent_platform.integrations.classification.base import ClassificationModel
from agent_platform.models.chunk import TextChunk


_SYSTEM_TEMPLATE = (
    "You are a text classifier. Classify the following text into exactly one "
    "of these categories: {labels}. Respond with ONLY the category name, "
    "nothing else."
)


class OpenAIZeroShotClassifier(ClassificationModel):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def classify(
        self,
        items: list[TextChunk],
        candidate_labels: list[str] | None = None,
    ) -> list[str]:

        labels = candidate_labels or ["positive", "negative", "neutral"]
        system_prompt = _SYSTEM_TEMPLATE.format(labels=", ".join(labels))

        results: list[str] = []
        for item in items:
            if not item.text.strip():
                results.append("")
                continue

            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": item.text},
                ],
                temperature=0.0,
                max_tokens=50,
            )

            label = response.choices[0].message.content

            if label:
                results.append(label.strip())

        return results
