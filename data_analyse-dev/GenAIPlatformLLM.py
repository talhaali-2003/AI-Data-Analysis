import json
import requests
from typing import Any, List, Mapping, Optional, Dict
from langchain_core.language_models.llms import LLM
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define the available models
AVAILABLE_MODELS = {
    "azureopenai": [
        "openai-gpt-4-turbo",
        "openai-gpt-35-turbo-16k"
    ],
    "bedrock": [
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-v2:1",
        "mistral.mixtral-8x7b-instruct-v0:1",
        "mistral.mistral-7b-instruct-v0:2",
        "mistral.mistral-large-2402-v1:0",
        "meta.llama3-8b-instruct-v1:0",
        "meta.llama3-70b-instruct-v1:0",
        "meta.llama2-13b-chat-v1",
        "meta.llama2-70b-chat-v1",
        "stability.stable-diffusion-xl-v1",
        "ai21.j2-mid-v1",
        "ai21.j2-ultra-v1",
        "cohere.command-text-v14",
        "cohere.command-light-text-v14",
        "cohere.embed-english-v3",
        "cohere.embed-multilingual-v3"
    ]
}

class GenAIPlatformLLM(LLM):
    api_key: str
    url: str
    provider: str
    model_id: str
    model_kwargs: Dict

    @property
    def _llm_type(self) -> str:
        """Define the LLM type."""
        return "GenAIPlatformLLM"

    def _prepare_payload(self, prompt: str, tools=None) -> Dict:
        """Prepare the payload for the API request."""
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "input": prompt,
            "tools": tools,
            "model_kwargs": json.dumps(self.model_kwargs),
            "adapter_args": "{}",
        }

    def _make_request(self, payload: Dict, headers: Dict) -> str:
        """Make the API request and return the response content."""
        response = requests.post(self.url, data=json.dumps(payload), headers=headers)
        if response.status_code == 200:
            return json.loads(response.text)["Answer"]["content"]
            #return response.text
        else:
            raise Exception(
                f"API request failed with status code: {response.status_code}, Response: {response.text}"
            )

    def _call(
        self,
        prompt: str,
        tools=None,
        stop: Optional[List[str]] = None
    ) -> str:
        """Makes an API call to the specified LLM."""
        headers = {"Content-type": "application/json", "X-API-Key": self.api_key}
        payload = self._prepare_payload(prompt, tools)
        response_text = self._make_request(payload, headers)

        if stop is not None:
            for token in stop:
                response_text = response_text.split(token)[0]

        return response_text

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Makes an API call to the specified LLM."""
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        headers = {"Content-type": "application/json", "X-API-Key": self.api_key}
        payload = self._prepare_payload(prompt)
        return self._make_request(payload, headers)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "api_key": self.api_key,
            "url": self.url,
            "model_id": self.model_id,
            "model_kwargs": self.model_kwargs,
        }

    @staticmethod
    def get_available_model_ids() -> List[str]:
        """Return the list of all available model IDs."""
        model_ids = []
        for models in AVAILABLE_MODELS.values():
            model_ids.extend(models)
        return model_ids

    @staticmethod
    def get_provider_by_model_id(model_id: str) -> Optional[str]:
        """Return the provider name for a given model ID."""
        for provider, models in AVAILABLE_MODELS.items():
            if model_id in models:
                return provider
        return None

    def bind_tools(self, tools, **kwargs):
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)
