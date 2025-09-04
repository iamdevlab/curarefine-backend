# In app/services/external_ai_service.py
import pandas as pd
from typing import Dict, List, Any
import json
import httpx


class ExternalAIServiceError(Exception):
    def __init__(self, message, status_code=500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


# --- The Adapter Function (The "Translator") ---
# This function acts as a "translator". Its only job is to take a standard
# set of inputs (like prompts and a model ID) and convert them into the
# specific JSON payload format that each different AI provider expects.
# This keeps the main service function clean and makes it easy to add new providers.
def create_normalized_request(
    provider: str, system_prompt: str, user_prompt: str, model_id: str
):
    """
    Takes a standard set of inputs and translates them into the specific
    payload format required by different AI providers.
    """
    # A dictionary of "adapter" lambda functions. Each key is a provider, and the
    # value is a function that returns the correctly structured dictionary (payload).
    adapters = {
        "anthropic": lambda: {
            "model": model_id or "claude-3-sonnet-20240229",
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        },
        "cohere": lambda: {
            "model": model_id,
            "chat_history": [{"role": "SYSTEM", "message": system_prompt}],
            "message": user_prompt,
        },
        "google": lambda: {
            "contents": [{"parts": [{"text": f"{system_prompt}\n{user_prompt}"}]}]
        },
        "ollama": lambda: {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": "json",
            "stream": False,
        },
        # This is the default format for any OpenAI-compatible API.
        # It's used for OpenAI itself, Mistral, Deepseek, and self-hosted models.
        "default_openai": lambda: {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        },
    }

    # List of providers that are known to use the OpenAI-compatible format.
    openai_compatible = [
        "openai",
        "deepseek",
        "fireworksai",
        "openrouter",
        "mistral",
        "self-hosted",
    ]

    if provider in openai_compatible:
        # If the provider is in our list, use the default OpenAI adapter.
        return adapters["default_openai"]()

    # Otherwise, find the provider's specific adapter in the dictionary.
    # If it's not found, fall back to the OpenAI format as a safe default.
    return adapters.get(provider, adapters["default_openai"])()


# --- Main Service Function ---
# This function orchestrates the entire process of getting AI recommendations.
# It prepares the data, calls the adapter to get a formatted payload, sets the
# correct headers, and makes the API call.
async def generate_external_recommendations(
    df: pd.DataFrame, settings: Dict[str, Any]
) -> List[Dict[str, Any]]:
    # 1. Extract settings from the input dictionary.
    # The 'final_endpoint_url' is now expected to be provided from the database logic.
    provider = settings.get("provider")
    api_key = settings.get("api_key")
    model_id = settings.get("model_id")
    endpoint_url = settings.get("final_endpoint_url")

    if not endpoint_url:
        print(f"No endpoint URL available for provider '{provider}'.")
        return []

    # 2. Handle Google's unique URL structure.
    # Google requires the model and API key to be part of the URL itself.
    if provider == "google":
        endpoint_url = f"{endpoint_url}/{model_id}:generateContent?key={api_key}"

    # 3. Prepare the data summary and prompts.
    # This section converts the DataFrame into a compact JSON format for the AI.
    df_head = df.head().to_json(orient="split")
    df_info = {
        "columns": df.columns.tolist(),
        "dtypes": {c: str(d) for c, d in df.dtypes.items()},
        "missing_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
    }
    system_prompt = "You are an expert data cleaning assistant. Analyze the provided dataset and suggest specific cleaning operations. Provide your response as a JSON object with a 'recommendations' key containing an array of recommendation objects. Each recommendation should include: 'column' (if applicable), 'operation', 'parameters' (if needed), and 'reason'."
    user_prompt = f"Data Summary: {json.dumps(df_info)}\nData Head: {df_head}\nProvide cleaning recommendations in the specified JSON format."

    # 4. Call the adapter to get the correctly formatted payload.
    # This single line replaces the previous large if/elif/else block.
    payload = create_normalized_request(
        provider=provider,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_id=model_id,
    )

    # 5. Set the appropriate HTTP headers for the target provider.
    headers = {"Content-Type": "application/json"}
    if provider == "anthropic":
        # Anthropic has its own unique set of required headers.
        headers.update({"x-api-key": api_key, "anthropic-version": "2023-06-01"})
    elif provider not in ["ollama", "google"]:
        # Most other providers use a standard Bearer token for authentication.
        headers["Authorization"] = f"Bearer {api_key}"

    # 6. Make the API call and process the response.
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            print(f"Connecting to {provider} at: {endpoint_url}")
            response = await client.post(endpoint_url, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()

            # 7. Parse the response to find the content.
            # This logic is still needed because each provider returns the AI's message
            # in a different JSON structure.
            response_content = None
            openai_compatible = [
                "openai",
                "deepseek",
                "fireworksai",
                "openrouter",
                "mistral",
                "self-hosted",
            ]

            if provider in openai_compatible:
                response_content = (
                    response_data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content")
                )
            elif provider == "ollama":
                response_content = response_data.get("message", {}).get("content")
            elif provider == "google":
                response_content = (
                    response_data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text")
                )
            elif provider == "cohere":
                response_content = response_data.get("text")
            elif provider == "anthropic":
                response_content = response_data.get("content", [{}])[0].get("text")

            if not response_content:
                return []

            # 8. Clean and load the final JSON output.
            # Some models wrap their JSON output in markdown code blocks.
            if response_content.strip().startswith("```json"):
                response_content = response_content.strip()[7:-3].strip()

            raw_recommendations = json.loads(response_content).get(
                "recommendations", []
            )

            # --- FIX: Standardize the response before sending it to the frontend ---
            standardized_recommendations = []
            for rec in raw_recommendations:
                # The external AI might use 'operation' instead of 'type'. We map it here.
                if "operation" in rec and "type" not in rec:
                    rec["type"] = rec.pop("operation")

                # Ensure that every recommendation has a 'type' field before we accept it.
                if "type" in rec:
                    standardized_recommendations.append(rec)

            return standardized_recommendations
            # --- END FIX ---

    except httpx.RequestError as e:
        print(f"Network error calling {provider}: {e}")
        return []
    except httpx.HTTPStatusError as e:
        error_message = f"External AI provider '{provider}' failed with status {e.response.status_code}: {e.response.text}"
        print(error_message)
        raise ExternalAIServiceError(
            message=error_message, status_code=e.response.status_code
        )

    except Exception as e:
        # Also raise for other errors
        error_message = f"Error processing response from '{provider}': {e}"
        print(error_message)
        raise ExternalAIServiceError(message=error_message)
