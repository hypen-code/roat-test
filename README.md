# Agno Docs Chatbot on Roar

A small FastAPI chatbot that connects to the Agno documentation MCP server. It uses `qwen3.8-27b` to retrieve documentation through MCP because Roar reports tool calling for that model, then uses the requested `qwen3.8-27b-lk` model to write the answer without tool calls. This works around the LK endpoint rejecting `tool_choice="auto"`. Assistant answers are rendered as sanitized Markdown in the glass-style chat UI.

## Run locally

Python 3.11 or newer is recommended.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ROAR_API_KEY="your Roar inference key"
export ROAR_BASE_URL="https://api.roar-ai.com/v1"
uvicorn main:app --reload --port 8000
```

Open <http://localhost:8000>. `ROAR_BASE_URL` is optional if you use the default. For local runs, create a Roar key with the `inference` scope. On Roar hosting, the platform injects `ROAR_API_KEY` and `ROAR_BASE_URL` automatically.

## Deploy on Roar

1. Push this folder to a GitHub repository.
2. In Roar, connect the GitHub account that can access that repository.
3. Create a new application from the repository. Choose the repository root as the app root and leave the database option off; this chatbot keeps conversation history in the browser.
4. Wait for the deployment to finish. Roar builds and starts the app using `Procfile` and supplies the AI gateway credentials.
5. Open the application's `https://…roarai.app` URL. The health endpoint is at `/health`.

For a public URL, set `CHATBOT_ACCESS_TOKEN` in the app's Environment settings before sharing it. Use a long random value and enter it once in the browser prompt. Without this optional setting, anyone who can reach the app can send messages that incur model usage on your Roar account. Never put `ROAR_API_KEY` in frontend code.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_ID` | `qwen3.8-27b-lk` | Roar model name |
| `MCP_MODEL_ID` | `qwen3.8-27b` | Model used for MCP tool calls; set this to another Roar model that supports tool calling if needed |
| `AGNO_MCP_URL` | `https://docs.agno.com/mcp` | Agno documentation MCP endpoint |
| `CHATBOT_ACCESS_TOKEN` | unset | Optional shared access token for the web app |

Agno's MCP integration uses Streamable HTTP. Roar's app-provided AI key and gateway URL are read only on the server. Each chat turn makes a docs-lookup call and an answer call. The app does not persist chat history after the browser session is cleared..
