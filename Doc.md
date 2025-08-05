# Gemini API Rotating Proxy

## 1. Overview

This project provides a high-performance, self-hosted proxy server that acts as a robust bridge to Google's Gemini API. It is designed to be a versatile tool for developers, offering two key features:

1.  **OpenAI-Compatible Endpoint**: Emulates the OpenAI API, allowing you to use Google's advanced Gemini models with any tool, library, or framework designed for OpenAI (e.g., LangChain, LlamaIndex). This includes full support for streaming and tool/function calling.
2.  **Credential Rotation**: The proxy can load and automatically rotate through multiple Google OAuth credentials. This allows you to distribute your API usage across different Google accounts, helping you manage rate limits and increase throughput.

The server is built with FastAPI for high performance and uses Pydantic for robust, type-safe configuration. It includes a simple, one-time web UI to generate the necessary credentials.

## 2. Core Features

-   **Credential Rotation**: Automatically cycles through a pool of Google accounts to avoid rate limits. Invalid credentials (due to expired refresh tokens) are automatically skipped.
-   **Dual API Support**: Use either OpenAI-compatible or native Gemini API formats.
-   **Tool Calling**: Full support for Gemini's function calling, translated seamlessly from the OpenAI format.
-   **Easy Credential Generation**: A simple web UI to authorize the proxy and generate credentials for multiple Google accounts.
-   **Flexible Credential Loading**: Load credentials from local JSON files or directly from a single environment variable for stateless deployments.
-   **Robust & Asynchronous**: Built with modern Python (`asyncio`, `httpx`) to handle many concurrent requests efficiently.
-   **API Key Security**: Protect your proxy endpoint with a simple password (bearer token).
-   **Configurable CORS**: Restrict access to your proxy from specific domains for enhanced security.
-   **Streaming & Multimodality**: Full support for streaming responses and image inputs (via base64).
-   **Docker Support**: Includes `Dockerfile` and `docker-compose.yml` for easy containerization.

---

## 3. Installation & Setup

Follow these steps to get the proxy server up and running.

### Prerequisites

-   Python 3.11+
-   [uv](https://github.com/astral-sh/uv) (recommended) or pip
-   Docker (for containerized deployment)
-   A Google Account. A Google Cloud Project is **not** strictly required, as the tool can often discover a default project for you.

### Step 1: Clone & Install Dependencies

```bash
# Clone the repository
git clone https://github.com/otwato/gcli2api.git
cd gcli2api

# Create a virtual environment and install dependencies with uv (recommended)
uv venv
uv sync
```

### Step 2: Configure Environment

Create a `.env` file by copying the `.env.example` file. The only variable you **must** change is `GEMINI_AUTH_PASSWORD`.

```bash
cp .env.example .env
```

**Edit `.env` and set a secure password:**

```env
GEMINI_AUTH_PASSWORD="your-super-secret-password"
```

### Step 3: Generate Credentials

This proxy uses a simple web server to guide you through the Google OAuth flow. You must run this process for **each Google account** you want to add to the rotation pool.

1.  **Start the credential generator:**
    ```bash
    uv run app.py --gen-creds
    ```

2.  **Open your browser** and navigate to `http://localhost:7860`.

3.  You will see a login page. You can optionally enter a specific Google Cloud Project ID, but it's recommended to leave it blank to let the tool discover it automatically.

4.  Click **"Login with Google & Generate Credential"**. You will be redirected to Google's authentication screen.

5.  Log in with the Google account you want to use and grant the requested permissions.

6.  Upon success, a new `oauth_creds_{email}_{project}.json` file will be saved in the `src/` directory.

7.  **Repeat this process** for every Google account you wish to add to the credential pool. Each successful login will create a new credential file.

### Step 4: Run the Proxy Server

Once you have generated all your desired credentials, stop the generator (`Ctrl+C`) and start the main proxy server:

```bash
# Run with uv
uv run app.py

# Or with uvicorn directly
uvicorn src.main:app --host 0.0.0.0 --port 7860
```

The server will automatically detect and load all `oauth_creds_*.json` files from the `src/` directory and begin rotating through them for each incoming API request.

---

## 4. Deployment with Docker

Using Docker is the recommended way to run the proxy in production.

### Method 1: Docker Compose (Easiest)

1.  **Generate Credentials First**: Complete Step 3 from the local installation to generate your `oauth_creds_*.json` files inside the `src/` directory.
2.  **Configure `.env`**: Ensure your `.env` file is configured, especially `GEMINI_AUTH_PASSWORD`.
3.  **Run Docker Compose**:
    ```bash
    docker-compose up -d --build
    ```
    The service will be available at `http://localhost:7860`.

### Method 2: Manual Docker Build

This method is useful for stateless deployments where you provide credentials via an environment variable.

1.  **Consolidate Credentials**: If you have multiple credential files, you can combine their contents into a single JSON array.
2.  **Set `CREDENTIALS_JSON_LIST`**: In your `.env` file, paste the JSON array into the `CREDENTIALS_JSON_LIST` variable. This method **overrides** file-based loading.
    ```env
    CREDENTIALS_JSON_LIST='[{"client_id":...}, {"client_id":...}]'
    ```
3.  **Build and Run the container**:
    ```bash
    # Build the image
    docker build -f Build-Dockerfile -t gcli-proxy .

    # Run the container, passing the .env file
    docker run -d -p 7860:7860 --env-file .env --name gcli-proxy gcli-proxy
    ```

---

## 5. API Usage

### Authentication

All API endpoints require a Bearer token. Use the password you set for `GEMINI_AUTH_PASSWORD`.

**Header:** `Authorization: Bearer your-super-secret-password`

### OpenAI-Compatible Endpoint (`/v1`)

This endpoint mimics the OpenAI API.

-   **Endpoint**: `POST /v1/chat/completions`
-   **Example Request (with Tool Calling)**:

    ```json
    {
      "model": "gemini-1.5-pro-latest", // Or any other supported model
      "messages": [
        {"role": "user", "content": "What is the weather like in Boston?"}
      ],
      "tools": [
        {
          "type": "function",
          "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
              "type": "object",
              "properties": {
                "location": {
                  "type": "string",
                  "description": "The city and state, e.g. San Francisco, CA"
                }
              },
              "required": ["location"]
            }
          }
        }
      ],
      "tool_choice": "auto"
    }
    ```

### Native Gemini Endpoint (`/v1beta`)

This endpoint provides direct access to the underlying Google API format.

-   **Endpoint**: `POST /v1beta/models/{model}:{action}`
    -   `{model}`: e.g., `gemini-1.5-pro-latest`
    -   `{action}`: e.g., `generateContent` or `streamGenerateContent`
-   **Example Request**:

    ```json
    {
      "contents": [
        {
          "role": "user",
          "parts": [
            {"text": "Explain quantum computing in simple terms."}
          ]
        }
      ],
      "safetySettings": [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
      ]
    }
    ```

---

## 6. Environment Variables

All configuration is managed via environment variables, loaded from the `.env` file.

| Variable                  | Description                                                                                                                              | Default                             |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| `PORT`                    | The port on which the FastAPI server will run.                                                                                           | `7860`                              |
| `DOMAIN_NAME`             | The public URL of your proxy, required for the OAuth redirect URI.                                                                       | `http://localhost:7860`             |
| `CORS_ALLOWED_ORIGINS`    | JSON list of allowed origins for CORS. **Set this for production.**                                                                      | `["*"]`                            |
| `GEMINI_AUTH_PASSWORD`    | **Required.** The password used to secure your proxy endpoints.                                                                          | `123456`                            |
| `CLIENT_ID`               | Your Google Cloud OAuth 2.0 Client ID. Overrides the public default.                                                                     | A public Google client ID           |
| `CLIENT_SECRET`           | Your Google Cloud OAuth 2.0 Client Secret. Overrides the public default.                                                                 | A public Google client secret       |
| `PERSISTENT_STORAGE_PATH` | The directory where generated `oauth_creds_*.json` files are stored.                                                                     | `src/`                              |
| `CREDENTIALS_JSON_LIST`   | A JSON string containing an array of credential objects. If set, this overrides file-based loading.                                      | `""` (empty string)                 |
| `THOUGHT_WRAPPER_TAGS`    | A JSON list of two strings to wrap 'thought' outputs from the model, e.g., `'["<t>", "</t>"]'`.                                        | `[]` (empty list)                   |
| `DEBUG`                   | Set to `true` to enable verbose debug logging.                                                                                           | `false`                             |
| `DEBUG_REDACT_LOGS`       | Set to `true` to redact sensitive text from debug logs.                                                                                  | `false`                             |