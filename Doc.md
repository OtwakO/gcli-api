# Gemini API Proxy Documentation

## 1. Overview

This project provides a powerful proxy server that acts as a bridge to Google's Gemini API. It is designed to be a versatile tool for developers, offering two main modes of operation:

1.  **OpenAI-Compatible Endpoint**: Emulates the OpenAI API, allowing you to use Google's Gemini models with any tool or library designed for OpenAI (e.g., LangChain, custom scripts). This includes full support for **Function Calling**.
2.  **Native Gemini Endpoint**: Provides a direct, pass-through proxy to the native Google Gemini API for developers who want to use its full feature set, including native function calling.

The server is built with FastAPI, uses Pydantic for robust configuration management, and features a simple web-based OAuth2 flow for initial authentication. It is designed for easy deployment and configuration.

## 2. Features

-   **Dual API Support**: Use either OpenAI-compatible or native Gemini API formats.
-   **Function Calling**: Full support for Gemini's function calling capabilities on both the OpenAI-compatible and native endpoints.
-   **Web-Based Authentication**: Simple, one-time browser-based login to authorize the proxy with your Google account.
-   **Robust Credential Handling**: Automatically refreshes expired tokens and can persist credentials in a local `oauth_creds.json` file or be configured statelessly via environment variables.
-   **API Key/Password Security**: Protect your proxy endpoint with a simple password.
-   **Environment Variable Configuration**: Easily configure all settings using a `.env` file.
-   **Streaming Support**: Full support for streaming responses for both API types.
-   **Response Validation**: Enforces response models for non-streaming OpenAI endpoints to ensure data integrity.
-   **Configurable Debug Logging**: Enable detailed logs and control whether sensitive data is redacted.
-   **Centralized Error Handling**: Consistent and informative JSON error responses.
-   **Docker Support**: Comes with `Dockerfile` and `docker-compose.yml` for easy containerized deployment.

## 3. Setup and Installation

Follow these steps to get the proxy server up and running.

### Prerequisites

-   Python 3.12+
-   A Google Cloud Project with the "Cloud AI Companion API" enabled.
-   OAuth 2.0 Client IDs for a "Web application" from your Google Cloud project.

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### Step 2: Install Dependencies

It is recommended to use a virtual environment.

```bash
uv sync
```

### Step 3: Configure Environment Variables

Create a file named `.env` in the root directory of the project. This file will hold your configuration. You can copy `.env.example` to get started.

```env
# Your Google Cloud OAuth Client ID and Secret
# (Must be a "Web application" type client)
CLIENT_ID="your-google-client-id.apps.googleusercontent.com"
CLIENT_SECRET="your-google-client-secret"

# The public domain name where this proxy will be hosted.
# For local development, this is typically http://localhost:7860
DOMAIN_NAME="http://localhost:7860"

# A simple password to protect access to your proxy.
# This will be used as a Bearer token or Basic Auth password.
GEMINI_AUTH_PASSWORD="a-secure-password"

# (Optional) The port the server will run on.
# PORT=7860

# (Optional) Your Google Cloud Project ID.
# If not set, the proxy will attempt to discover it automatically.
# GOOGLE_CLOUD_PROJECT="your-gcp-project-id"

# (Optional) For stateless deployments, you can paste the entire
# content of your oauth_creds.json file here.
# OAUTH_CREDS_JSON='{"client_id": "...", "client_secret": "..."}'

# (Optional) Set to `true` or `1` to enable debug logging.
# This shows the request object being sent to Google's API.
# By default, logs are NOT redacted when DEBUG is true.
# DEBUG=False

# (Optional) Set to `true` or `1` to redact sensitive content in debug logs.
# Only has an effect if DEBUG is also enabled.
# DEBUG_REDACT_LOGS=False
```

### Step 4: Run the Server

Start the server using `uvicorn`.

```
uv run app.py
```

### Step 5: Initial Authentication

1.  When you first run the server, the console will log a warning that the proxy is not authenticated.
2.  Open your web browser and navigate to the URL specified in your `DOMAIN_NAME` setting (e.g., `http://localhost:7860`).
3.  Click the "Click here to log in" link, which will redirect you to the Google authentication screen.
4.  Log in with your Google account and grant the requested permissions.
5.  After successful authentication, you will be redirected back to the proxy, and a file named `oauth_creds.json` will be created in the `src/` directory.

The server is now authenticated and ready to accept API requests.

## 4. Authentication

All API endpoints (except `/`, `/health`, `/login`, and `/oauth2callback`) require authentication. You can authenticate your requests in one of the following ways, using the password defined in `GEMINI_AUTH_PASSWORD`.

### Bearer Token (Recommended)

```bash
curl -X POST http://localhost:7860/v1/chat/completions \
  -H "Authorization: Bearer your-secure-password" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

### Basic Authentication

```bash
curl -X POST http://localhost:7860/v1/chat/completions \
  -u "user:your-secure-password" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

### Query Parameter

```bash
curl -X POST "http://localhost:7860/v1/chat/completions?key=your-secure-password" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

### `x-goog-api-key` Header

```bash
curl -X POST http://localhost:7860/v1/chat/completions \
  -H "x-goog-api-key: your-secure-password" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

## 5. API Endpoints

### OpenAI-Compatible Endpoints

These endpoints are prefixed with `/v1`.

#### List Models

-   **Endpoint**: `GET /v1/models`
-   **Description**: Retrieves a list of available Gemini models in a format compatible with the OpenAI API.
-   **Example Request**:
    ```bash
    curl http://localhost:7860/v1/models \
      -H "Authorization: Bearer your-secure-password"
    ```

#### Chat Completions

-   **Endpoint**: `POST /v1/chat/completions`
-   **Description**: Generates a model response for a given chat conversation. Supports both standard and streaming requests, as well as **Function Calling**.
-   **Example Request (Non-Streaming)**:
    ```bash
    curl http://localhost:7860/v1/chat/completions \
      -H "Authorization: Bearer your-secure-password" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "gemini-1.5-pro",
        "messages": [
          {"role": "user", "content": "Hello, what is the capital of France?"}
        ]
      }'
    ```
-   **Example Request (Function Calling)**:
    ```bash
    curl http://localhost:7860/v1/chat/completions \
      -H "Authorization: Bearer your-secure-password" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "gemini-2.5-flash",
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
                  },
                  "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
              }
            }
          }
        ]
      }'
    ```

### Native Gemini Endpoints

These endpoints are prefixed with `/v1beta` and act as a direct proxy to the Google Gemini API.

#### List Models

-   **Endpoint**: `GET /v1beta/models`
-   **Description**: Retrieves a list of available models in the native Gemini API format.
-   **Example Request**:
    ```bash
    curl http://localhost:7860/v1beta/models \
      -H "Authorization: Bearer your-secure-password"
    ```

#### Generate Content (and other actions)

The proxy forwards any request from `/v1beta/` to the corresponding Google API endpoint.

-   **Endpoint**: `POST /v1beta/models/{model}:{action}`
-   **Description**: Performs an action on a model. Common actions are `generateContent` and `streamGenerateContent`.
-   **Example Request (`generateContent`)**:
    ```bash
    curl http://localhost:7860/v1beta/models/gemini-1.5-pro:generateContent \
      -H "Authorization: Bearer your-secure-password" \
      -H "Content-Type: application/json" \
      -d '{
        "contents": [{
          "role": "user",
          "parts": [{"text": "Explain quantum computing in simple terms."}]
        }]
      }'
    ```
-   **Example Request (`streamGenerateContent`)**:
    ```bash
    curl http://localhost:7860/v1beta/models/gemini-1.5-pro:streamGenerateContent \
      -H "Authorization: Bearer your-secure-password" \
      -H "Content-Type: application/json" \
      -d '{
        "contents": [{
          "role": "user",
          "parts": [{"text": "Tell me a long story about space exploration."}]
        }]
      }'
    ```

## 6. Supported Models

The proxy supports a wide range of Gemini models. The exact list can be retrieved from the `/v1/models` or `/v1beta/models` endpoints. As of the latest update, supported models include:

| Model Name (OpenAI ID)          | Description                                          |
| ------------------------------- | ---------------------------------------------------- |
| `gemini-2.5-pro-preview-05-06`  | Preview version of Gemini 2.5 Pro from May 6th       |
| `gemini-2.5-pro-preview-06-05`  | Preview version of Gemini 2.5 Pro from June 5th      |
| `gemini-2.5-pro`                | Advanced multimodal model with enhanced capabilities |
| `gemini-2.5-flash-preview-05-20`| Preview version of Gemini 2.5 Flash from May 20th   |
| `gemini-2.5-flash`              | Fast and efficient multimodal model                  |
| `gemini-embedding-001`          | Text embedding model for semantic search             |

*Note: This list is subject to change. Always query the API for the most up-to-date models.*

## 7. Configuration Details

Configuration is managed via Pydantic in `src/settings.py` and can be overridden with environment variables in a `.env` file.

| Variable                 | Description                                                                                                                            | Default                                     |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `CLIENT_ID`              | Your Google Cloud OAuth 2.0 Client ID.                                                                                                 | `6812...apps.googleusercontent.com`         |
| `CLIENT_SECRET`          | Your Google Cloud OAuth 2.0 Client Secret.                                                                                             | `GOCSPX-...`                                |
| `GOOGLE_CLOUD_PROJECT`   | Your specified Google Cloud Project ID. If not provided, the application will attempt to discover it automatically.                                | `""` (empty string)                         |
| `PORT`                   | The port on which the FastAPI server will run.                                                                                         | `7860`                                      |
| `DOMAIN_NAME`            | The public URL of your proxy. Used for the OAuth redirect URI.                                                                         | `http://localhost:7860`                     |
| `GEMINI_AUTH_PASSWORD`   | The password used to secure your proxy endpoints.                                                                                      | `123456`                                    |
| `OAUTH_CREDS_JSON`       | A JSON string containing the entire content of `oauth_creds.json`. Useful for stateless deployments.                                     | `""` (empty string)                         |
| `DEBUG`                  | Set to `true` or `1` to enable debug logging.                                                                                          | `false`                                     |
| `DEBUG_REDACT_LOGS`      | Set to `true` or `1` to redact sensitive content in debug logs. Only has an effect if `DEBUG` is also enabled.                           | `false`                                     |

## 8. Project Structure

The `src` directory contains the core application logic:

-   `main.py`: The main FastAPI application file. Initializes the app, middleware, and API routers.
-   `settings.py`: Pydantic-based configuration management.
-   `constants.py`: Holds static values like supported models and API scopes.
-   `auth.py`: Handles all authentication, authorization, and the OAuth2 flow.
-   `google_api_client.py`: A client for communicating with the backend Google Gemini API.
-   `openai_routes.py`: Defines the OpenAI-compatible API endpoints.
-   `gemini_routes.py`: Defines the native Gemini API proxy endpoints.
-   `openai_transformers.py`: Logic for converting data between OpenAI and Gemini formats.
-   `models.py`: Pydantic models for request and response validation.
-   `utils.py`: Helper functions used across the application.
-   `oauth_creds.json`: (Created after login) Stores the persistent OAuth credentials.
