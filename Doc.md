# Gemini API Rotating Proxy: Comprehensive Documentation

## 1. Overview

This project provides a powerful, self-hosted proxy server that acts as a bridge to Google's Gemini API. It is designed to be a versatile and robust tool for developers, offering two key features:

1.  **OpenAI-Compatible Endpoint**: Emulates the OpenAI API, allowing you to use Google's advanced Gemini models with any tool, library, or framework designed for OpenAI (e.g., LangChain, LlamaIndex, custom scripts). This includes full support for **Function Calling**.
2.  **Credential Rotation**: The proxy can load and automatically rotate through multiple Google OAuth credentials. This allows you to distribute your API usage across different Google accounts, helping you manage rate limits and increase throughput.

It also provides a native Gemini endpoint for direct access. The server is built with FastAPI, uses Pydantic for robust configuration, and features a simple, one-time web UI to generate the necessary credentials.

## 2. Core Features

-   **Credential Rotation**: Automatically cycles through a pool of Google accounts to avoid rate limits.
-   **Dual API Support**: Use either OpenAI-compatible or native Gemini API formats.
-   **Function Calling**: Full support for Gemini's function calling on both endpoints.
-   **Easy Credential Generation**: A simple web UI to authorize the proxy and generate credentials for multiple Google accounts.
-   **Flexible Credential Loading**: Load credentials from local JSON files or directly from environment variables for stateless deployments.
-   **Robust Authentication**: Automatically refreshes expired tokens.
-   **API Key Security**: Protect your proxy endpoint with a simple password (bearer token).
-   **Streaming & Multimodality**: Full support for streaming responses and image inputs (via base64).
-   **Configurable Debugging**: Enable detailed logs to inspect requests, responses, and which credential was used.
-   **Docker Support**: Includes `Dockerfile` and `docker-compose.yml` for easy containerization.

## 3. Setup and Installation

Follow these steps to get the proxy server up and running.

### Prerequisites

-   Python 3.12+
-   A Google Cloud Project with the **"Cloud AI Companion API"** enabled.
-   An OAuth 2.0 Client ID for a **"Web application"** from your Google Cloud project.

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### Step 2: Install Dependencies

It is highly recommended to use a virtual environment.

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

Create a file named `.env` by copying the `.env.example` file. This file will hold your configuration.

```env
# --- Server Configuration ---
DOMAIN_NAME="http://localhost:7860"
PORT=7860

# --- Security ---
GEMINI_AUTH_PASSWORD="a-secure-password"

# --- Google OAuth Client (Optional) ---
# The application is pre-configured with a default OAuth client.
# You can override these by uncommenting the lines below.
# CLIENT_ID="your-google-client-id.apps.googleusercontent.com"
# CLIENT_SECRET="your-google-client-secret"

# --- Credential Loading (Choose one method) ---
# Method 1: File-based (Default)
# PERSISTENT_STORAGE_PATH="src/"

# Method 2: Environment Variable (for Stateless Deployments)
# CREDENTIALS_JSON_LIST='[]'

# --- Debugging ---
DEBUG=false
DEBUG_REDACT_LOGS=true
```

### Step 4: Generate Credentials

This proxy uses a simple web server to guide you through the Google OAuth flow. You need to run this process for **each Google account** you want to add to the rotation pool.

1.  **Start the credential generator:**
    ```bash
    uv run app.py --gen-creds
    ```

2.  **Open your browser** and navigate to `http://localhost:7860` (or your configured `DOMAIN_NAME`).

3.  You will see a login page. You can optionally enter a Google Cloud Project ID. If you leave it blank, the tool will try to discover it automatically.

4.  Click **"Login with Google"**. You will be redirected to Google's authentication screen.

5.  Log in with the Google account you want to use and grant the requested permissions.

6.  Upon success, a new `oauth_creds_{email}_{project}.json` file will be saved in the directory specified by `PERSISTENT_STORAGE_PATH` (e.g., `src/`).

7.  **Repeat this process** for every Google account you wish to add to the credential pool.

### Step 5: Run the Proxy Server

Once you have generated all your desired credentials, stop the generator (Ctrl+C) and start the main proxy server:

```bash
uv run app.py
```

The server will automatically detect and load all `oauth_creds_*.json` files and begin rotating through them for each incoming API request.

## 4. Authentication

All API endpoints require authentication. Use the password from `GEMINI_AUTH_PASSWORD` as a Bearer token.

```bash
curl -X POST http://localhost:7860/v1/chat/completions \
  -H "Authorization: Bearer your-secure-password" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

## 5. API Endpoints

### OpenAI-Compatible Endpoints (`/v1`)

#### Chat Completions

-   **Endpoint**: `POST /v1/chat/completions`
-   **Description**: The main endpoint for generating content. It supports streaming, function calling, and image inputs.
-   **Example (Function Calling)**:
    ```bash
    curl http://localhost:7860/v1/chat/completions \
      -H "Authorization: Bearer your-secure-password" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "gemini-1.5-pro-latest",
        "messages": [
          {"role": "user", "content": "What is the weather in Boston?"}
        ],
        "tools": [/* ... your function definitions ... */]
      }'
    ```

#### List Models

-   **Endpoint**: `GET /v1/models`
-   **Description**: Retrieves a list of available Gemini models in an OpenAI-compatible format.

### Native Gemini Endpoints (`/v1beta`)

These endpoints act as a direct proxy to the Google Gemini API.

-   **Endpoint**: `POST /v1beta/models/{model}:{action}`
-   **Example (`generateContent`)**:
    ```bash
    curl http://localhost:7860/v1beta/models/gemini-1.5-pro:generateContent \
      -H "Authorization: Bearer your-secure-password" \
      -H "Content-Type: application/json" \
      -d '{
        "contents": [{
          "role": "user",
          "parts": [{"text": "Explain quantum computing."}]
        }]
      }'
    ```

## 6. Configuration Details

Configuration is managed via Pydantic in `src/settings.py` and can be overridden with environment variables in a `.env` file.

| Variable                  | Description                                                                                                                                                              | Default                             |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------- |
| `DOMAIN_NAME`             | The public URL of your proxy, required for the OAuth redirect URI.                                                                                                       | `http://localhost:7860`             |
| `PORT`                    | The port on which the FastAPI server will run.                                                                                                                           | `7860`                              |
| `GEMINI_AUTH_PASSWORD`    | The password used to secure your proxy endpoints.                                                                                                                        | `a-secure-password`                 |
| `CLIENT_ID`               | Your Google Cloud OAuth 2.0 Client ID for a "Web application".                                                                                                         | `6812...` (public example)          |
| `CLIENT_SECRET`           | Your Google Cloud OAuth 2.0 Client Secret.                                                                                                                               | `GOCSPX-...` (public example)       |
| `PERSISTENT_STORAGE_PATH` | The directory where generated `oauth_creds_*.json` files are stored.                                                                                                     | `src/`                              |
| `CREDENTIALS_JSON_LIST`   | A JSON string containing an array of credential objects. If set, this overrides file-based loading. Ideal for stateless deployments.                                      | `[]` (empty list)                   |
| `DEBUG`                   | Set to `true` or `1` to enable verbose debug logging. Logs include credential details (Project ID, token snippet) and full request/response data.                      | `false`                             |
| `DEBUG_REDACT_LOGS`       | Set to `true` or `1` to redact sensitive text from debug logs. Only has an effect if `DEBUG` is also `true`.                                                              | `true`                              |

## 7. Verifying Credential Rotation

To confirm that the proxy is correctly rotating through your credentials, you can enable debug mode.

1.  Set `DEBUG=true` in your `.env` file.
2.  Restart the proxy server.
3.  Make several API requests.

In the server logs, you will now see a `--- Credential Details ---` block for each request, showing which Google Cloud Project and unique refresh token was used for the upstream call to Google's API.

```log
INFO: --- Credential Details ---
INFO: Project ID: gcp-project-one
INFO: Credential Used (Refresh Token ending in): ...a1b2c
INFO: --- Upstream Request to Google ---
...
INFO: --- Credential Details ---
INFO: Project ID: gcp-project-two
INFO: Credential Used (Refresh Token ending in): ...d3e4f
INFO: --- Upstream Request to Google ---
...
```

This output confirms that your requests are being distributed across your pool of credentials.

## 8. Deployment

### Docker

The project includes a `Dockerfile` and `docker-compose.yml` for easy containerization.

1.  **Build the Docker image:**
    ```bash
    docker build -t gemini-proxy .
    ```

2.  **Run the container:**
    ```bash
    docker run -d -p 7860:7860 --env-file .env --name gemini-proxy gemini-proxy
    ```

### Docker Compose

1.  **Configure your `.env` file** as described in the setup section.
2.  **Run Docker Compose:**
    ```bash
    docker-compose up -d
    ```

## 9. API Request Examples

### OpenAI-Compatible API

#### Basic Chat Completion

```bash
curl http://localhost:7860/v1/chat/completions \
  -H "Authorization: Bearer your-secure-password" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-1.5-pro-latest",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

#### Streaming Chat Completion

```bash
curl http://localhost:7860/v1/chat/completions \
  -H "Authorization: Bearer your-secure-password" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-1.5-pro-latest",
    "messages": [
      {"role": "user", "content": "Write a short story about a robot who discovers music."}
    ],
    "stream": true
  }'
```

### Native Gemini API

#### Generate Content

```bash
curl http://localhost:7860/v1beta/models/gemini-1.5-pro:generateContent \
  -H "Authorization: Bearer your-secure-password" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      {
        "role": "user",
        "parts": [
          {"text": "What are the main differences between Python and JavaScript?"}
        ]
      }
    ]
  }'
```

#### Streaming Generate Content

```bash
curl http://localhost:7860/v1beta/models/gemini-1.5-pro:streamGenerateContent?alt=sse \
  -H "Authorization: Bearer your-secure-password" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      {
        "role": "user",
        "parts": [
          {"text": "Write a poem about the ocean."}
        ]
      }
    ]
  }'
```
