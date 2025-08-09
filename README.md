# Gemini API Rotating Proxy

## 1. Overview

This project provides a high-performance, self-hosted proxy server that acts as a robust bridge to Google's Gemini API. It is designed to be a versatile tool for developers, offering key features:

1.  **Multi-API Compatibility**: Endpoints that emulate the OpenAI and Claude APIs, allowing you to use Google's advanced Gemini models with any tool, library, or framework designed for those ecosystems (e.g., LangChain, LlamaIndex). This includes full support for streaming and tool/function calling.
2.  **Dual Authentication Model**: The proxy intelligently uses the best authentication method for each service:
    -   **Rotating OAuth Credentials**: For generative models (e.g., `gemini-2.5-pro`), the proxy automatically rotates through multiple Google accounts to distribute API usage and manage rate limits.
    -   **Static API Key**: For embedding models, the proxy uses a dedicated Google Cloud API key, which is the required method for the public embedding API.

The server is built with FastAPI for high performance and uses Pydantic for robust, type-safe configuration. It includes a simple, one-time web UI to generate the necessary OAuth credentials for the generative models.

## 2. Core Features

-   **Dual Authentication**: Automatically uses rotating OAuth credentials for chat models and a static API key for embedding models.
-   **Credential Rotation**: Cycles through a pool of Google accounts for generative models to avoid rate limits. Invalid credentials are automatically skipped.
-   **Triple API Support**: Use OpenAI-compatible, Claude-compatible, or native Gemini API formats.
-   **Tool Calling**: Full support for Gemini's function calling, translated seamlessly from the OpenAI and Claude formats.
-   **Easy Credential Generation**: A simple web UI to authorize the proxy and generate OAuth credentials for multiple Google accounts.
-   **Flexible Credential Loading**: Load OAuth credentials from local JSON files or directly from a single environment variable for stateless deployments.
-   **Robust & Asynchronous**: Built with modern Python (`asyncio`, `httpx`) to handle many concurrent requests efficiently.
-   **API Key Security**: Protect your proxy endpoint with a simple password (bearer token).
-   **Configurable CORS**: Restrict access to your proxy from specific domains for enhanced security.
-   **Streaming & Multimodality**: Full support for streaming responses and image inputs (via base64).
-   **Docker Support**: Includes `Dockerfile` and `docker-compose.yml` for easy containerization.

---

## 3. Project Structure

The codebase is organized into functional modules to ensure clarity and maintainability.

```
gcli2api/
├── credentials/            # Default location for generated OAuth credentials.
├── src/
│   ├── api/                # FastAPI routers and web-layer components.
│   ├── adapters/           # Handles transformation between different API schemas.
│   ├── core/               # Core application logic (auth, credential management, etc.).
│   ├── models/             # Pydantic models for each API (Gemini, OpenAI, Claude).
│   ├── services/           # Encapsulates business logic (e.g., calling embedding APIs).
│   ├── tools/              # Standalone scripts, like the credential generator.
│   └── utils/              # Shared utilities like logging and constants.
├── app.py                  # Main application entrypoint.
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 4. Installation & Setup

Follow these steps to get the proxy server up and running.

### Prerequisites

-   Python 3.11+
-   [uv](https://github.com/astral-sh/uv) (recommended) or pip
-   Docker (for containerized deployment)
-   A Google Account. A Google Cloud Project is **not** strictly required for generative models, but is **required** to create an API key for embeddings.

### Step 1: Clone & Install Dependencies

```bash
# Clone the repository
git clone https://github.com/otwato/gcli2api.git
cd gcli2api

# Create a virtual environment and install dependencies with uv (recommended)
uv sync
```

### Step 2: Configure Environment

Create a `.env` file by copying the `.env.example` file.

```bash
cp .env.example .env
```

**Edit your `.env` file and set the following variables:**

1.  `GEMINI_AUTH_PASSWORD`: Set a secure password to protect your proxy.
2.  `EMBEDDING_GEMINI_API_KEY`: **This is required if you plan to use embedding endpoints.**
    -   Go to the Google Cloud Console -> "APIs & Services" -> "Credentials".
    -   Click "+ CREATE CREDENTIALS" and select "API key".
    -   Copy the key and paste it here. It is highly recommended to restrict the key to the "Generative Language API".

```env
GEMINI_AUTH_PASSWORD="your-super-secret-password"
EMBEDDING_GEMINI_API_KEY="your-google-cloud-api-key"
```

### Step 3: Generate OAuth Credentials (For Chat Models)

This proxy uses a simple web server to guide you through the Google OAuth flow. This is only required for **generative/chat models**. You must run this process for **each Google account** you want to add to the rotation pool.

1.  **Start the credential generator:**
    ```bash
    uv run app.py --gen-creds
    ```

2.  **Open your browser** and navigate to `http://localhost:7860`.

3.  You will see a login page. You can optionally enter a specific Google Cloud Project ID, but it's recommended to leave it blank to let the tool discover it automatically.

4.  Click **"Login with Google & Generate Credential"**. You will be redirected to Google's authentication screen.

5.  Log in with the Google account you want to use and grant the requested permissions.

6.  Upon success, a new `oauth_creds_{email}_{project}.json` file will be saved in the `credentials/` directory at the project root.

7.  **Repeat this process** for every Google account you wish to add to the credential pool. Each successful login will create a new credential file.

### Step 4: Run the Proxy Server

Once you have configured your `.env` file and generated any desired OAuth credentials, stop the generator (`Ctrl+C`) and start the main proxy server:

```bash
# Run with uv
uv run app.py

# Or with uvicorn directly
uv run uvicorn src.main:app --host 0.0.0.0 --port 7860
```

The server will automatically detect and load all `oauth_creds_*.json` files from the `credentials/` directory and begin rotating through them for chat requests.

---

## 5. Deployment with Docker

Using Docker is the recommended way to run the proxy in production.

### Method 1: Docker Compose (Easiest)

1.  **Generate OAuth Credentials First**: Complete Step 3 from the local installation to generate your `oauth_creds_*.json` files inside the `credentials/` directory.
2.  **Configure `.env`**: Ensure your `.env` file is configured with `GEMINI_AUTH_PASSWORD` and `EMBEDDING_GEMINI_API_KEY`.
3.  **Run Docker Compose**:
    ```bash
    docker-compose up -d --build
    ```
    The service will be available at `http://localhost:7860`. The `docker-compose.yml` file is configured to mount the local `./credentials` directory and read the `.env` file.

### Method 2: Manual Docker Build

This method is useful for stateless deployments where you provide OAuth credentials via an environment variable.

1.  **Consolidate OAuth Credentials**: If you have multiple credential files, you can combine their contents into a single JSON array.
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

## 6. API Usage

### Authentication

All API endpoints require a Bearer token. Use the password you set for `GEMINI_AUTH_PASSWORD`.

**Header:** `Authorization: Bearer your-super-secret-password`

### OpenAI-Compatible Endpoint (`/v1`)

-   **Endpoints**:
    -   `POST /v1/chat/completions` (Uses rotating OAuth credentials)
    -   `POST /v1/embeddings` (Uses `EMBEDDING_GEMINI_API_KEY`)
    -   `GET /v1/models`

### Claude-Compatible Endpoint (`/v1`)

-   **Endpoints**:
    -   `POST /v1/messages` (Uses rotating OAuth credentials)

### Native Gemini Endpoint (`/v1beta`)

-   **Endpoint**: `POST /v1beta/models/{model}:{action}`
    -   `{model}`: e.g., `gemini-2.5-pro` or `gemini-embedding-001`
    -   `{action}`: e.g., `generateContent`, `streamGenerateContent`, `embedContent`
    -   Authentication depends on the action: `generateContent` uses OAuth, while `embedContent` uses the API key.

---

## 7. Environment Variables

All configuration is managed via environment variables, loaded from the `.env` file.

| Variable                     | Description                                                                                                                              | Default                             |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| `PORT`                       | The port on which the FastAPI server will run.                                                                                           | `7860`                              |
| `DOMAIN_NAME`                | The public URL of your proxy, required for the OAuth redirect URI.                                                                       | `http://localhost:7860`             |
| `CORS_ALLOWED_ORIGINS`       | JSON list of allowed origins for CORS. **Set this for production.**                                                                      | `["*"]`                            |
| `GEMINI_AUTH_PASSWORD`       | **Required.** The password used to secure your proxy endpoints.                                                                          | `123456`                            |
| `EMBEDDING_GEMINI_API_KEY`   | **Required for embeddings.** An API key from Google Cloud for the public embedding API.                                                  | `""` (empty string)                 |
| `CLIENT_ID`                  | Your Google Cloud OAuth 2.0 Client ID. Overrides the public default.                                                                     | A public Google client ID           |
| `CLIENT_SECRET`              | Your Google Cloud OAuth 2.0 Client Secret. Overrides the public default.                                                                 | A public Google client secret       |
| `PERSISTENT_STORAGE_PATH`    | The directory where generated `oauth_creds_*.json` files are stored.                                                                     | `credentials/`                      |
| `CREDENTIALS_JSON_LIST`      | A JSON string containing an array of credential objects. If set, this overrides file-based loading.                                      | `""` (empty string)                 |
| `DEBUG`                      | Set to `true` to enable verbose debug logging.                                                                                           | `false`                             |
| `DEBUG_REDACT_LOGS`          | Set to `true` to redact sensitive text from debug logs.                                                                                  | `false`                             |
