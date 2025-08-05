from fastapi.responses import HTMLResponse

# --- UI & Styling ---
STYLE = """
<style>
    /* --- CSS Variables for Theming --- */
    :root {
        --primary-color: #007bff;
        --primary-hover: #0056b3;
        --secondary-color: #6c757d;
        --background-color: #f8f9fa;
        --container-bg: #ffffff;
        --text-color: #343a40;
        --text-muted: #6c757d;
        --border-color: #dee2e6;
        --success-bg: #d4edda;
        --success-text: #155724;
        --error-bg: #f8d7da;
        --error-text: #721c24;
        --code-bg: #e9ecef;
        --code-text: #212529;
        --shadow: 0 10px 30px rgba(0, 0, 0, 0.07);
        --border-radius: 12px;
        --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* --- Base & Body --- */
    body {
        font-family: var(--font-family);
        background-color: var(--background-color);
        color: var(--text-color);
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        margin: 0;
        padding: 1rem;
        box-sizing: border-box;
    }

    /* --- Main Container --- */
    .container {
        background: var(--container-bg);
        padding: 2.5rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        max-width: 650px;
        width: 100%;
        text-align: center;
        box-sizing: border-box;
    }

    /* --- Typography --- */
    h1 {
        color: var(--primary-color);
        font-weight: 700;
        font-size: 1.75rem;
        margin-bottom: 1rem;
    }
    p {
        line-height: 1.7;
        color: var(--text-muted);
        margin-bottom: 1.5rem;
    }
    strong {
        color: var(--primary-color);
    }

    /* --- Code Blocks & Snippets --- */
    .code-block {
        background-color: var(--code-bg);
        padding: 1rem;
        border-radius: 8px;
        text-align: left;
        color: var(--code-text);
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
        font-size: 0.9em;
        display: flex;
        justify-content: space-between;
        align-items: center;
        word-break: break-all;
        white-space: pre-wrap;
    }
    .code-block code {
        background: none;
        padding: 0;
    }
    code {
        background-color: var(--code-bg);
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        color: var(--code-text);
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
        font-size: 0.95em;
    }

    /* --- Buttons --- */
    .btn,
    input[type="submit"] {
        background: linear-gradient(145deg, var(--primary-color), #0056b3);
        color: white;
        padding: 0.9rem 1.75rem;
        border: none;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        text-decoration: none;
        margin-top: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(0, 123, 255, 0.2);
    }
    .btn:hover,
    input[type="submit"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 123, 255, 0.3);
    }
    .copy-btn {
        background: var(--secondary-color);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        cursor: pointer;
        font-size: 0.8rem;
        margin-left: 1rem;
        transition: background-color 0.2s ease;
        flex-shrink: 0; /* Prevents the button from shrinking */
    }
    .copy-btn:hover {
        background: #5a6268;
    }
    .copy-btn.copied {
        background: var(--success-text);
    }

    /* --- Forms --- */
    form {
        display: flex;
        flex-direction: column;
        margin-top: 2rem;
    }
    label {
        font-weight: 600;
        text-align: left;
        margin-bottom: 0.5rem;
        color: var(--text-color);
    }
    input[type="text"] {
        padding: 0.9rem;
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        font-size: 1rem;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    input[type="text"]:focus {
        outline: none;
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.15);
    }

    /* --- Status & Info Boxes --- */
    .status {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 99px;
        font-weight: 700;
        margin-bottom: 1.5rem;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .status.active {
        background-color: var(--success-bg);
        color: var(--success-text);
        border: 1px solid var(--success-text);
    }
    .status.inactive {
        background-color: var(--error-bg);
        color: var(--error-text);
        border: 1px solid var(--error-text);
    }
    .info-box {
        background-color: var(--background-color);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        text-align: left;
    }
    .info-box p {
        margin: 0;
    }
    .info-box strong {
        color: var(--text-color);
    }

    /* --- Footer --- */
    .footer {
        margin-top: 2rem;
        font-size: 0.9rem;
        color: var(--text-muted);
    }

    /* --- Responsive Design --- */
    @media (max-width: 768px) {
        body {
            padding: 0.5rem;
        }
        .container {
            padding: 1.5rem;
        }
        h1 {
            font-size: 1.5rem;
        }
        .code-block {
            flex-direction: column;
            align-items: flex-start;
        }
        .copy-btn {
            margin-top: 0.5rem;
            margin-left: 0;
            width: 100%;
        }
    }
</style>
"""

SCRIPT = """
<script>
    function copyToClipboard(elementId, button) {
        const text = document.getElementById(elementId).innerText;
        navigator.clipboard.writeText(text).then(() => {
            button.innerText = 'Copied!';
            button.classList.add('copied');
            setTimeout(() => {
                button.innerText = 'Copy';
                button.classList.remove('copied');
            }, 2000);
        }).catch(err => {
            console.error('Failed to copy text: ', err);
            button.innerText = 'Error';
        });
    }
</script>
"""

def create_page(title: str, body_content: str) -> HTMLResponse:
    """Creates a full HTML page with consistent styling and scripts."""
    return HTMLResponse(f"""
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{title}</title>
                {STYLE}
            </head>
            <body>
                <div class="container">
                    {body_content}
                </div>
                {SCRIPT}
            </body>
        </html>
    """)