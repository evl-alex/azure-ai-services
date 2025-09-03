# Azure AI Image Analysis (Python)

This repo contains a simple Python script (`image-analysis.py`) that uses Azure AI Vision Image Analysis to caption images, extract tags, detect objects and people, and optionally save an annotated image with detected people.

## Prerequisites
- Python 3
- An Azure AI Vision resource obtained from project's resource details in Azure portal
  - Endpoint (e.g., https://<your-resource>.cognitiveservices.azure.com/)
  - Key

## Setup
1. Clone or open this folder in VS Code/terminal.
2. (Recommended) Create and activate a virtual environment:
   
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables. Create a `.env` file in the project root with:
   
   ```env
   VISION_ENDPOINT=<your-endpoint>
   VISION_KEY=<your-key>
   ```

## Usage
1. Specify features in `visual_features`
2. Run the script with a local image path as the first argument:

```bash
python image-analysis.py img/image.jpg
```

What it does:
- Loads your Azure Vision endpoint/key from `.env`
- Sends the image for analysis (text, captions, tags, objects, people)
- Prints results to the terminal
- If any people are detected (confidence > 0.2), saves an annotated image to `img/results/people.jpg`

## Output
- Console output: text, caption, dense captions, tags, and objects with confidences
- Annotated image: If people detected, saved to `img/results/people.jpg` (directory created if missing)

## Troubleshooting
- File not found: Ensure the image path exists (relative to repo root or absolute path)
- Auth/Endpoint errors: Verify `VISION_ENDPOINT` and `VISION_KEY` in `.env`
- SSL/requests issues on macOS: Make sure certs are up to date; consider `pip install certifi`
- Matplotlib backend issues in headless terminals: The script uses `pyplot` only to save figures; this typically works without a display. If you hit backend errors, set `MPLBACKEND=Agg` in your environment.

## Notes
- To use all Azure Vision features (i.e. captions) select Northern or Western Europe as Region in you project as some feature are region locked.
- The script currently requests TEXT, CAPTION, DENSE_CAPTIONS, TAGS, OBJECTS, and PEOPLE features by defaults.
- People boxes are drawn in cyan and saved only when at least one person is detected.
