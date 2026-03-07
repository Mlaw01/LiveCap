# Live Caption Overlay (Windows)

Desktop app that captures system audio (WASAPI loopback), transcribes English with Deepgram, translates to Chinese with DeepL, and shows live captions in a movable/resizable overlay window.

## Requirements

- Windows
- Python 3.10+
- A working speaker/output device
- Deepgram API key
- DeepL API key

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install PyQt6 deepgram-sdk deepl numpy python-dotenv pyaudiowpatch
```

3. Create local env file from template:

```powershell
copy .env.example .env
```

4. Edit `.env` and set:

- `DEEPGRAM_API_KEY`
- `DEEPL_API_KEY`

## Run

```powershell
python overlay.py
```

## UI Controls

- `Start`: begin audio capture + API streaming
- `Stop`: stop streaming
- `Clear`: clear current captions and transcript history
- `Always on top`: toggle on-top behavior at runtime
- `Ctrl+Shift+H`: hide/show window

## Safety Features

- Session auto-stop timer (`MAX_RUN_MINUTES` in `overlay.py`)
- Visible countdown (`MM:SS left`)
- Scrollable transcript history with timestamps

## Files

- `overlay.py`: primary app (recommended entrypoint)
- `main.py`: alternate script path (optional)
- `.env.example`: environment variable template (safe to commit)

## Security

- Do not commit `.env`
- Rotate keys immediately if exposed

