# 🦯 AI Navigation for Visually Impaired

A Streamlit web app that helps visually impaired people move around safely. Take a photo of the path ahead with a phone camera, and the app:

1. **Detects obstacles** (people, vehicles, animals, chairs, tables, signs and more) using the YOLOv8 AI model.
2. **Works out where each one is**: on your left, ahead, or on your right.
3. **Estimates how close** it is: very close (< 1 m), close (1–2 m), a few steps away (2–4 m), or far.
4. **Gives a clear instruction**: *Stop*, *Move slightly left/right*, *Walk slowly*, or *Path is clear*.
5. **Speaks it aloud** in **English, Hindi or Kannada**.

## Project structure

```
ai-navigation-visually-impaired/
├── app.py                 # Streamlit user interface
├── navigator.py           # Detection, position/distance logic, guidance, voice
├── requirements.txt       # Python libraries
├── packages.txt           # System libraries needed on Streamlit Cloud
├── .streamlit/
│   └── config.toml        # High-contrast dark theme
├── .gitignore
└── README.md
```

## Run on your computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

The AI model file (`yolov8n.pt`, about 6 MB) downloads automatically the first time.

## Upload to GitHub

1. Go to [github.com/new](https://github.com/new) and create a repository, e.g. `ai-navigation-visually-impaired` (Public).
2. Click **"uploading an existing file"**.
3. Drag in **all files**, including the `.streamlit` folder.
   *(Files starting with a dot can be hidden on Windows/Mac. If `.streamlit/config.toml` doesn't upload, click **Add file → Create new file**, type `.streamlit/config.toml` as the name, and paste its contents.)*
4. Click **Commit changes**.

## Deploy on Streamlit Community Cloud (free)

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **Create app → Deploy a public app from GitHub**.
3. Choose your repository, branch `main`, main file `app.py`.
4. Click **Deploy**. The first build takes 5–10 minutes (it installs the AI libraries).

Open the app link on a phone to use the camera. The browser will ask for camera permission; allow it.

## How it works

| Step | Method |
|------|--------|
| Object detection | YOLOv8 nano (Ultralytics), 80 everyday object types |
| Left / ahead / right | Object centre in the left 35%, middle 30% or right 35% of the photo |
| Distance | How much of the photo the object fills (bigger = closer) |
| In walking path | Object overlaps the middle 30% of the photo |
| Direction advice | Moves you towards the side with fewer/farther obstacles; says *Stop* if blocked ahead and both sides are busy |
| Voice | Google Text-to-Speech (gTTS) |

## Limitations

- Distances are rough estimates from a single photo, not measurements.
- Works on photos (one at a time), not continuous live video.
- Cannot detect steps, potholes, kerbs or wet floors (the model was not trained on them).
- Voice needs an internet connection.

**This is an assistive prototype. Always use a white cane, guide dog or human guide as the primary mobility aid.**

## Ideas for future work

- Live video mode with `streamlit-webrtc`
- Read signboards and text aloud (OCR)
- Train on Indian road scenes: potholes, open drains, auto-rickshaws, stray animals
- Depth estimation model for real distances
