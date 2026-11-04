# Generator

Free, open-source music video generator for Marianas Grown Productions.

It listens to a song, finds the tempo, and builds a 1080p video that:

- cuts between your artwork **on the beat** (every 8 or 16 beats)
- adds a slow Ken Burns move and a beat pulse
- shows a live gold spectrum along the bottom
- shows **karaoke-style lyrics** (words fill in gold as they are sung), if you give it timings
- opens with a title card and closes with the MG logo outro

Everything runs on numpy, OpenCV, Pillow and ffmpeg. No paid services.

> **This repo is public. Code only.** Run it locally and never commit songs, artwork or
> lyrics here (the `.gitignore` blocks common media types as a safety net).

## Run it

    pip install numpy opencv-python-headless pillow     # plus ffmpeg on PATH
    python make_video.py --audio song.wav --images art1.png art2.jpg art3.png \
        --title "SONG TITLE" --subtitle "Marianas Grown Productions" \
        --logo mglogo.png --lyrics lyrics.json --out video.mp4

Useful flags: `--w/--h` (default 1920x1080), `--fps`, `--t0/--t1` to preview part
of the song, `--crop FILE:X0:X1` to trim an image's pixel columns.

Fonts: it uses Liberation Sans. On Windows, change `FONT_BOLD` / `FONT_REG` at the
top of `make_video.py` to a font you have (for example `C:/Windows/Fonts/arialbd.ttf`).

## Lyrics file format

`lyrics.json` is a list with one entry per sung line. `starts` has one time (seconds)
per word, and `end` is when the line finishes:

    [
      {"text": "First line of words", "starts": [0.0, 0.5, 1.0, 1.5], "end": 2.0},
      {"text": "Second line", "starts": [3.0, 3.5], "end": 4.0}
    ]

Word counts must match. Suno's lyric alignment gives these timings for songs made there.

## Export the edit to Premiere or DaVinci Resolve

`export_timeline.py` writes a Final Cut XML (imports into Premiere), a CMX3600 EDL and
a beats list from the same beat plan. See the top of that file for its arguments.
