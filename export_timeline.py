#!/usr/bin/env python3
"""Export the beat-synced edit as an editable timeline package.

Writes, with no downloads and no paid tools:
  timeline.xml   Final Cut Pro 7 XML (xmeml v4) - imports into Adobe Premiere Pro
  timeline.edl   CMX3600 EDL (video + audio) - Premiere, DaVinci Resolve, others
  beats.csv      every detected beat (time, frame, whether a cut lands there)
It uses the same beat grid and scene plan as make_video.py.
"""
import csv
import importlib.util
import os
import subprocess
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("mv", os.path.join(HERE, "make_video.py"))
mv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mv)

FPS = 30


def plan(audio, outro_len=9.0):
    """Same scene logic as make_video.py main()."""
    x, sr = mv.decode_audio(audio)
    dur = len(x) / sr
    bpm, beats = mv.beat_grid(x, sr)
    beat_len = 60.0 / bpm
    per = 8 if 8 * beat_len >= 3.8 else 16
    outro_start_t = max(dur - outro_len, dur * 0.7)
    first = beats[np.searchsorted(beats, 4.5)]
    bounds = [0.0]
    k = int(np.searchsorted(beats, first))
    while k < len(beats) and beats[k] < outro_start_t - 2.0:
        bounds.append(float(beats[k]))
        k += per
    outro_t = float(beats[np.argmin(np.abs(beats - outro_start_t))])
    bounds.append(outro_t)
    return dur, bpm, beats, per, bounds, outro_t


def tc(frames):
    f = int(frames)
    return f"{f // (FPS * 3600):02d}:{(f // (FPS * 60)) % 60:02d}:{(f // FPS) % 60:02d}:{f % FPS:02d}"


def sub(parent, tag, text=None, **attrs):
    e = ET.SubElement(parent, tag, attrs)
    if text is not None:
        e.text = str(text)
    return e


def rate(parent):
    r = sub(parent, "rate")
    sub(r, "timebase", FPS)
    sub(r, "ntsc", "FALSE")


def main():
    audio, media_dir, out_dir, win_dir = sys.argv[1:5]
    names = sys.argv[5:]  # image file names (in media_dir), scene order, then outro card last
    outro_card = names[-1]
    images = names[:-1]

    dur, bpm, beats, per, bounds, outro_t = plan(audio)
    audio_name = os.path.basename(audio)
    total = int(round(float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", audio]).decode().strip()) * FPS))
    n_scenes = len(bounds) - 1
    cuts = [int(round(b * FPS)) for b in bounds]  # frame of each scene start; last = outro start
    cuts.append(total)

    def pathurl(name):
        p = (win_dir.rstrip("\\/") + "/media/" + name).replace("\\", "/")
        return "file://localhost/" + urllib.parse.quote(p, safe="/:").replace(":", "%3a", 1)

    clips = []  # (name, start, end)
    for i in range(n_scenes):
        clips.append((images[i % len(images)], cuts[i], cuts[i + 1]))
    clips.append((outro_card, cuts[n_scenes], total))

    # ---------------------------------------------------------------- XML
    root = ET.Element("xmeml", version="4")
    seq = sub(root, "sequence", id="sequence-1")
    sub(seq, "name", "Beat cut")
    sub(seq, "duration", total)
    rate(seq)
    media = sub(seq, "media")
    video = sub(media, "video")
    fmt = sub(video, "format")
    sc = sub(fmt, "samplecharacteristics")
    rate(sc)
    sub(sc, "width", 1920)
    sub(sc, "height", 1080)
    sub(sc, "anamorphic", "FALSE")
    sub(sc, "pixelaspectratio", "square")
    sub(sc, "fielddominance", "none")
    track = sub(video, "track")
    file_ids = {}
    for n, (name, s, e) in enumerate(clips, 1):
        ci = sub(track, "clipitem", id=f"clipitem-{n}")
        sub(ci, "name", name)
        sub(ci, "duration", 108000)
        rate(ci)
        sub(ci, "start", s)
        sub(ci, "end", e)
        sub(ci, "in", 0)
        sub(ci, "out", e - s)
        sub(ci, "alphatype", "none")
        sub(ci, "stillframe", "TRUE")
        if name not in file_ids:
            fid = f"file-{len(file_ids) + 1}"
            file_ids[name] = fid
            fe = sub(ci, "file", id=fid)
            sub(fe, "name", name)
            sub(fe, "pathurl", pathurl(name))
            rate(fe)
            sub(fe, "duration", 108000)
            fm = sub(sub(sub(fe, "media"), "video"), "samplecharacteristics")
            w, h = Image.open(os.path.join(media_dir, name)).size
            rate(fm)
            sub(fm, "width", w)
            sub(fm, "height", h)
        else:
            sub(ci, "file", id=file_ids[name])
    sub(video, "enabled", "TRUE")
    sub(video, "locked", "FALSE")

    aud = sub(media, "audio")
    afmt = sub(sub(aud, "format"), "samplecharacteristics")
    sub(afmt, "depth", 16)
    sub(afmt, "samplerate", 48000)
    sub(aud, "numOutputChannels", 2)
    for ch in (1, 2):
        atrack = sub(aud, "track")
        ci = sub(atrack, "clipitem", id=f"clipitem-a{ch}")
        sub(ci, "name", audio_name)
        sub(ci, "duration", total)
        rate(ci)
        sub(ci, "start", 0)
        sub(ci, "end", total)
        sub(ci, "in", 0)
        sub(ci, "out", total)
        if ch == 1:
            fe = sub(ci, "file", id="file-audio")
            sub(fe, "name", audio_name)
            sub(fe, "pathurl", pathurl(audio_name))
            rate(fe)
            sub(fe, "duration", total)
            aa = sub(sub(fe, "media"), "audio")
            ac = sub(aa, "samplecharacteristics")
            sub(ac, "depth", 16)
            sub(ac, "samplerate", 48000)
            sub(aa, "channelcount", 2)
        else:
            sub(ci, "file", id="file-audio")
        st = sub(ci, "sourcetrack")
        sub(st, "mediatype", "audio")
        sub(st, "trackindex", ch)
    tcn = sub(seq, "timecode")
    rate(tcn)
    sub(tcn, "string", "00:00:00:00")
    sub(tcn, "frame", 0)
    sub(tcn, "displayformat", "NDF")
    for n in range(n_scenes + 1):
        m = sub(seq, "marker")
        sub(m, "name", "Outro" if n == n_scenes else f"Cut {n + 1:02d}")
        sub(m, "comment", f"{cuts[n] / FPS:.2f}s - on beat" if n else "start")
        sub(m, "in", cuts[n])
        sub(m, "out", -1)

    xml = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(indent="  ")
    xml = xml.replace('<?xml version="1.0" ?>', '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>')
    with open(os.path.join(out_dir, "timeline.xml"), "w", encoding="utf-8") as f:
        f.write(xml)

    # ---------------------------------------------------------------- EDL
    lines = ["TITLE: Beat cut", "FCM: NON-DROP FRAME", ""]
    for n, (name, s, e) in enumerate(clips, 1):
        lines.append(f"{n:03d}  AX       V     C        {tc(0)} {tc(e - s)} {tc(s)} {tc(e)}")
        lines.append(f"* FROM CLIP NAME: {name}")
        lines.append("")
    n = len(clips) + 1
    lines.append(f"{n:03d}  AX       AA    C        {tc(0)} {tc(total)} {tc(0)} {tc(total)}")
    lines.append(f"* FROM CLIP NAME: {audio_name}")
    with open(os.path.join(out_dir, "timeline.edl"), "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")

    # ---------------------------------------------------------------- beats
    cutset = set(cuts)
    with open(os.path.join(out_dir, "beats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["beat", "seconds", "frame_30fps", "timecode", "cut_here"])
        for i, b in enumerate(beats, 1):
            fr = int(round(b * FPS))
            w.writerow([i, f"{b:.3f}", fr, tc(fr), "CUT" if fr in cutset else ""])

    print(f"duration {dur:.2f}s  bpm {bpm:.1f}  cut every {per} beats  scenes {n_scenes}  clips {len(clips)}  total frames {total}")
    print("outro at", round(outro_t, 2), "s")


if __name__ == "__main__":
    main()
