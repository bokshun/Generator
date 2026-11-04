#!/usr/bin/env python3
"""Beat-synced music video generator.

Free tools only: numpy, OpenCV, Pillow and ffmpeg.
Detects the tempo of a song, cuts between your artwork on the beat, adds a
Ken Burns move, a beat pulse and a live spectrum, a title card and an outro.

Example:
  python make_video.py --audio song.wav --images art/*.png \
      --title "SONG TITLE" --subtitle "Artist or Label" \
      --logo mglogo.png --out video.mp4
"""
import argparse
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

GOLD = (217, 164, 65)  # RGB
FONT_BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"


# --------------------------------------------------------------------------
# audio analysis
# --------------------------------------------------------------------------
def decode_audio(path, sr=22050):
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr),
         "-f", "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32).copy(), sr


def stft_mag(x, n_fft, hop):
    win = np.hanning(n_fft).astype(np.float32)
    n = 1 + max(0, (len(x) - n_fft) // hop)
    out = np.empty((n, n_fft // 2 + 1), dtype=np.float32)
    for i0 in range(0, n, 512):
        i1 = min(n, i0 + 512)
        idx = np.arange(n_fft)[None, :] + hop * np.arange(i0, i1)[:, None]
        out[i0:i1] = np.abs(np.fft.rfft(x[idx] * win, axis=1))
    return out


def beat_grid(x, sr):
    """Return (bpm, beat_times_seconds) assuming a steady tempo."""
    hop = 512
    S = np.log1p(10 * stft_mag(x, 1024, hop))
    flux = np.maximum(0, np.diff(S, axis=0)).sum(1)
    flux = np.concatenate([[0.0], flux])
    flux = np.convolve(flux, np.hanning(5) / np.hanning(5).sum(), mode="same")
    flux = np.maximum(flux - flux.mean(), 0)
    fps = sr / hop
    ac = np.correlate(flux, flux, "full")[len(flux) - 1:]
    bpms = np.arange(70, 181, 0.25)
    lags = 60 * fps / bpms
    prior = np.exp(-0.5 * (np.log2(bpms / 115) / 0.6) ** 2)
    sc = np.interp(lags, np.arange(len(ac)), ac) * prior
    cand = bpms[np.argsort(sc)[::-1][:40]]
    idx = np.arange(len(flux))

    best = (-1.0, 120.0, 0.0)
    for bpm in np.unique(np.round(cand, 2)):
        period = 60 * fps / bpm
        for ph in np.arange(0, period, 0.5):
            t = np.arange(ph, len(flux) - 1, period)
            s = np.interp(t, idx, flux).mean()
            if s > best[0]:
                best = (s, bpm, ph)
    _, bpm, ph = best
    period = 60 * fps / bpm
    beats = np.arange(ph, len(flux) - 1, period) / fps
    return float(bpm), beats


def spectrum_frames(x, sr, n_frames, fps, n_bands=56):
    """Per-video-frame band levels in 0..1 with a fast-attack, slow-decay peak."""
    n_fft = 2048
    win = np.hanning(n_fft).astype(np.float32)
    edges = np.geomspace(45, 9000, n_bands + 1)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    bins = [np.where((freqs >= edges[i]) & (freqs < edges[i + 1]))[0]
            for i in range(n_bands)]
    raw = np.zeros((n_frames, n_bands), dtype=np.float32)
    xp = np.pad(x, (n_fft, n_fft))
    for f in range(n_frames):
        c = int(f / fps * sr) + n_fft
        seg = xp[c - n_fft // 2: c + n_fft // 2] * win
        m = np.abs(np.fft.rfft(seg))
        for b in range(n_bands):
            sel = bins[b]
            raw[f, b] = m[sel].mean() if len(sel) else m[np.searchsorted(freqs, edges[b])]
    raw = np.log1p(raw * 30)
    # tilt so highs are visible, normalise to a robust ceiling
    raw *= np.linspace(0.85, 1.35, n_bands)[None, :]
    raw /= np.percentile(raw, 98)
    raw = np.clip(raw, 0, 1.15)
    out = np.zeros_like(raw)
    prev = np.zeros(n_bands, dtype=np.float32)
    for f in range(n_frames):
        prev = np.maximum(raw[f], prev * 0.86)
        out[f] = prev
    return np.clip(out, 0, 1)


# --------------------------------------------------------------------------
# image helpers
# --------------------------------------------------------------------------
def load_rgb(path):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (0, 0, 0, 255))
        bg.alpha_composite(im)
        im = bg
    return np.array(im.convert("RGB"))


def prep_scene(path, W, H, crops=None):
    img = load_rgb(path)
    if crops and path in crops:
        x0, x1 = crops[path]
        img = img[:, x0:x1]
    h, w = img.shape[:2]
    # blurred cover background
    s = max(W / w, H / h)
    big = cv2.resize(img, (int(w * s) + 1, int(h * s) + 1), interpolation=cv2.INTER_AREA)
    y0 = (big.shape[0] - H) // 2
    x0 = (big.shape[1] - W) // 2
    bg = big[y0:y0 + H, x0:x0 + W]
    bg = cv2.GaussianBlur(bg, (0, 0), 28)
    bg = (bg.astype(np.float32) * 0.42).astype(np.uint8)
    # sharp, contained foreground on a full-size RGBA canvas
    fit = min(W * 0.94 / w, H * 0.92 / h)
    fw, fh = int(w * fit), int(h * fit)
    fg = cv2.resize(img, (fw, fh), interpolation=cv2.INTER_AREA if fit < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((H, W, 4), dtype=np.uint8)
    oy, ox = (H - fh) // 2, (W - fw) // 2
    canvas[oy:oy + fh, ox:ox + fw, :3] = fg
    canvas[oy:oy + fh, ox:ox + fw, 3] = 255
    canvas[:, :, 3] = cv2.GaussianBlur(canvas[:, :, 3], (0, 0), 1.2)
    return bg, canvas


def draw_text_rgba(W, H, lines, anchor="left"):
    """Pre-render text lines [(text,size,font,color,spacing)] onto an RGBA layer."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d, ds = ImageDraw.Draw(layer), ImageDraw.Draw(shadow)
    total_h = sum(int(sz * 1.35) for _, sz, *_ in lines)
    y = int(H * 0.5 - total_h / 2) if anchor == "center" else int(H * 0.52)
    for text, size, fontpath, color, spacing in lines:
        font = ImageFont.truetype(fontpath, size)
        widths = [d.textlength(ch, font=font) + spacing for ch in text]
        tw = sum(widths) - spacing
        x = int((W - tw) / 2) if anchor == "center" else int(W * 0.06)
        for ch, wch in zip(text, widths):
            ds.text((x + 3, y + 4), ch, font=font, fill=(0, 0, 0, 230))
            d.text((x, y), ch, font=font, fill=color + (255,))
            x += wch
        y += int(size * 1.35)
    shadow = shadow.filter(ImageFilter.GaussianBlur(7))
    shadow.alpha_composite(layer)
    return np.array(shadow)


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def blend_rgba(frame, rgba, alpha):
    """frame uint8 HxWx3 (BGR) <- rgba (RGB order) scaled by alpha."""
    if alpha <= 0.003:
        return
    a = (rgba[:, :, 3:4].astype(np.float32) / 255.0) * alpha
    rgb = rgba[:, :, 2::-1].astype(np.float32)  # RGB -> BGR
    frame[:] = (frame.astype(np.float32) * (1 - a) + rgb * a).astype(np.uint8)


def blend_region(frame, rgba, alpha, y0, x_end=None):
    """Blend an RGBA band (RGB order) into frame rows y0.., optionally only up to column x_end."""
    if alpha <= 0.003:
        return
    h, w = rgba.shape[:2]
    xe = w if x_end is None else int(min(w, x_end))
    if xe <= 0:
        return
    reg = frame[y0:y0 + h, :xe].astype(np.float32)
    a = (rgba[:, :xe, 3:4].astype(np.float32) / 255.0) * alpha
    rgb = rgba[:, :xe, 2::-1].astype(np.float32)
    frame[y0:y0 + h, :xe] = (reg * (1 - a) + rgb * a).astype(np.uint8)


def build_lyric_pack(lines, W, H):
    """Pre-render each lyric line as a white layer (with shadow) and a gold layer."""
    size = int(H * 0.05)
    font = ImageFont.truetype(FONT_BOLD, size)
    band_h = int(H * 0.085)
    pack = []
    for ln in lines:
        text, starts, end = ln["text"], ln["starts"], ln["end"]
        words = text.split(" ")
        tw = font.getlength(text)
        x0 = (W - tw) / 2
        y = (band_h - size) // 2 - int(size * 0.12)
        white = Image.new("RGBA", (W, band_h), (0, 0, 0, 0))
        shadow = Image.new("RGBA", (W, band_h), (0, 0, 0, 0))
        gold = Image.new("RGBA", (W, band_h), (0, 0, 0, 0))
        ImageDraw.Draw(shadow).text((x0 + 3, y + 4), text, font=font, fill=(0, 0, 0, 235))
        ImageDraw.Draw(white).text((x0, y), text, font=font, fill=(255, 255, 255, 255))
        ImageDraw.Draw(gold).text((x0, y), text, font=font, fill=GOLD + (255,))
        shadow = shadow.filter(ImageFilter.GaussianBlur(6))
        shadow.alpha_composite(white)
        spans = []
        for i, w0 in enumerate(starts):
            w1 = min(starts[i + 1], w0 + 0.55) if i < len(starts) - 1 else end
            xs = x0 + (font.getlength(" ".join(words[:i]) + " ") if i else 0.0)
            xe = x0 + font.getlength(" ".join(words[:i + 1]))
            spans.append((w0, max(w1, w0 + 0.05), xs, xe))
        pack.append({"start": starts[0], "end": end, "white": np.array(shadow),
                     "gold": np.array(gold), "spans": spans, "x0": x0})
    # soft dark band behind the lyrics for legibility
    dh = band_h + int(H * 0.05)
    prof = np.sin(np.linspace(0, np.pi, dh)) ** 1.2 * 0.55
    dark = np.repeat(prof[:, None], W, axis=1).astype(np.float32)[:, :, None]
    return pack, dark


def gold_edge(spans, x0, t):
    """Right edge (px) of the already-sung part of a line at time t."""
    edge = x0
    for w0, w1, xs, xe in spans:
        if t >= w1:
            edge = xe
        elif t >= w0:
            edge = xs + (xe - xs) * (t - w0) / (w1 - w0)
            break
        else:
            break
    return edge


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--logo", default=None, help="dark-on-light logo, shown white")
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--t0", type=float, default=0.0, help="preview start (s)")
    ap.add_argument("--t1", type=float, default=0.0, help="preview end (s), 0 = full")
    ap.add_argument("--outro", type=float, default=9.0)
    ap.add_argument("--lyrics", default=None,
                    help="JSON list of {text, starts[], end} (word start times in seconds)")
    ap.add_argument("--crop", action="append", default=[],
                    help="PATH:X0:X1 pixel columns to keep (repeatable)")
    a = ap.parse_args()
    W, H, FPS = a.w, a.h, a.fps

    x, sr = decode_audio(a.audio)
    dur = len(x) / sr
    bpm, beats = beat_grid(x, sr)
    print(f"duration {dur:.1f}s  tempo {bpm:.1f} BPM  beats {len(beats)}", flush=True)

    n_frames = int(dur * FPS)
    spec = spectrum_frames(x, sr, n_frames, FPS)

    # ---- scene boundaries on the beat -------------------------------------
    beat_len = 60.0 / bpm
    per = 8 if 8 * beat_len >= 3.8 else 16
    outro_start_t = max(dur - a.outro, dur * 0.7)
    first = beats[np.searchsorted(beats, 4.5)]
    bounds = [0.0]
    k = int(np.searchsorted(beats, first))
    while k < len(beats) and beats[k] < outro_start_t - 2.0:
        bounds.append(float(beats[k]))
        k += per
    outro_t = float(beats[np.argmin(np.abs(beats - outro_start_t))])
    bounds.append(outro_t)          # last image scene ends, outro begins
    bounds.append(dur + 5)
    n_scenes = len(bounds) - 2      # image scenes (+ outro at index n_scenes)
    print(f"{n_scenes} scenes, cut every {per} beats, outro at {outro_t:.1f}s", flush=True)

    # ---- assets -----------------------------------------------------------
    order = list(a.images)
    crops = {}
    for c in a.crop:
        pth, x0, x1 = c.rsplit(":", 2)
        crops[pth] = (int(x0), int(x1))
    prepped = {p: prep_scene(p, W, H, crops) for p in order}
    scenes = [prepped[order[i % len(order)]] for i in range(n_scenes)]

    title_layer = draw_text_rgba(W, H, [
        (a.title.upper(), int(H * 0.115), FONT_BOLD, (255, 255, 255), int(H * 0.012)),
        (a.subtitle.upper(), int(H * 0.034), FONT_REG, GOLD, int(H * 0.008)),
    ]) if a.title else None

    logo_white = None
    if a.logo:
        lg = Image.open(a.logo).convert("RGBA")
        gray = np.array(lg.convert("L"))
        alpha = np.array(lg)[:, :, 3]
        # dark-on-light logo -> white shape on transparent
        mask = np.where(alpha > 0, 255 - gray, 0).astype(np.uint8)
        lg_rgba = np.dstack([np.full_like(mask, 255)] * 3 + [mask])
        logo_white = lg_rgba

    outro_layer = np.zeros((H, W, 4), dtype=np.uint8)
    if logo_white is not None:
        ls = int(H * 0.42)
        lg = cv2.resize(logo_white, (ls, ls), interpolation=cv2.INTER_AREA)
        oy, ox = int(H * 0.07), (W - ls) // 2
        outro_layer[oy:oy + ls, ox:ox + ls] = lg
    outro_text = Image.fromarray(np.zeros((H, W, 4), dtype=np.uint8))
    od = ImageDraw.Draw(outro_text)
    f1 = ImageFont.truetype(FONT_BOLD, int(H * 0.05))
    f2 = ImageFont.truetype(FONT_REG, int(H * 0.03))
    for text, f, y, col in ((a.subtitle.upper() or "", f1, 0.55, (255, 255, 255)),
                            (a.title.upper(), f2, 0.64, GOLD)):
        if not text:
            continue
        sp = int(H * 0.006)
        ws = [od.textlength(c, font=f) + sp for c in text]
        xx = (W - (sum(ws) - sp)) / 2
        for c, wc in zip(text, ws):
            od.text((xx, int(H * y)), c, font=f, fill=col + (255,))
            xx += wc
    outro_text = np.array(outro_text)
    outro_layer = np.where(outro_text[:, :, 3:4] > 0, outro_text, outro_layer)

    watermark = None
    if logo_white is not None:
        ws_ = int(H * 0.085)
        watermark = np.zeros((H, W, 4), dtype=np.uint8)
        wm = cv2.resize(logo_white, (ws_, ws_), interpolation=cv2.INTER_AREA)
        watermark[int(H * 0.035):int(H * 0.035) + ws_, W - ws_ - int(W * 0.025):W - int(W * 0.025)] = wm

    lyric_pack, lyric_dark, lyric_y = None, None, int(H * 0.765)
    if a.lyrics:
        import json
        lyric_pack, lyric_dark = build_lyric_pack(json.load(open(a.lyrics, encoding="utf-8")), W, H)
        lyric_starts = np.array([ln["start"] for ln in lyric_pack])
        print(f"{len(lyric_pack)} lyric lines", flush=True)

    # vignette
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    vig = np.clip(1.0 - 0.38 * np.clip(r - 0.55, 0, None) ** 1.5, 0.45, 1.0)[:, :, None].astype(np.float32)

    # ---- per-frame helpers ------------------------------------------------
    def pulse_at(t):
        i = np.searchsorted(beats, t, side="right") - 1
        if i < 0:
            return 0.0
        return float(np.exp(-(t - beats[i]) * 7.5))

    def render_scene(i, t, fidx):
        """BGR frame for image scene i at time t."""
        bg, fgc = scenes[i]
        length = bounds[i + 1] - bounds[i]
        p = (t - bounds[i]) / length
        pul = pulse_at(t)
        low = float(spec[min(fidx, n_frames - 1), :8].mean())
        s = 1.0 + 0.085 * p + 0.022 * pul * (0.6 + 0.4 * low)
        dirn = 1 if i % 2 == 0 else -1
        tx = dirn * (p - 0.5) * 0.03 * W
        ty = (0.5 - p) * 0.012 * H * (1 if i % 3 else -1)
        M = np.array([[s, 0, W / 2 - s * W / 2 + tx],
                      [0, s, H / 2 - s * H / 2 + ty]], dtype=np.float32)
        fg = cv2.warpAffine(fgc, M, (W, H), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        # bg drifts slowly the opposite way
        sb = 1.03 + 0.03 * p
        Mb = np.array([[sb, 0, W / 2 - sb * W / 2 - tx * 0.5],
                       [0, sb, H / 2 - sb * H / 2]], dtype=np.float32)
        base = cv2.warpAffine(bg, Mb, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        al = fg[:, :, 3:4].astype(np.float32) / 255.0
        out = base.astype(np.float32) * (1 - al) + fg[:, :, 2::-1].astype(np.float32) * al
        out += 14.0 * pul  # tiny flash on the beat
        return out

    def render_outro(t, fidx):
        o = np.zeros((H, W, 3), dtype=np.float32)
        return o

    def draw_bars(frame, fidx, alpha):
        lv = spec[min(fidx, n_frames - 1)]
        nb = len(lv)
        total_w = int(W * 0.78)
        bw = total_w // nb
        gap = max(2, bw // 2)
        x0 = (W - nb * bw) // 2
        base_y = H - int(H * 0.04)
        max_h = int(H * 0.10)
        col = np.array(GOLD[::-1], dtype=np.float32)  # BGR
        for b in range(nb):
            hh = max(2, int((lv[b] ** 1.6) * max_h))
            xa, xb = x0 + b * bw, x0 + b * bw + bw - gap
            reg = frame[base_y - hh:base_y, xa:xb].astype(np.float32)
            frame[base_y - hh:base_y, xa:xb] = (reg * (1 - 0.7 * alpha) + col * 0.7 * alpha).astype(np.uint8)

    # ---- encode -----------------------------------------------------------
    t_start = a.t0
    t_end = a.t1 if a.t1 > 0 else dur
    f_start, f_end = int(t_start * FPS), min(n_frames, int(t_end * FPS))
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-ss", f"{t_start:.3f}", "-t", f"{t_end - t_start:.3f}", "-i", a.audio,
           "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", a.out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    hw = 0.25  # half-width of the crossfade around each cut
    fade_in, fade_out = 1.6, 1.8
    for f in range(f_start, f_end):
        t = f / FPS
        # which scene (cut happens at the middle of the crossfade)
        i = max(j for j in range(len(bounds) - 1) if bounds[j] - hw <= t or j == 0)
        def layer(j):
            return render_outro(t, f) if j >= n_scenes else render_scene(j, t, f)
        cur = layer(i)
        if i > 0 and t < bounds[i] + hw:
            al = smooth((t - (bounds[i] - hw)) / (2 * hw))
            prev = layer(i - 1)
            cur = prev * (1 - al) + cur * al
        frame = np.clip(cur * vig if i < n_scenes else cur, 0, 255).astype(np.uint8)

        is_outro = i >= n_scenes
        if is_outro:
            k = smooth((t - outro_t) / 1.0)
            blend_rgba(frame, outro_layer, k)
        if watermark is not None and not is_outro:
            blend_rgba(frame, watermark, 0.55 * smooth((t - 1.5) / 1.5))
        if title_layer is not None and t < 7.0:
            ta = smooth((t - 0.9) / 1.0) * (1 - smooth((t - 4.6) / 1.0))
            blend_rgba(frame, title_layer, ta)
        if lyric_pack is not None:
            li = int(np.searchsorted(lyric_starts, t + 0.25, side="right")) - 1
            if li >= 0:
                ln = lyric_pack[li]
                la = smooth((t - (ln["start"] - 0.25)) / 0.15) * (1 - smooth((t - (ln["end"] + 0.35)) / 0.4))
                if la > 0.003:
                    dy0 = lyric_y - int(H * 0.025)
                    reg = frame[dy0:dy0 + lyric_dark.shape[0]].astype(np.float32)
                    frame[dy0:dy0 + lyric_dark.shape[0]] = (reg * (1 - lyric_dark * la)).astype(np.uint8)
                    blend_region(frame, ln["white"], la, lyric_y)
                    blend_region(frame, ln["gold"], la, lyric_y, gold_edge(ln["spans"], ln["x0"], t))
        draw_bars(frame, f, 1.0 if not is_outro else 1 - smooth((t - outro_t) / 1.0))
        # global fades
        g = smooth(t / fade_in) * (1 - smooth((t - (dur - fade_out)) / fade_out))
        if g < 0.999:
            frame = (frame.astype(np.float32) * g).astype(np.uint8)
        enc.stdin.write(frame.tobytes())
        if f % 150 == 0:
            print(f"  frame {f}/{f_end}  ({100 * (f - f_start) / max(1, f_end - f_start):.0f}%)", flush=True)
    enc.stdin.close()
    rc = enc.wait()
    print("done" if rc == 0 else f"ffmpeg exit {rc}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
