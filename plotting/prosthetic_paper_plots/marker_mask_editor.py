#!/usr/bin/env python3
r"""Local marker-removal pilot for single frames and short video clips.

Run (Windows / PowerShell):
    uv run --with pillow --with opencv-python-headless python marker_mask_editor.py
    uv run --with pillow --with opencv-python-headless python marker_mask_editor.py "C:\data\camera.mp4" --frame 300
    uv run --with pillow --with opencv-python-headless python marker_mask_editor.py "C:\data\camera.mp4" --frame 300 --frames 60

Use a Python installation with tkinter (included in the standard Windows installer).
Left-drag on the ORIGINAL pane to paint a removal mask; use Erase to shrink it.
Wheel: zoom. Right-drag: pan both panes. Ctrl+Z: undo. Ctrl+S: save.
Radius is in ORIGINAL image pixels. Preview updates after each brush stroke.
In single-frame mode, each save creates a new folder with original.png, edited.png, mask.png,
overlay.png, comparison.png, and session.json. No source file is modified.
The JSON records the exact frame, decoded-image hash, and inpainting settings.
CLIP WORKFLOW (all processing stays on this computer):
    Open clip... -> choose first frame and frame count (try 30-60 frames).
    Paint the visible markers, or Load mask... from the old frame editor.
    Track forward proposes masks for the remaining frames, stopping on a failed check.
    Use Prev/Next, the timeline, or Play to inspect. Correct with Paint/Erase.
    Review + next confirms each inspected frame. Painting invalidates its review.
    When a marker is hidden, erase its region; when it reappears, paint it again.
    Re-run Track forward from a corrected frame to update later proposals.
    Manual and reviewed masks are preserved as anchors during tracking.
    Save project... preserves progress; Open project... resumes it against the source.
    Export clip... requires every frame reviewed and writes matched original_frames/,
    edited_frames/, masks/, frame_metadata.csv, clip.json, and playback previews.
    PNGs retain original resolution; preview videos are reduced and lossy, for inspection.

Tracking translates each connected mask region using local optical flow and checks
forward/backward consistency and patch similarity. It does not recognize marker
identity or establish visibility. Inspect every frame, including masks that pass.
Separated markers should have separated masks; touching masks track as one region.
Masks dropped by a failed check require manual correction before continuing.
No model inference is performed here. No video or image is uploaded.

These edits test sensitivity to visible marker cues. Inpainting is an approximation;
check marker bases, silhouettes, and residual cues before running paired inference.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SCHEMA_VERSION = 1
CLIP_SCHEMA_VERSION = 2
TRACKING_SETTINGS = {"window_px": 21, "pyramid_levels": 3, "max_motion_px": 64.0,
                     "forward_backward_px": 1.5, "min_patch_correlation": 0.6}


def read_frame(path, frame_index=0):
    """Return a BGR uint8 frame and provenance; image paths support Unicode."""
    path = Path(path).expanduser().resolve()
    if frame_index < 0:
        raise ValueError("Frame index must be zero or greater.")
    if not path.is_file():
        raise ValueError(f"File not found: {path}")
    if path.suffix.lower() in IMAGE_SUFFIXES:
        if frame_index != 0:
            raise ValueError("Use --frame only with video files.")
        frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Could not decode image: {path.name}")
        return frame, {"source_kind": "image", "source_path": str(path), "frame_index": None}
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {path.name}")
        count, fps = capture.get(cv2.CAP_PROP_FRAME_COUNT), capture.get(cv2.CAP_PROP_FPS)
        if count > 0 and frame_index >= count:
            raise ValueError(f"Frame {frame_index} is outside this video (reported count: {int(count)}).")
        if frame_index and not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            raise ValueError("This video cannot seek to that frame. Extract the frame separately first.")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"Could not decode frame {frame_index}.")
        reported_next = capture.get(cv2.CAP_PROP_POS_FRAMES)
        if reported_next > 0 and abs(reported_next - (frame_index + 1)) > 0.5:
            raise ValueError("The decoder did not report the requested frame position.")
        metadata = {"source_kind": "video", "source_path": str(path), "frame_index": frame_index,
                    "reported_frame_count": int(count) if math.isfinite(count) and count > 0 else None,
                    "reported_fps": fps if math.isfinite(fps) and fps > 0 else None,
                    "decoder_backend": capture.getBackendName()}
        return frame, metadata
    finally:
        capture.release()


def image_hash(frame):
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def inpaint(frame, mask, method="Telea", radius=3.0):
    if mask.shape != frame.shape[:2] or mask.dtype != np.uint8:
        raise ValueError("Mask must be an 8-bit single-channel image with matching dimensions.")
    if method not in {"Telea", "Navier-Stokes"} or not math.isfinite(radius) or radius <= 0:
        raise ValueError("Invalid inpainting method or neighborhood radius.")
    if not np.any(mask):
        return frame.copy()
    flag = cv2.INPAINT_TELEA if method == "Telea" else cv2.INPAINT_NS
    edited = cv2.inpaint(frame, mask, float(radius), flag)
    edited[mask == 0] = frame[mask == 0]
    return edited


def mask_overlay(frame, mask):
    overlay = frame.copy()
    selected = mask > 0
    overlay[selected] = (0.5 * frame[selected] + 0.5 * np.array([40, 40, 255])).astype(np.uint8)
    return overlay


def write_png(path, array):
    ok, encoded = cv2.imencode(".png", array)
    if not ok:
        raise ValueError(f"Could not encode {path.name}.")
    encoded.tofile(path)


def comparison_image(frame, edited):
    """A labeled, full-resolution visual comparison; never used as inference input."""
    left = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    right = Image.fromarray(cv2.cvtColor(edited, cv2.COLOR_BGR2RGB))
    canvas = Image.new("RGB", (2 * left.width + 12, left.height + 30), "#20242a")
    canvas.paste(left, (0, 30))
    canvas.paste(right, (left.width + 12, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "ORIGINAL", fill="white")
    draw.text((left.width + 20, 8), "EDITED", fill="white")
    return canvas


def save_session(output, frame, mask, metadata, method, radius):
    if not np.any(mask):
        raise ValueError("Paint at least one marker region before saving.")
    edited = inpaint(frame, mask, method, radius)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    tag = metadata.get("frame_index")
    tag = "image" if tag is None else f"frame_{tag:06d}"
    stem = Path(metadata["source_path"]).stem
    folder = output / f"{stem}_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    folder.mkdir(exist_ok=False)
    document = {"schema_version": SCHEMA_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                **metadata, "width": frame.shape[1], "height": frame.shape[0],
                "decoded_bgr_sha256": image_hash(frame), "mask_file": "mask.png",
                "mask_pixels": int(np.count_nonzero(mask)), "inpaint_method": method,
                "inpaint_radius_px": float(radius), "opencv_version": cv2.__version__,
                "mask_coordinate_space": "original decoded image pixels",
                "outside_mask_pixels_unchanged": True,
                "note": "Digitally edited frame; not a recording without physical markers."}
    try:
        write_png(folder / "original.png", frame)
        write_png(folder / "edited.png", edited)
        write_png(folder / "mask.png", mask)
        write_png(folder / "overlay.png", mask_overlay(frame, mask))
        comparison_image(frame, edited).save(folder / "comparison.png")
        (folder / "session.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    except Exception:
        # Leave partial output for diagnosis, but never report it as a completed save.
        (folder / "SAVE_INCOMPLETE.txt").write_text("Save failed; do not use this folder for inference.")
        raise
    return folder


def load_session(path, frame):
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported session version.")
    if document.get("decoded_bgr_sha256") != image_hash(frame):
        raise ValueError("This mask belongs to a different frame. Open its original.png or the exact source frame.")
    if (document.get("height"), document.get("width")) != frame.shape[:2]:
        raise ValueError("Saved dimensions do not match this frame.")
    mask_path = path.parent / "mask.png"
    mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != frame.shape[:2]:
        raise ValueError("The saved mask is missing or has different dimensions.")
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    method, radius = document["inpaint_method"], float(document["inpaint_radius_px"])
    inpaint(frame, mask, method, radius)  # Validate settings before changing the editor.
    return mask, method, radius


class MarkerEditor:
    HEADER, GAP = 28, 12

    def __init__(self, root, args):
        self.root, self.args = root, args
        self.frame = self.edited = self.mask = None
        self.metadata, self.history, self.photos = {}, [], []
        self.scale, self.ox, self.oy = 1.0, 0.0, 0.0
        self.last_point = self.pan_point = None
        self.stroke, self.dirty, self.pending = False, False, None
        self.radius = tk.IntVar(value=6)
        self.mode, self.method = tk.StringVar(value="Paint"), tk.StringVar(value="Telea")
        self.fill_radius = tk.DoubleVar(value=3.0)
        self.show_mask = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Open a frame or video to begin.")
        root.title("Marker mask editor - local frame pilot")
        root.geometry("1280x850")
        toolbar = ttk.Frame(root, padding=8)
        toolbar.pack(fill="x")
        for label, command in [("Open...", self.open_dialog), ("Load mask...", self.load_mask),
                               ("Undo", self.undo), ("Clear", self.clear), ("Fit", self.fit), ("Save...", self.save)]:
            ttk.Button(toolbar, text=label, command=command).pack(side="left", padx=2)
        ttk.Label(toolbar, text="Brush radius (px):").pack(side="left", padx=(12, 3))
        ttk.Spinbox(toolbar, from_=1, to=200, width=5, textvariable=self.radius).pack(side="left")
        for mode in ("Paint", "Erase"):
            ttk.Radiobutton(toolbar, text=mode, variable=self.mode, value=mode).pack(side="left", padx=3)
        ttk.Checkbutton(toolbar, text="Show mask", variable=self.show_mask, command=self.render).pack(side="left", padx=8)
        settings = ttk.Frame(root, padding=(8, 0, 8, 8))
        settings.pack(fill="x")
        ttk.Label(settings, text="Fill method:").pack(side="left")
        selector = ttk.Combobox(settings, textvariable=self.method, values=["Telea", "Navier-Stokes"], state="readonly", width=15)
        selector.pack(side="left", padx=5)
        selector.bind("<<ComboboxSelected>>", lambda event: self.settings_changed())
        ttk.Label(settings, text="Fill neighborhood (px):").pack(side="left", padx=(8, 3))
        spin = ttk.Spinbox(settings, from_=0.5, to=30, increment=0.5, textvariable=self.fill_radius, width=5, command=self.settings_changed)
        spin.pack(side="left")
        spin.bind("<Return>", lambda event: self.settings_changed())
        spin.bind("<FocusOut>", lambda event: self.settings_changed())
        ttk.Label(settings, text="Left-drag: paint/erase   |   Wheel: zoom   |   Right-drag: pan   |   Ctrl+Z: undo").pack(side="left", padx=18)
        self.canvas = tk.Canvas(root, bg="#20242a", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        ttk.Label(root, textvariable=self.status, padding=8, wraplength=1200).pack(fill="x")
        self.canvas.bind("<Configure>", lambda event: self.render())
        self.canvas.bind("<ButtonPress-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.drag_stroke)
        root.bind("<ButtonRelease-1>", self.end_stroke)
        self.canvas.bind("<ButtonPress-3>", lambda event: setattr(self, "pan_point", (event.x, event.y)))
        self.canvas.bind("<B3-Motion>", self.pan)
        self.canvas.bind("<ButtonRelease-3>", lambda event: setattr(self, "pan_point", None))
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Button-4>", lambda event: self.zoom(event, 1.2))
        self.canvas.bind("<Button-5>", lambda event: self.zoom(event, 1 / 1.2))
        root.bind("<Control-z>", lambda event: self.undo())
        root.bind("<Control-s>", lambda event: self.save())
        root.protocol("WM_DELETE_WINDOW", self.close)
        if args.source:
            root.after(80, lambda: self.open_source(args.source, args.frame))

    def proceed(self):
        return not self.dirty or messagebox.askyesno("Unsaved edits", "Discard the unsaved mask edits?", parent=self.root)

    def open_dialog(self):
        if not self.proceed():
            return
        path = filedialog.askopenfilename(title="Open an image or video", parent=self.root)
        if path:
            if Path(path).suffix.lower() in IMAGE_SUFFIXES:
                self.open_source(path, 0)
            else:
                from tkinter.simpledialog import askinteger
                index = askinteger("Video frame", "Frame number (zero-based):", initialvalue=0, minvalue=0, parent=self.root)
                if index is not None:
                    self.open_source(path, index)

    def open_source(self, path, index):
        try:
            frame, metadata = read_frame(path, index)
        except Exception as error:
            messagebox.showerror("Could not open frame", str(error), parent=self.root)
            return
        self.frame, self.metadata = frame, metadata
        self.mask, self.edited = np.zeros(frame.shape[:2], np.uint8), frame.copy()
        self.history, self.dirty = [], False
        self.fit()
        self.update_status()

    def layout(self):
        return max(1, (self.canvas.winfo_width() - self.GAP) // 2), max(1, self.canvas.winfo_height() - self.HEADER)

    def fit(self):
        if self.frame is None:
            return
        width, height = self.layout()
        h, w = self.frame.shape[:2]
        self.scale = min(width / w, height / h)
        self.ox, self.oy = (width - w * self.scale) / 2, (height - h * self.scale) / 2
        self.render()

    def render(self):
        if self.pending is not None:
            self.root.after_cancel(self.pending)
        self.pending = None
        if self.frame is None:
            return
        width, height = self.layout()
        left = mask_overlay(self.frame, self.mask) if self.show_mask.get() else self.frame
        self.canvas.delete("all")
        self.photos = []
        affine = (1 / self.scale, 0, -self.ox / self.scale, 0, 1 / self.scale, -self.oy / self.scale)
        for index, (array, title) in enumerate([(left, "ORIGINAL / REMOVAL MASK"), (self.edited, "EDITED PREVIEW")]):
            rgb = Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
            view = rgb.transform((width, height), Image.Transform.AFFINE, affine, Image.Resampling.BILINEAR, fillcolor="#20242a")
            photo = ImageTk.PhotoImage(view)
            self.photos.append(photo)
            start = index * (width + self.GAP)
            self.canvas.create_image(start, self.HEADER, image=photo, anchor="nw")
            self.canvas.create_text(start + 10, 14, text=title, fill="white", anchor="w", font=("Segoe UI", 10, "bold"))

    def queue_render(self):
        if self.pending is None:
            self.pending = self.root.after(30, self.render)

    def original_point(self, event):
        width, height = self.layout()
        if not (0 <= event.x < width and self.HEADER <= event.y < height + self.HEADER):
            return None
        x, y = (event.x - self.ox) / self.scale, (event.y - self.HEADER - self.oy) / self.scale
        h, w = self.frame.shape[:2]
        return (min(w - 1, int(x)), min(h - 1, int(y))) if 0 <= x < w and 0 <= y < h else None

    def remember(self):
        self.history.append(self.mask.copy())
        self.history = self.history[-20:]

    def start_stroke(self, event):
        if self.frame is None:
            return
        point = self.original_point(event)
        if point is None:
            return
        try:
            radius = self.radius.get()
            if not 1 <= radius <= 200:
                raise ValueError()
        except (tk.TclError, ValueError):
            messagebox.showerror("Brush radius", "Enter a radius between 1 and 200 original-image pixels.")
            return
        self.remember()
        self.stroke, self.last_point = True, point
        self.stroke_radius = radius
        self.stroke_value = 255 if self.mode.get() == "Paint" else 0
        cv2.circle(self.mask, point, radius, self.stroke_value, -1)
        self.dirty = True
        self.queue_render()

    def drag_stroke(self, event):
        if not self.stroke:
            return
        point = self.original_point(event)
        if point is not None:
            if self.last_point is not None:
                cv2.line(self.mask, self.last_point, point, self.stroke_value, 2 * self.stroke_radius + 1)
            cv2.circle(self.mask, point, self.stroke_radius, self.stroke_value, -1)
        self.last_point = point
        self.queue_render()

    def end_stroke(self, event=None):
        if self.stroke:
            self.stroke, self.last_point = False, None
            self.refresh_preview()

    def refresh_preview(self):
        if self.frame is None:
            return
        try:
            self.edited = inpaint(self.frame, self.mask, self.method.get(), self.fill_radius.get())
        except (ValueError, tk.TclError, cv2.error) as error:
            messagebox.showerror("Preview settings", str(error), parent=self.root)
            return
        self.render()
        self.update_status()

    def settings_changed(self):
        if self.frame is not None:
            self.dirty = True
            self.refresh_preview()

    def update_status(self):
        h, w = self.frame.shape[:2]
        index = self.metadata.get("frame_index")
        tag = "image" if index is None else f"frame {index}"
        self.status.set(f"{Path(self.metadata['source_path']).name} | {tag} | {w} x {h} px | "
                        f"{np.count_nonzero(self.mask):,} masked pixels | Check the filled regions and limb outline before inference.")

    def undo(self):
        if self.history and not self.stroke:
            self.mask, self.dirty = self.history.pop(), True
            self.refresh_preview()

    def clear(self):
        if self.frame is not None and np.any(self.mask) and messagebox.askyesno("Clear mask", "Clear all removal regions?", parent=self.root):
            self.remember()
            self.mask.fill(0)
            self.dirty = True
            self.refresh_preview()

    def pan(self, event):
        if self.frame is not None and self.pan_point is not None and not self.stroke:
            self.ox += event.x - self.pan_point[0]
            self.oy += event.y - self.pan_point[1]
            self.pan_point = (event.x, event.y)
            self.render()

    def zoom(self, event, factor=None):
        if self.frame is None or self.stroke or event.y < self.HEADER:
            return
        if factor is None and not event.delta:
            return
        factor = factor or (1.2 if event.delta > 0 else 1 / 1.2)
        width, _ = self.layout()
        local_x = event.x if event.x < width else event.x - width - self.GAP
        if not 0 <= local_x < width:
            return
        local_y = event.y - self.HEADER
        new_scale = min(12.0, max(0.02, self.scale * factor))
        ratio = new_scale / self.scale
        self.ox, self.oy = local_x - (local_x - self.ox) * ratio, local_y - (local_y - self.oy) * ratio
        self.scale = new_scale
        self.render()

    def load_mask(self):
        if self.frame is None:
            messagebox.showinfo("Open a frame", "Open the original frame before loading its saved mask.")
            return
        if not self.proceed():
            return
        path = filedialog.askopenfilename(title="Load session.json for this exact frame", filetypes=[("Session JSON", "*.json")], parent=self.root)
        if path:
            try:
                mask, method, radius = load_session(path, self.frame)
            except Exception as error:
                messagebox.showerror("Could not load mask", str(error), parent=self.root)
                return
            self.remember()
            self.mask, self.dirty = mask, True
            self.method.set(method)
            self.fill_radius.set(radius)
            self.refresh_preview()

    def save(self):
        if self.frame is None:
            return
        self.end_stroke()
        parent = self.args.output or filedialog.askdirectory(title="Choose a local output folder", parent=self.root)
        if not parent:
            return
        try:
            folder = save_session(parent, self.frame, self.mask, self.metadata, self.method.get(), self.fill_radius.get())
        except Exception as error:
            messagebox.showerror("Save failed", str(error), parent=self.root)
            return
        self.dirty = False
        self.status.set(f"Saved: {folder}")
        messagebox.showinfo("Saved", f"Saved original, edited frame, mask, previews, and settings to:\n\n{folder}\n\nUse original.png and edited.png for paired inference.", parent=self.root)

    def close(self):
        if self.proceed():
            self.root.destroy()


def encode_mask(mask):
    ok, encoded = cv2.imencode(".png", mask)
    if not ok:
        raise ValueError("Could not encode mask.")
    return encoded.tobytes()


class ClipSource:
    """Decode on demand with a small cache; don't hold a whole video in RAM."""

    def __init__(self, path, start, count):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file() or self.path.suffix.lower() in IMAGE_SUFFIXES:
            raise ValueError("Clip mode needs an existing video file.")
        if start < 0 or not 1 <= count <= 300:
            raise ValueError("Use a nonnegative first frame and 1-300 frames per clip.")
        self.start, self.count, self.cache, self.states = start, count, OrderedDict(), {}
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            self.close()
            raise ValueError("Could not open the video.")
        total, fps = self.capture.get(cv2.CAP_PROP_FRAME_COUNT), self.capture.get(cv2.CAP_PROP_FPS)
        self.fps = fps if math.isfinite(fps) and fps > 0 else None
        self.backend = self.capture.getBackendName()
        self.next_index = 0
        self.shape = None
        try:
            if math.isfinite(total) and total > 0 and start + count > total:
                raise ValueError(f"Clip extends beyond the video (reported count: {int(total)}).")
            self.shape = self.read(0).shape[:2]
            self.read(count - 1)  # Catch truncated clips before annotation begins.
        except Exception:
            self.close()
            raise

    def close(self):
        if hasattr(self, "capture"):
            self.capture.release()

    def read(self, index):
        if not 0 <= index < self.count:
            raise ValueError("Frame is outside the selected clip.")
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        absolute = self.start + index
        if self.next_index != absolute and not self.capture.set(cv2.CAP_PROP_POS_FRAMES, absolute):
            raise ValueError(f"Cannot seek to source frame {absolute}.")
        ok, frame = self.capture.read()
        reported = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
        if not ok or frame is None or (reported > 0 and abs(reported - absolute - 1) > 0.5):
            raise ValueError(f"Cannot decode the requested source frame {absolute}.")
        if self.shape is not None and frame.shape[:2] != self.shape:
            raise ValueError("Frame dimensions changed inside the clip.")
        self.next_index = absolute + 1
        self.cache[index] = frame
        while len(self.cache) > 3:
            self.cache.popitem(last=False)
        return frame

    def mask(self, index):
        state = self.states.get(index)
        if state is None:
            return np.zeros(self.shape, np.uint8)
        mask = cv2.imdecode(np.frombuffer(state["mask_png"], np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != self.shape:
            raise ValueError("Invalid stored mask.")
        return mask

    def put(self, index, mask, origin="manual", reviewed=False, diagnostics=None):
        if mask.shape != self.shape or mask.dtype != np.uint8:
            raise ValueError("Mask does not match this clip.")
        self.states[index] = {"mask_png": encode_mask(mask), "origin": origin, "reviewed": bool(reviewed),
                              "diagnostics": diagnostics or [], "decoded_bgr_sha256": image_hash(self.read(index))}

    def metadata(self, index):
        return {"source_kind": "video", "source_path": str(self.path), "frame_index": self.start + index,
                "reported_fps": self.fps, "decoder_backend": self.backend}


def track_mask(previous, current, mask, settings=None):
    """Propose independent region translations; drop uncertain regions and flag them."""
    settings = settings or TRACKING_SETTINGS
    if previous.shape != current.shape or mask.shape != previous.shape[:2]:
        raise ValueError("Tracking inputs must have matching dimensions.")
    gray0 = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    proposed, diagnostics = np.zeros_like(mask), []
    if count == 1:
        return proposed, [{"region": None, "accepted": False, "reason": "No source regions; inspect visibility and paint reappearing markers."}]
    height, width = mask.shape
    lk = {"winSize": (settings["window_px"], settings["window_px"]), "maxLevel": settings["pyramid_levels"],
          "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)}
    for label in range(1, count):
        x, y, w, h, area = stats[label].tolist()
        region = (labels == label).astype(np.uint8) * 255
        report = {"region": label, "source_bounds_xywh": [x, y, w, h], "accepted": False}
        diagnostics.append(report)
        try:
            # Prefer corners within the actual removal region, then its immediate neighborhood.
            points = cv2.goodFeaturesToTrack(gray0, maxCorners=24, qualityLevel=0.01, minDistance=2, mask=region)
            if points is None or len(points) < 3:
                neighborhood = cv2.dilate(region, np.ones((7, 7), np.uint8))
                points = cv2.goodFeaturesToTrack(gray0, maxCorners=24, qualityLevel=0.01, minDistance=2, mask=neighborhood)
            if points is None or len(points) < 3:
                raise ValueError("Too few local features.")
            forward, status0, error0 = cv2.calcOpticalFlowPyrLK(gray0, gray1, points, None, **lk)
            if forward is None:
                raise ValueError("Forward tracking failed.")
            safe = forward.copy()
            invalid = ~np.isfinite(safe).all(axis=(1, 2))
            safe[invalid] = points[invalid]
            backward, status1, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, safe, None, **lk)
            if backward is None:
                raise ValueError("Backward tracking failed.")
            source, destination = points[:, 0], forward[:, 0]
            fb = np.linalg.norm(backward[:, 0] - source, axis=1)
            valid = (status0[:, 0] > 0) & (status1[:, 0] > 0) & ~invalid & (fb <= settings["forward_backward_px"])
            valid &= (error0[:, 0] <= 35) & np.isfinite(destination).all(axis=1)
            valid &= (destination[:, 0] >= 0) & (destination[:, 0] < width) & (destination[:, 1] >= 0) & (destination[:, 1] < height)
            if np.count_nonzero(valid) < max(3, math.ceil(len(points) * 0.5)):
                raise ValueError("Too few consistent forward/backward tracks.")
            displacement = destination[valid] - source[valid]
            median = np.median(displacement, axis=0)
            inliers = np.linalg.norm(displacement - median, axis=1) <= 2.5
            if np.count_nonzero(inliers) < max(3, math.ceil(len(displacement) * 0.6)):
                raise ValueError("Local features disagree on movement.")
            dx, dy = np.rint(np.median(displacement[inliers], axis=0)).astype(int).tolist()
            report.update({"translation_px": [dx, dy], "consistent_features": int(np.count_nonzero(inliers))})
            if math.hypot(dx, dy) > settings["max_motion_px"]:
                raise ValueError("Movement exceeds the tracking limit.")
            if x + dx < 0 or y + dy < 0 or x + w + dx > width or y + h + dy > height:
                raise ValueError("Region reaches the image boundary.")
            x0, y0, x1, y1 = max(0, x - 3), max(0, y - 3), min(width, x + w + 3), min(height, y + h + 3)
            if x0 + dx < 0 or y0 + dy < 0 or x1 + dx > width or y1 + dy > height:
                raise ValueError("Patch reaches the image boundary.")
            patch0 = gray0[y0:y1, x0:x1].astype(np.float32)
            patch1 = gray1[y0 + dy:y1 + dy, x0 + dx:x1 + dx].astype(np.float32)
            patch0 -= patch0.mean()
            patch1 -= patch1.mean()
            denominator = float(np.linalg.norm(patch0) * np.linalg.norm(patch1))
            correlation = float(np.sum(patch0 * patch1) / denominator) if denominator > 1e-6 else -1.0
            report["patch_correlation"] = correlation
            if correlation < settings["min_patch_correlation"]:
                raise ValueError("Patch appearance changed; inspect occlusion or drift.")
            shifted = cv2.warpAffine(region, np.float32([[1, 0, dx], [0, 1, dy]]), (width, height), flags=cv2.INTER_NEAREST)
            proposed = cv2.bitwise_or(proposed, shifted)
            report.update({"accepted": True, "reason": "Passed proposal checks; visual review still required."})
        except (ValueError, cv2.error) as error:
            report["reason"] = str(error)
    return proposed, diagnostics


def propagate_clip(clip, start):
    """Yield progress and return the last frame; preserve manually corrected anchors."""
    last = start
    for index in range(start + 1, clip.count):
        existing = clip.states.get(index)
        if existing and (existing["origin"] == "manual" or existing["reviewed"]):
            last = index
            yield f"Preserved corrected/reviewed frame {index + 1}/{clip.count}"
            continue
        mask, reports = track_mask(clip.read(index - 1), clip.read(index), clip.mask(index - 1))
        clip.put(index, mask, "tracked", False, reports)
        last = index
        yield f"Tracked frame {index + 1}/{clip.count}"
        if any(not report["accepted"] for report in reports):
            return last
    return last


def clip_document(clip, method, radius, kind):
    h, w = clip.shape
    return {"schema_version": CLIP_SCHEMA_VERSION, "kind": kind, "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_path": str(clip.path), "start_frame": clip.start, "frame_count": clip.count,
            "reported_fps": clip.fps, "width": w, "height": h, "inpaint_method": method,
            "inpaint_radius_px": float(radius), "opencv_version": cv2.__version__,
            "tracking_settings": dict(TRACKING_SETTINGS), "frames": [],
            "note": "Digitally edited marker-removal pilot. PNG frames are inference inputs; lossy preview videos are for inspection."}


def make_preview_writer(folder, name, fps, size):
    """Try MP4, then AVI; PNG export does not depend on a video codec being installed."""
    for suffix, codec in [(".mp4", "mp4v"), (".avi", "MJPG")]:
        path = folder / (name + suffix)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, path.name
        writer.release()
        if path.exists():
            path.unlink()
    return None, None


def save_clip_job(output, clip, method, radius, export=False):
    """Checkpoint or reviewed export, yielding after each frame so the UI can respond."""
    inpaint(clip.read(0), clip.mask(0), method, radius)
    if export:
        unreviewed = [i + 1 for i in range(clip.count) if not clip.states.get(i, {}).get("reviewed")]
        if unreviewed:
            raise ValueError(f"Review every frame first. Unreviewed clip frames include: {unreviewed[:10]}")
        if not any(np.any(clip.mask(i)) for i in range(clip.count)):
            raise ValueError("All masks are empty; there is no edited comparison to export.")
    kind = "clip_export" if export else "clip_project"
    parent = Path(output).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    folder = parent / f"{clip.path.stem}_{kind}_{clip.start:06d}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    folder.mkdir(exist_ok=False)
    (folder / "masks").mkdir()
    if export:
        (folder / "original_frames").mkdir()
        (folder / "edited_frames").mkdir()
    document = clip_document(clip, method, radius, kind)
    writers, complete = [], False
    preview_width = min(640, clip.shape[1])
    preview_width -= preview_width % 2
    preview_width = max(2, preview_width)
    preview_height = max(2, int(clip.shape[0] * preview_width / clip.shape[1]) // 2 * 2)
    preview_size = (preview_width, preview_height)
    document["preview_videos"] = []
    try:
        if export and clip.fps:
            for name, size in [("original_preview", preview_size), ("edited_preview", preview_size),
                               ("comparison_preview", (2 * preview_width, preview_height + 30))]:
                writer, filename = make_preview_writer(folder, name, clip.fps, size)
                writers.append(writer)
                if filename:
                    document["preview_videos"].append(filename)
            if any(writer is None for writer in writers):
                document["preview_warning"] = "A playback codec was unavailable for one or more previews. PNG frame sequences are complete."
        elif export:
            document["preview_warning"] = "Video did not report a valid FPS; preview videos omitted."
        for index in range(clip.count):
            original, mask = clip.read(index), clip.mask(index)
            state = clip.states.get(index, {"origin": "unset", "reviewed": False, "diagnostics": []})
            digest = image_hash(original)
            if state.get("decoded_bgr_sha256", digest) != digest:
                raise ValueError(f"Source frame {clip.start + index} changed since annotation.")
            name = f"frame_{index:06d}.png"
            write_png(folder / "masks" / name, mask)
            record = {"clip_frame": index, "source_frame": clip.start + index, "filename": name,
                      "decoded_bgr_sha256": digest, "mask_sha256": image_hash(mask),
                      "mask_pixels": int(np.count_nonzero(mask)), "origin": state["origin"],
                      "reviewed": bool(state["reviewed"]), "diagnostics": state["diagnostics"]}
            document["frames"].append(record)
            if export:
                edited = inpaint(original, mask, method, radius)
                write_png(folder / "original_frames" / name, original)
                write_png(folder / "edited_frames" / name, edited)
                if writers:
                    small_original = cv2.resize(original, preview_size, interpolation=cv2.INTER_AREA)
                    small_edited = cv2.resize(edited, preview_size, interpolation=cv2.INTER_AREA)
                    left = cv2.resize(mask_overlay(original, mask), preview_size, interpolation=cv2.INTER_AREA)
                    paired = np.vstack([np.zeros((30, preview_width * 2, 3), np.uint8), np.hstack([left, small_edited])])
                    cv2.putText(paired, f"MASK / frame {clip.start + index}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(paired, "EDITED", (preview_width + 8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    for writer, image in zip(writers, [small_original, small_edited, paired]):
                        if writer is not None:
                            writer.write(image)
            yield f"{'Exporting' if export else 'Saving project'} frame {index + 1}/{clip.count}"
        with (folder / "frame_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["clip_frame", "source_frame", "filename", "mask_pixels", "origin", "reviewed", "decoded_bgr_sha256", "mask_sha256"]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(document["frames"])
        (folder / "clip.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
        complete = True
    finally:
        for writer in writers:
            if writer is not None:
                writer.release()
        if not complete:
            (folder / "SAVE_INCOMPLETE.txt").write_text("Export/save was cancelled or failed. Do not use this folder for inference.", encoding="utf-8")
    return folder


def load_clip_project(path, source_override=None):
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != CLIP_SCHEMA_VERSION or document.get("kind") not in {"clip_project", "clip_export"}:
        raise ValueError("Select clip.json from a saved clip project or export.")
    if (path.parent / "SAVE_INCOMPLETE.txt").exists():
        raise ValueError("This project/export is incomplete.")
    clip = ClipSource(source_override or document["source_path"], document["start_frame"], document["frame_count"])
    try:
        if clip.shape != (document["height"], document["width"]) or len(document["frames"]) != clip.count:
            raise ValueError("Project dimensions or frame count do not match.")
        for index, record in enumerate(document["frames"]):
            if record["clip_frame"] != index or record["source_frame"] != clip.start + index:
                raise ValueError("Project frame ordering does not match.")
            if image_hash(clip.read(index)) != record["decoded_bgr_sha256"]:
                raise ValueError(f"Source frame {clip.start + index} differs from the saved project.")
            mask_path = path.parent / "masks" / f"frame_{index:06d}.png"
            mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != clip.shape or image_hash(mask) != record["mask_sha256"]:
                raise ValueError(f"Saved mask {index} is missing or changed.")
            if record["origin"] != "unset":
                clip.put(index, mask, record["origin"], record["reviewed"], record["diagnostics"])
        method, radius = document["inpaint_method"], float(document["inpaint_radius_px"])
        inpaint(clip.read(0), clip.mask(0), method, radius)
        return clip, method, radius
    except Exception:
        clip.close()
        raise


class ClipEditor(MarkerEditor):
    """Extend the existing frame editor; the same brush and fill controls still apply."""

    def __init__(self, root, args):
        self.clip, self.clip_index = None, 0
        self.busy = self.playing = self.cancel_requested = False
        self.clip_dirty, self.ignore_timeline, self.play_job = False, False, None
        self.last_settings = ("Telea", 3.0)
        source, args.source = args.source, None
        super().__init__(root, args)
        args.source = source
        root.title("Marker mask editor - frames and short clips")
        row = ttk.Frame(root, padding=(8, 0, 8, 6))
        row.pack(fill="x", before=self.canvas)
        for label, command in [("Open clip...", self.open_clip_dialog), ("Open project...", self.open_project),
                               ("Save project...", self.save_project), ("Track forward", self.track_forward),
                               ("Prev", lambda: self.go(-1)), ("Next", lambda: self.go(1)),
                               ("Review + next", self.review_next), ("Play / pause", self.toggle_play),
                               ("Export clip...", self.export_clip)]:
            ttk.Button(row, text=label, command=command).pack(side="left", padx=2)
        self.cancel_button = ttk.Button(row, text="Cancel task", command=lambda: setattr(self, "cancel_requested", True))
        self.cancel_button.pack(side="left", padx=2)
        timeline_row = ttk.Frame(root, padding=(8, 0, 8, 6))
        timeline_row.pack(fill="x", before=self.canvas)
        self.frame_label = tk.StringVar(value="Single-frame mode")
        ttk.Label(timeline_row, textvariable=self.frame_label, width=47).pack(side="left")
        self.timeline = ttk.Scale(timeline_row, from_=0, to=1, command=self.timeline_changed)
        self.timeline.pack(side="left", fill="x", expand=True, padx=8)
        self.progress = ttk.Progressbar(timeline_row, mode="determinate", length=130)
        self.progress.pack(side="right", padx=3)
        if source:
            action = lambda: self.open_clip(source, args.frame, args.frames) if args.frames > 1 else self.open_source(source, args.frame)
            root.after(80, action)

    def proceed(self):
        if self.busy:
            return False
        if self.playing:
            self.stop_play()
        dirty = self.clip_dirty if self.clip is not None else self.dirty
        return not dirty or messagebox.askyesno("Unsaved edits", "Discard unsaved edits? Save project first to retain clip progress.", parent=self.root)

    def open_source(self, path, index):
        if self.busy:
            return
        # Decode before replacing the current clip, so a failed open preserves progress.
        try:
            read_frame(path, index)
        except Exception as error:
            messagebox.showerror("Could not open frame", str(error), parent=self.root)
            return
        self.stop_play()
        if self.clip is not None:
            self.clip.close()
        self.clip, self.clip_dirty = None, False
        self.frame_label.set("Single-frame mode")
        super().open_source(path, index)

    def open_clip_dialog(self):
        if not self.proceed():
            return
        from tkinter.simpledialog import askinteger
        path = filedialog.askopenfilename(title="Choose a local video", parent=self.root)
        if not path:
            return
        start = askinteger("Clip start", "First source frame (zero-based):", initialvalue=0, minvalue=0, parent=self.root)
        if start is None:
            return
        count = askinteger("Clip length", "Number of consecutive frames (try 30-60):", initialvalue=60, minvalue=1, maxvalue=300, parent=self.root)
        if count is not None:
            self.open_clip(path, start, count)

    def open_clip(self, path, start, count):
        if self.busy:
            return
        try:
            clip = ClipSource(path, start, count)
        except Exception as error:
            messagebox.showerror("Could not open clip", str(error), parent=self.root)
            return
        self.install_clip(clip)

    def install_clip(self, clip):
        self.stop_play()
        if self.clip is not None:
            self.clip.close()
        self.clip, self.clip_dirty, self.dirty = clip, False, False
        self.last_settings = (self.method.get(), self.fill_radius.get())
        self.ignore_timeline = True
        self.timeline.configure(to=max(1, clip.count - 1))
        self.timeline.set(0)
        self.ignore_timeline = False
        self.set_frame(0)
        self.fit()

    def set_frame(self, index):
        if self.clip is None or self.busy or self.stroke or not 0 <= index < self.clip.count:
            return
        try:
            frame = self.clip.read(index)
        except Exception as error:
            self.stop_play()
            messagebox.showerror("Frame decode failed", str(error), parent=self.root)
            return
        self.clip_index, self.frame = index, frame
        self.metadata, self.mask = self.clip.metadata(index), self.clip.mask(index)
        self.history, self.dirty = [], False
        self.ignore_timeline = True
        self.timeline.set(index)
        self.ignore_timeline = False
        self.refresh_preview()

    def timeline_changed(self, value):
        if not self.ignore_timeline and not self.busy and not self.playing and self.clip is not None:
            self.set_frame(min(self.clip.count - 1, int(float(value) + 0.5)))

    def go(self, offset):
        if not self.busy and self.clip is not None:
            self.stop_play()
            self.set_frame(self.clip_index + offset)

    def mark_manual(self):
        if self.clip is not None:
            self.clip.put(self.clip_index, self.mask, "manual", False)
            self.clip_dirty = True
            self.update_status()

    def start_stroke(self, event):
        if not self.busy and not self.playing:
            super().start_stroke(event)

    def end_stroke(self, event=None):
        changed = self.stroke
        super().end_stroke(event)
        if changed:
            self.mark_manual()

    def undo(self):
        if self.busy or self.playing:
            return
        changed = bool(self.history) and not self.stroke
        super().undo()
        if changed:
            self.mark_manual()

    def clear(self):
        if self.busy or self.playing:
            return
        previous = self.mask.copy() if self.mask is not None else None
        super().clear()
        if previous is not None and not np.array_equal(previous, self.mask):
            self.mark_manual()

    def settings_changed(self):
        if self.busy:
            return
        try:
            settings = (self.method.get(), self.fill_radius.get())
        except tk.TclError:
            return super().settings_changed()
        if settings != self.last_settings and self.clip is not None:
            for state in self.clip.states.values():
                state["reviewed"] = False
            self.clip_dirty = True
        self.last_settings = settings
        super().settings_changed()

    def load_mask(self):
        if self.busy or self.playing:
            return
        if self.clip is None:
            return super().load_mask()
        path = filedialog.askopenfilename(title="Load a single-frame session.json for this exact frame", filetypes=[("JSON", "*.json")], parent=self.root)
        if not path:
            return
        try:
            mask, method, radius = load_session(path, self.frame)
        except Exception as error:
            messagebox.showerror("Could not load mask", str(error), parent=self.root)
            return
        self.remember()
        self.mask, self.dirty = mask, True
        self.method.set(method)
        self.fill_radius.set(radius)
        self.settings_changed()
        self.mark_manual()

    def review_next(self):
        if self.busy or self.playing or self.clip is None or self.stroke:
            return
        # Explicit confirmation accepts the current mask, including intentionally empty ones.
        state = self.clip.states.get(self.clip_index)
        reports = state["diagnostics"] if state else []
        self.clip.put(self.clip_index, self.mask, "manual", True, reports)
        self.clip_dirty = True
        self.update_status()
        self.set_frame(min(self.clip.count - 1, self.clip_index + 1))

    def update_status(self):
        super().update_status()
        if self.clip is None:
            return
        state = self.clip.states.get(self.clip_index, {})
        reviewed = sum(bool(s["reviewed"]) for s in self.clip.states.values())
        flag = "REVIEWED" if state.get("reviewed") else "NEEDS REVIEW"
        self.frame_label.set(f"Clip {self.clip_index + 1}/{self.clip.count} | source {self.clip.start + self.clip_index} | {flag}")
        self.progress.configure(maximum=self.clip.count, value=reviewed)
        failed = [d for d in state.get("diagnostics", []) if not d["accepted"]]
        extra = f" | {reviewed}/{self.clip.count} reviewed"
        if failed and not state.get("reviewed"):
            extra += f" | TRACKING STOP: {failed[0]['reason']}"
        self.status.set(self.status.get() + extra)

    def stop_play(self):
        self.playing = False
        if self.play_job is not None:
            self.root.after_cancel(self.play_job)
            self.play_job = None

    def toggle_play(self):
        if self.busy or self.clip is None or self.stroke:
            return
        if self.playing:
            self.stop_play()
            return
        self.playing = True
        if self.clip_index == self.clip.count - 1:
            self.set_frame(0)
        self.play_tick()

    def play_tick(self):
        self.play_job = None
        if not self.playing:
            return
        if self.clip_index >= self.clip.count - 1:
            self.playing = False
            return
        self.set_frame(self.clip_index + 1)
        if self.playing:
            interval = max(20, int(1000 / (self.clip.fps or 30)))
            self.play_job = self.root.after(interval, self.play_tick)

    def set_busy(self, enabled):
        self.busy = enabled
        if enabled:
            self.widget_states = []
            def walk(parent):
                for widget in parent.winfo_children():
                    if isinstance(widget, (ttk.Button, ttk.Spinbox, ttk.Combobox, ttk.Radiobutton, ttk.Checkbutton, ttk.Scale)) and widget is not self.cancel_button:
                        self.widget_states.append((widget, "disabled" in widget.state()))
                        widget.state(["disabled"])
                    walk(widget)
            walk(self.root)
        else:
            for widget, disabled in self.widget_states:
                widget.state(["disabled"] if disabled else ["!disabled"])

    def run_job(self, generator, on_done):
        self.stop_play()
        self.cancel_requested = False
        self.set_busy(True)
        def tick():
            try:
                if self.cancel_requested:
                    generator.close()
                    self.set_busy(False)
                    self.status.set("Task cancelled. Completed masks remain available; interrupted output folders are marked incomplete.")
                    return
                self.status.set(next(generator))
                self.root.after(1, tick)
            except StopIteration as result:
                self.set_busy(False)
                on_done(result.value)
            except Exception as error:
                generator.close()
                self.set_busy(False)
                messagebox.showerror("Task failed", str(error), parent=self.root)
                self.update_status()
        self.root.after(1, tick)

    def track_forward(self):
        if self.busy or self.clip is None or self.stroke:
            return
        self.stop_play()
        if self.clip_index not in self.clip.states:
            self.clip.put(self.clip_index, self.mask, "manual", False)
        self.clip_dirty = True
        def finished(index):
            self.set_frame(index)
            state = self.clip.states.get(index, {})
            if not state.get("reviewed") and any(not d["accepted"] for d in state.get("diagnostics", [])):
                messagebox.showinfo("Tracking needs correction", "Stopped at a frame with an uncertain region. Paint/erase the visible marker regions here, then Track forward again. Inspect reappearing markers manually.", parent=self.root)
        self.run_job(propagate_clip(self.clip, self.clip_index), finished)

    def output_parent(self):
        return self.args.output or filedialog.askdirectory(title="Choose a local output parent folder", parent=self.root)

    def save_project(self):
        self.write_clip(False)

    def export_clip(self):
        self.write_clip(True)

    def write_clip(self, export):
        if self.busy or self.clip is None or self.stroke:
            return
        self.stop_play()
        try:
            method, radius = self.method.get(), self.fill_radius.get()
            inpaint(self.frame, self.mask, method, radius)
        except Exception as error:
            messagebox.showerror("Fill settings", str(error), parent=self.root)
            return
        if export:
            missing = [i + 1 for i in range(self.clip.count) if not self.clip.states.get(i, {}).get("reviewed")]
            if missing:
                self.set_frame(missing[0] - 1)
                messagebox.showinfo("Review remaining frames", f"Use Review + next on every inspected frame. {len(missing)} frame(s) still need review.", parent=self.root)
                return
        parent = self.output_parent()
        if not parent:
            return
        def finished(folder):
            self.clip_dirty = False
            self.status.set(f"Saved: {folder}")
            messagebox.showinfo("Saved", f"{'Clip exported' if export else 'Project saved'} to:\n\n{folder}\n\n" + ("Use original_frames and edited_frames for paired inference; videos are inspection previews." if export else "Use Open project... and select clip.json to resume."), parent=self.root)
        self.run_job(save_clip_job(parent, self.clip, method, radius, export), finished)

    def open_project(self):
        if not self.proceed():
            return
        path = filedialog.askopenfilename(title="Open clip.json", filetypes=[("JSON", "*.json")], parent=self.root)
        if not path:
            return
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
            override = None
            if not Path(document.get("source_path", "")).is_file():
                override = filedialog.askopenfilename(title="Locate the original video", parent=self.root)
                if not override:
                    return
            clip, method, radius = load_clip_project(path, override)
        except Exception as error:
            messagebox.showerror("Could not open project", str(error), parent=self.root)
            return
        self.method.set(method)
        self.fill_radius.set(radius)
        self.last_settings = (method, radius)
        self.install_clip(clip)

    def save(self):
        if self.busy:
            return
        if self.clip is not None:
            self.export_clip()
        else:
            super().save()

    def close(self):
        if self.proceed():
            self.stop_play()
            if self.clip is not None:
                self.clip.close()
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Paint and preview marker-removal masks locally.", epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", help="Image or video path; omit to select a file in the window.")
    parser.add_argument("--frame", type=int, default=0, help="Zero-based video frame index (default: 0).")
    parser.add_argument("--frames", type=int, default=1, help="Clip length in frames, 1-300 (default: single-frame mode).")
    parser.add_argument("--output", type=Path, help="Local output parent folder; omit to choose at each save.")
    args = parser.parse_args()
    if args.frame < 0 or not 1 <= args.frames <= 300 or ((args.frame or args.frames > 1) and not args.source):
        parser.error("Use a video source for --frame/--frames; frame must be nonnegative and frames must be 1-300.")
    try:
        root = tk.Tk()
    except tk.TclError as error:
        sys.exit(f"Could not start the desktop window: {error}\nRun locally with a Python installation that includes tkinter.")
    ClipEditor(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
