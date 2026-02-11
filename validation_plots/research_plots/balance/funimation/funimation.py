from __future__ import annotations

import json
import sqlite3
import socket
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------
# EDIT THESE
# ------------------------
db_path = Path("validation.db")

path_to_viewer_folder = Path(r"C:\Users\aaron\Documents\GitHub\freemocap_validation\validation_plots\research_plots\balance\funimation")  # <- folder containing index.html + sway_viewer.js

TRIAL_NAME = "2025-07-31_16-00-42_GMT-4_jsm_nih_trial_1"
TRACKER = "mediapipe"

CONDITIONS = {
    "Eyes Open / Solid Ground": ("Eyes Open/Solid Ground_x", "Eyes Open/Solid Ground_y"),
    "Eyes Closed / Foam": ("Eyes Closed/Foam_x", "Eyes Closed/Foam_y"),
}


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def compute_velocity(ml: np.ndarray, ap: np.ndarray) -> np.ndarray:
    dml = np.gradient(ml)
    dap = np.gradient(ap)
    vel = np.sqrt(dml**2 + dap**2)
    if len(vel) >= 5:
        vel = np.convolve(vel, np.ones(5) / 5, mode="same")
    return vel


def load_final_df(conn: sqlite3.Connection) -> pd.DataFrame:
    query = """
    SELECT t.participant_code,
           t.trial_name,
           a.path,
           a.component_name,
           a.condition,
           a.tracker
    FROM artifacts a
    JOIN trials t ON a.trial_id = t.id
    WHERE t.trial_type = "balance"
      AND a.category = "com_analysis"
      AND a.tracker IN ("mediapipe", "qualisys")
      AND a.file_exists = 1
      AND a.component_name LIKE '%balance_positions'
    ORDER BY t.trial_name, a.path;
    """
    path_df = pd.read_sql_query(query, conn)

    dfs = []
    for _, row in path_df.iterrows():
        p = Path(row["path"])
        if not p.exists():
            continue
        sub = pd.read_csv(p)
        sub["trial_name"] = row["trial_name"]
        sub["tracker"] = row["tracker"]
        dfs.append(sub)

    if not dfs:
        raise RuntimeError("No balance_positions CSVs found (or paths missing).")
    return pd.concat(dfs, ignore_index=True)


def make_sway_payload(final_df: pd.DataFrame) -> bytes:
    df = final_df[(final_df["trial_name"] == TRIAL_NAME) & (final_df["tracker"] == TRACKER)].copy()
    if df.empty:
        raise RuntimeError(f"No data found for trial={TRIAL_NAME} tracker={TRACKER}")

    if "Frame" in df.columns:
        df = df.sort_values("Frame").reset_index(drop=True)
        t = df["Frame"].to_numpy(dtype=float)
    else:
        df = df.reset_index(drop=True)
        t = np.arange(len(df), dtype=float)

    out = {"trial_name": TRIAL_NAME, "tracker": TRACKER, "conditions": {}}

    for label, (ml_col, ap_col) in CONDITIONS.items():
        ml = df[ml_col].to_numpy(dtype=float)
        ap = df[ap_col].to_numpy(dtype=float)
        mask = ~(np.isnan(ml) | np.isnan(ap))
        ml, ap, tt = ml[mask], ap[mask], t[mask]

        # center for visuals
        ml = ml - np.nanmean(ml)
        ap = ap - np.nanmean(ap)

        vel = compute_velocity(ml, ap)

        # time-as-height (0..200)
        zz = (tt - tt.min()) / max(1.0, (tt.max() - tt.min())) * 200.0

        out["conditions"][label] = {
            "t": tt.tolist(),
            "ml": ml.tolist(),
            "ap": ap.tolist(),
            "z": zz.tolist(),
            "vel": vel.tolist(),
        }

    return json.dumps(out).encode("utf-8")


def run_server():
    viewer_dir = path_to_viewer_folder.resolve()

    # sanity checks (same style as your working server) :contentReference[oaicite:4]{index=4}
    for fname in ("index.html", "sway_viewer.js"):
        if not (viewer_dir / fname).exists():
            raise FileNotFoundError(f"Missing {fname} in {viewer_dir}")

    conn = sqlite3.connect(db_path)
    final_df = load_final_df(conn)
    sway_bytes = make_sway_payload(final_df)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(viewer_dir), **kwargs)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/sway.json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(sway_bytes)))
                self.end_headers()
                self.wfile.write(sway_bytes)
                return
            return super().do_GET()

    port = pick_free_port()
    host = "127.0.0.1"
    url = f"http://{host}:{port}/index.html"

    print(f"Viewer folder: {viewer_dir}")
    print(f"Sway JSON:     http://{host}:{port}/sway.json")
    print(f"Open:          {url}")

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Timer(0.25, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
