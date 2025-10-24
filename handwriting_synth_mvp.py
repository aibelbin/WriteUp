#!/usr/bin/env python3
"""
handwriting_synth_mvp.py

Full pipeline (modes):
 - preprocess_only
 - strokes_preview
 - prepare_style_dataset
 - convert_lines_to_strokes
 - train_style_encoder

Assumes torch is installed if you run train_style_encoder.
Dependencies:
 pip install pymupdf opencv-python-headless numpy scikit-image pillow
 Optional for OCR: pytesseract
"""

import os
import math
import argparse
import numpy as np
import cv2
import fitz  # PyMuPDF
from skimage.morphology import skeletonize

# ---------------------------
# Utility: load PDF pages
# ---------------------------
def load_pdf_pages(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        # pix.samples is a bytes buffer; reshape accordingly
        arr = np.frombuffer(pix.samples, dtype=np.uint8)
        if pix.n == 1:
            img = arr.reshape(pix.height, pix.width)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = arr.reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        yield i + 1, img

# ---------------------------
# Preprocessing: deskew, binarize, crop
# ---------------------------
def deskew_image(gray):
    # expects gray uint8 image
    coords = np.column_stack(np.where(gray < 255))
    if coords.size == 0:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    # adjust angle semantics
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def preprocess_image(img):
    """
    Input: BGR image (numpy)
    Output: binary image (uint8) where ink ~ 255, background ~ 0 (adaptiveThreshold INV)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # deskew first
    gray = deskew_image(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 35, 11)
    # find handwriting bounding box and crop
    coords = cv2.findNonZero(thresh)
    if coords is None:
        # nothing detected, return small blank
        h, w = thresh.shape
        return np.zeros((min(256,h), min(256,w)), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = thresh[y:y+h, x:x+w]
    return cropped

# ---------------------------
# Skeleton -> polylines utility
# ---------------------------
def skeleton_to_polylines(skel):
    """
    skel: 2D binary numpy array (1 where skeleton exists)
    Returns list of polylines, each polyline is Nx2 float32 np.array (x, y)
    This is a heuristic walk (no explicit stroke-order correctness).
    """
    h, w = skel.shape
    sk_map = (skel > 0).astype(np.uint8)
    visited = np.zeros_like(sk_map, dtype=bool)
    neighbors8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    def get_neighbors(pt):
        y, x = pt
        n = []
        for dy, dx in neighbors8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and sk_map[ny, nx] > 0:
                n.append((ny, nx))
        return n

    pts = set(map(tuple, np.column_stack(np.where(sk_map > 0)).tolist()))  # (y,x)
    if not pts:
        return []

    # endpoints and junctions
    endpoints = [p for p in pts if len(get_neighbors(p)) != 2]
    starts = endpoints + list(pts - set(endpoints))

    polylines = []
    for s in starts:
        if visited[s]:
            continue
        cur = s
        poly = [(cur[1], cur[0])]  # (x,y)
        visited[cur] = True
        prev = None
        while True:
            neigh = [n for n in get_neighbors(cur) if not visited[n]]
            if not neigh:
                break
            if prev is None:
                nxt = neigh[0]
            else:
                # choose neighbor with minimal angle change
                best = neigh[0]
                best_angle = None
                vx = cur[1] - prev[1]
                vy = cur[0] - prev[0]
                for cand in neigh:
                    cx = cand[1] - cur[1]
                    cy = cand[0] - cur[0]
                    dot = vx * cx + vy * cy
                    mag = math.hypot(vx, vy) * math.hypot(cx, cy) + 1e-9
                    angle = math.acos(max(-1, min(1, dot / mag)))
                    if best_angle is None or angle < best_angle:
                        best_angle = angle
                        best = cand
                nxt = best
            prev = cur
            cur = nxt
            if visited[cur]:
                break
            visited[cur] = True
            poly.append((cur[1], cur[0]))
        if len(poly) >= 2:
            polylines.append(np.array(poly, dtype=np.float32))
    return polylines

# ---------------------------
# Mode: prepare_style_dataset (line segmentation + normalize)
# ---------------------------
def prepare_style_dataset(pdf_path, target_h=64):
    os.makedirs("data/style_source", exist_ok=True)
    for page_num, img in load_pdf_pages(pdf_path):
        processed = preprocess_image(img)  # ink=255 background=0
        h, w = processed.shape
        row_sum = np.sum(processed > 0, axis=1)
        min_pixels = max(2, int(w * 0.002))
        mask = row_sum > min_pixels
        runs = []
        start = None
        for i, v in enumerate(mask):
            if v and start is None:
                start = i
            elif not v and start is not None:
                runs.append((start, i - 1))
                start = None
        if start is not None:
            runs.append((start, len(mask) - 1))
        outdir = f"data/style_source/page_{page_num}"
        os.makedirs(outdir, exist_ok=True)
        line_idx = 1
        for (r0, r1) in runs:
            pad = 6
            y0 = max(0, r0 - pad)
            y1 = min(h - 1, r1 + pad)
            line_crop = processed[y0:y1+1, :]
            # invert: ink=0, bg=255
            line_crop = 255 - line_crop
            # trim empty columns
            col_sum = np.sum(line_crop < 255, axis=0)
            if np.max(col_sum) == 0:
                continue
            c_idxs = np.where(col_sum > 0)[0]
            cx0, cx1 = c_idxs[0], c_idxs[-1]
            line_crop = line_crop[:, cx0:cx1+1]
            lh, lw = line_crop.shape
            scale = target_h / float(lh)
            new_w = max(1, int(round(lw * scale)))
            line_resized = cv2.resize(line_crop, (new_w, target_h), interpolation=cv2.INTER_AREA)
            out_path = os.path.join(outdir, f"line_{line_idx:02d}.npy")
            np.save(out_path, line_resized.astype(np.uint8))
            print(f"Saved {out_path}")
            line_idx += 1
        if line_idx == 1:
            # fallback: save whole page as single line if segmentation failed
            out_path = os.path.join(outdir, f"line_01.npy")
            full_img = 255 - processed
            scale = target_h / float(full_img.shape[0])
            new_w = max(1, int(round(full_img.shape[1] * scale)))
            resized = cv2.resize(full_img, (new_w, target_h), interpolation=cv2.INTER_AREA)
            np.save(out_path, resized.astype(np.uint8))
            print(f"Saved fallback {out_path}")

# ---------------------------
# Mode: convert_lines_to_strokes (skeleton -> polylines)
# ---------------------------
def convert_lines_to_strokes():
    stroke_out_dir = "data/strokes"
    os.makedirs(stroke_out_dir, exist_ok=True)
    src_root = "data/style_source"
    if not os.path.isdir(src_root):
        print("No data/style_source found. Run prepare_style_dataset first.")
        return
    for page_dir in sorted(os.listdir(src_root)):
        page_path = os.path.join(src_root, page_dir)
        if not os.path.isdir(page_path):
            continue
        out_page_dir = os.path.join(stroke_out_dir, page_dir)
        os.makedirs(out_page_dir, exist_ok=True)
        for fname in sorted(os.listdir(page_path)):
            if not fname.endswith('.npy'):
                continue
            arr = np.load(os.path.join(page_path, fname))  # ink=0, bg=255
            # binarize (foreground True where ink < 255)
            bw = (arr < 255).astype(np.uint8)
            if bw.sum() == 0:
                print(f"Warning: empty line {fname}")
                continue
            sk = skeletonize(bw).astype(np.uint8)
            polylines = skeleton_to_polylines(sk)
            save_dict = {}
            for i, stroke in enumerate(polylines):
                save_dict[f"stroke_{i}"] = stroke
            out_name = os.path.join(out_page_dir, fname.replace('.npy', '.npz'))
            if save_dict:
                np.savez_compressed(out_name, **save_dict)
                print(f"Saved strokes -> {out_name} (num_strokes={len(save_dict)})")
            else:
                print(f"Warning: no strokes found for {fname}")

# ---------------------------
# Simple visualization helpers
# ---------------------------
def save_preprocessed_pages(pdf_path, out_prefix="out_preprocessed_page"):
    for page_num, img in load_pdf_pages(pdf_path):
        processed = preprocess_image(img)
        out_path = f"{out_prefix}_{page_num}.png"
        cv2.imwrite(out_path, processed)
        print(f"Saved {out_path}")

def save_strokes_preview(pdf_path, out_prefix="out_strokes_page", thickness=2):
    for page_num, img in load_pdf_pages(pdf_path):
        processed = preprocess_image(img)
        bw = (processed > 0).astype(np.uint8)
        sk = skeletonize(bw).astype(np.uint8)
        # thicken for visibility
        vis = cv2.dilate((sk * 255).astype(np.uint8), np.ones((thickness, thickness), np.uint8), iterations=1)
        # ensure white background, black strokes
        canvas = 255 - vis
        out_path = f"{out_prefix}_{page_num}.png"
        cv2.imwrite(out_path, canvas)
        print(f"Saved {out_path}")

# ---------------------------
# Minimal PyTorch style encoder scaffold
# ---------------------------
def train_style_encoder(strokes_root="data/strokes", epochs=5, batch_size=8, max_points=2048):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
    except Exception as e:
        print("PyTorch not available. Install torch to run training.")
        raise

    class StrokeDataset(Dataset):
        def __init__(self, strokes_root, max_points=2048):
            self.files = []
            for page in sorted(os.listdir(strokes_root)):
                pdir = os.path.join(strokes_root, page)
                if not os.path.isdir(pdir):
                    continue
                for fn in sorted(os.listdir(pdir)):
                    if fn.endswith('.npz'):
                        self.files.append(os.path.join(pdir, fn))
            self.max_points = max_points

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            path = self.files[idx]
            data = np.load(path)
            strokes = []
            for k in data.files:
                strokes.append(data[k])
            pts = []
            for s in strokes:
                for p in s:
                    pts.append([p[0], p[1], 1.0])  # pen-down
                pts.append([0.0, 0.0, 0.0])  # pen-up separator
            pts = np.array(pts, dtype=np.float32)
            if pts.shape[0] == 0:
                pts = np.zeros((1,3), dtype=np.float32)
            max_coord = max(1.0, np.max(np.abs(pts[:,:2])))
            pts[:,:2] = pts[:,:2] / float(max_coord)
            if pts.shape[0] >= self.max_points:
                pts = pts[:self.max_points]
            else:
                pad = np.zeros((self.max_points - pts.shape[0], 3), dtype=np.float32)
                pts = np.vstack([pts, pad])
            return torch.from_numpy(pts), torch.from_numpy(pts)

    class StyleEncoder(nn.Module):
        def __init__(self, input_dim=3, hidden=256, n_layers=2, emb_size=128):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, num_layers=n_layers, batch_first=True, bidirectional=True)
            self.fc = nn.Sequential(
                nn.Linear(hidden * 2, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, emb_size)
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            pooled = out.mean(dim=1)
            emb = self.fc(pooled)
            return emb

    dataset = StrokeDataset(strokes_root, max_points=max_points)
    if len(dataset) == 0:
        print("No stroke files found in data/strokes — run convert_lines_to_strokes first.")
        return
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StyleEncoder().to(device)
    decoder = nn.Sequential(nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, 3 * max_points)).to(device)
    optim = torch.optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            emb = model(xb)
            recon = decoder(emb)
            recon = recon.view(xb.size(0), max_points, 3)
            loss = loss_fn(recon, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            tot += loss.item() * xb.size(0)
        print(f"Epoch {ep+1}/{epochs} loss={tot/len(dataset):.6f}")
    torch.save(model.state_dict(), "style_encoder.pth")
    print("Saved style encoder to style_encoder.pth")

# ---------------------------
# CLI / main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="handwriting_synth_mvp")
    parser.add_argument("--pdf", required=True, help="Input PDF containing handwriting pages")
    parser.add_argument("--mode", required=True, choices=[
        "preprocess_only", "strokes_preview", "prepare_style_dataset",
        "convert_lines_to_strokes", "train_style_encoder"
    ], help="Operation mode")
    args = parser.parse_args()

    if args.mode == "preprocess_only":
        save_preprocessed_pages(args.pdf, out_prefix="out_preprocessed_page")
    elif args.mode == "strokes_preview":
        save_strokes_preview(args.pdf, out_prefix="out_strokes_page", thickness=3)
    elif args.mode == "prepare_style_dataset":
        prepare_style_dataset(args.pdf, target_h=64)
    elif args.mode == "convert_lines_to_strokes":
        convert_lines_to_strokes()
    elif args.mode == "train_style_encoder":
        train_style_encoder(strokes_root="data/strokes", epochs=5, batch_size=8, max_points=2048)
    else:
        print("Unknown mode")

if __name__ == "__main__":
    main()
