import os
import math
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import torch
from torchvision import transforms
from PIL import Image, ImageTk
import numpy as np
import cv2

from models import ResNet101Iris, DenseNet201Iris


def get_model(model_name, checkpoint_path, num_classes=1500):
    model = None
    input_size = 224

    if model_name == "resnet101":
        model = ResNet101Iris(num_classes=num_classes)
        input_size = 224
    elif model_name == "densenet201":
        model = DenseNet201Iris(num_classes=num_classes)
        input_size = 224
    else:
        raise ValueError("Only 'resnet101' and 'densenet201' are supported in this demo")

    # Load weights onto CPU first to avoid CUDA deserialization issues
    state = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(state)
    return model, input_size


def build_transform(input_size):
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def load_tensor_image(path, input_size):
    img = Image.open(path)
    # If TIFF is multi-frame, take the first frame
    try:
        if getattr(img, "is_animated", False):
            img.seek(0)
    except Exception:
        pass
    img = img.convert('RGB')
    t = build_transform(input_size)(img)
    return t.unsqueeze(0)  # [1, C, H, W]


def _detect_circles(gray):
    h, w = gray.shape[:2]
    pupil = None
    iris = None
    try:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=h//8,
                                   param1=120, param2=15, minRadius=max(8, h//40), maxRadius=h//4)
        if circles is not None:
            circles = np.around(circles[0]).astype(int)
            best = None
            best_mean = 1e9
            for x, y, r in circles:
                x, y, r = int(x), int(y), int(r)
                x0, x1 = max(0, x - r), min(w, x + r)
                y0, y1 = max(0, y - r), min(h, y + r)
                roi = gray[y0:y1, x0:x1]
                m = roi.mean() if roi.size else 1e9
                if m < best_mean:
                    best_mean = m
                    best = (x, y, r)
            pupil = best
    except Exception:
        pass

    try:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=h//8,
                                   param1=150, param2=30, minRadius=(pupil[2]+10 if pupil else h//8), maxRadius=h//2)
        if circles is not None:
            x, y, r = np.around(circles[0][0]).astype(int)
            iris = (int(x), int(y), int(r))
    except Exception:
        pass

    if pupil is None:
        cx, cy = w//2, h//2
        pupil = (cx, cy, max(8, h//20))
    if iris is None:
        iris = (pupil[0], pupil[1], min(h//2-1, int(pupil[2]*3)))
    cx = int(0.5*(pupil[0] + iris[0]))
    cy = int(0.5*(pupil[1] + iris[1]))
    rp = int(pupil[2])
    ri = int(iris[2])
    if ri <= rp:
        ri = rp + max(5, h//12)
    return (cx, cy, rp), (cx, cy, ri)


def preprocess_raw_to_stacked_rgb(pil_img, radial_res=64, angular_res=256):
    img = np.array(pil_img.convert('L'))
    img_blur = cv2.medianBlur(img, 5)
    pupil, iris = _detect_circles(img_blur)
    cx, cy, rp = pupil
    _, _, ri = iris

    thetas = np.linspace(0, 2*math.pi, angular_res, endpoint=False)
    rs = np.linspace(rp, ri, radial_res)
    map_x = np.zeros((radial_res, angular_res), dtype=np.float32)
    map_y = np.zeros((radial_res, angular_res), dtype=np.float32)
    for j, th in enumerate(thetas):
        map_x[:, j] = cx + rs * np.cos(th)
        map_y[:, j] = cy + rs * np.sin(th)
    polar = cv2.remap(img_blur, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    polar = clahe.apply(polar)

    stacked = np.tile(polar, (4, 1))
    stacked = np.clip(stacked, 0, 255).astype(np.uint8)
    stacked_rgb = cv2.cvtColor(stacked, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(stacked_rgb)


def extract_feature(model, x, device, model_name):
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        if hasattr(model, 'feature_extract_avg_pool'):
            feat = model.feature_extract_avg_pool(x)
        else:
            # Fallback: global average pooling minus final classifier (not expected here)
            outputs = model(x)
            feat = outputs
        feat = feat.cpu().numpy().astype(np.float32).reshape(-1)
        # L2 normalize
        norm = np.linalg.norm(feat) + 1e-12
        return feat / norm


class DemoGUI:
    def __init__(self, root):
        self.root = root
        root.title("Iris Recognition Demo (Open-set)")

        # Model selection panel
        model_frame = ttk.LabelFrame(root, text="Model")
        model_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        model_frame.columnconfigure(1, weight=1)

        ttk.Label(model_frame, text="Model name").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar(value="resnet101")
        self.model_cb = ttk.Combobox(model_frame, textvariable=self.model_var, values=["resnet101", "densenet201"], state="readonly")
        self.model_cb.grid(row=0, column=1, sticky="ew", padx=6)

        ttk.Label(model_frame, text="Checkpoint").grid(row=1, column=0, sticky="w")
        self.ckpt_var = tk.StringVar(value="./models/resnet101_e_80_lr_2e-05_best.pth")
        self.ckpt_entry = ttk.Entry(model_frame, textvariable=self.ckpt_var)
        self.ckpt_entry.grid(row=1, column=1, sticky="ew", padx=6)
        ttk.Button(model_frame, text="Browse", command=self.browse_ckpt).grid(row=1, column=2, padx=6)

        # Enrollment panel (5 people)
        enroll_frame = ttk.LabelFrame(root, text="Enrollment (5 people)")
        enroll_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        enroll_frame.columnconfigure(2, weight=1)

        self.enroll_names = []
        self.enroll_paths = []
        for i in range(5):
            ttk.Label(enroll_frame, text=f"Person {i+1} name").grid(row=i, column=0, sticky="w")
            name_var = tk.StringVar(value=f"person{i+1}")
            path_var = tk.StringVar(value="")
            self.enroll_names.append(name_var)
            self.enroll_paths.append(path_var)

            ttk.Entry(enroll_frame, textvariable=name_var, width=20).grid(row=i, column=1, padx=4, sticky="w")
            ttk.Entry(enroll_frame, textvariable=path_var).grid(row=i, column=2, padx=4, sticky="ew")
            ttk.Button(enroll_frame, text="Select image", command=lambda idx=i: self.browse_enroll(idx)).grid(row=i, column=3, padx=4)

        # Query panel
        query_frame = ttk.LabelFrame(root, text="Query (1 image to identify)")
        query_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=6)
        query_frame.columnconfigure(1, weight=1)

        self.query_path = tk.StringVar(value="")
        ttk.Label(query_frame, text="Image").grid(row=0, column=0, sticky="w")
        ttk.Entry(query_frame, textvariable=self.query_path).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(query_frame, text="Select image", command=self.browse_query).grid(row=0, column=2, padx=6)

        self.preprocess_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(query_frame, text="Preprocess raw image (segment + normalize + stack)", variable=self.preprocess_var, command=self.update_all_previews).grid(row=1, column=0, columnspan=3, sticky="w", pady=4)

        self.show_images_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(query_frame, text="Show all preview images", variable=self.show_images_var, command=self.toggle_all_images).grid(row=2, column=0, columnspan=3, sticky="w", pady=2)

        # Run button and output
        run_frame = ttk.Frame(root)
        run_frame.grid(row=3, column=0, sticky="ew", padx=8, pady=6)
        ttk.Button(run_frame, text="Run Identification", command=self.run_demo).grid(row=0, column=0, padx=4)

        # Open-set threshold
        ttk.Label(run_frame, text="Threshold (reject if <)").grid(row=0, column=1, sticky="w")
        self.threshold_var = tk.DoubleVar(value=0.85)
        ttk.Entry(run_frame, textvariable=self.threshold_var, width=8).grid(row=0, column=2, padx=6, sticky="w")

        self.out_text = tk.Text(root, height=10)
        self.out_text.grid(row=4, column=0, sticky="nsew", padx=8, pady=6)
        root.rowconfigure(4, weight=1)
        root.columnconfigure(0, weight=1)

        self.grid_prev_frame = ttk.Frame(root)
        self.grid_prev_frame.grid(row=5, column=0, sticky="ew", padx=8)
        self.grid_top_widgets = []
        self.grid_bot_widgets = []
        self.grid_name_widgets = []
        self._grid_top_tk = [None]*6
        self._grid_bot_tk = [None]*6
        for c in range(6):
            top = tk.Label(self.grid_prev_frame)
            bot = tk.Label(self.grid_prev_frame)
            name_lbl = ttk.Label(self.grid_prev_frame, text="")
            top.grid(row=0, column=c, padx=6, pady=(0,2))
            bot.grid(row=1, column=c, padx=6, pady=(0,2))
            name_lbl.grid(row=2, column=c, padx=6, pady=(0,6))
            self.grid_top_widgets.append(top)
            self.grid_bot_widgets.append(bot)
            self.grid_name_widgets.append(name_lbl)

        # Initialize and keep names synced
        for i in range(5):
            self.grid_name_widgets[i].configure(text=self.enroll_names[i].get().strip() or f"person{i+1}")
            self.enroll_names[i].trace_add('write', lambda *_args, idx=i: self.grid_name_widgets[idx].configure(text=self.enroll_names[idx].get().strip() or f"person{idx+1}"))
        self.grid_name_widgets[5].configure(text="query")

        self.preview_frame = ttk.Frame(root)
        self.preview_frame.grid(row=6, column=0, sticky="ew", padx=8)
        ttk.Label(self.preview_frame, text="Original:").grid(row=0, column=0, sticky="w")
        ttk.Label(self.preview_frame, text="Preprocessed:").grid(row=0, column=1, sticky="w", padx=(20,0))
        self.orig_img_widget = tk.Label(self.preview_frame)
        self.orig_img_widget.grid(row=1, column=0, padx=0, pady=4)
        self.proc_img_widget = tk.Label(self.preview_frame)
        self.proc_img_widget.grid(row=1, column=1, padx=(20,0), pady=4)
        self._orig_tk = None
        self._proc_tk = None
        # Hide the large preview area as requested
        self.preview_frame.grid_remove()

    def toggle_all_images(self):
        if self.show_images_var.get():
            self.grid_prev_frame.grid()
            self.update_all_previews()
        else:
            self.grid_prev_frame.grid_remove()

    def browse_ckpt(self):
        path = filedialog.askopenfilename(title="Select checkpoint .pth", filetypes=[("PyTorch checkpoint", "*.pth"), ("All files", "*.*")])
        if path:
            self.ckpt_var.set(path)

    def browse_enroll(self, idx):
        path = filedialog.askopenfilename(
            title=f"Select enrollment image for person {idx+1}",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")]
        )
        if path:
            self.enroll_paths[idx].set(path)
            self.update_enroll_preview(idx)

    def browse_query(self):
        path = filedialog.askopenfilename(
            title="Select query image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")]
        )
        if path:
            self.query_path.set(path)
            self.update_query_preview()

    def log(self, msg):
        self.out_text.insert(tk.END, msg + "\n")
        self.out_text.see(tk.END)

    def update_query_preview(self):
        path = self.query_path.get().strip()
        if not path or not os.path.isfile(path):
            return
        pil = Image.open(path)
        try:
            if getattr(pil, "is_animated", False):
                pil.seek(0)
        except Exception:
            pass
        pil = pil.convert('RGB')
        if self.preprocess_var.get():
            proc = preprocess_raw_to_stacked_rgb(pil)
        else:
            proc = pil.copy()

        def to_tk(im):
            max_w = 320
            w, h = im.size
            if w > max_w:
                ratio = max_w / float(w)
                im = im.resize((int(w*ratio), int(h*ratio)))
            return ImageTk.PhotoImage(im)

        self._orig_tk = to_tk(pil)
        self._proc_tk = to_tk(proc)
        self.orig_img_widget.configure(image=self._orig_tk)
        self.proc_img_widget.configure(image=self._proc_tk)

        # Also update grid column 5 (query as 6th tile)
        def to_tk_small(im):
            max_w = 120
            w, h = im.size
            if w > max_w:
                ratio = max_w / float(w)
                im = im.resize((int(w*ratio), int(h*ratio)))
            return ImageTk.PhotoImage(im)
        self._grid_top_tk[5] = to_tk_small(pil)
        self._grid_bot_tk[5] = to_tk_small(proc)
        self.grid_top_widgets[5].configure(image=self._grid_top_tk[5])
        self.grid_bot_widgets[5].configure(image=self._grid_bot_tk[5])
        self.grid_name_widgets[5].configure(text="query")

    def update_enroll_preview(self, idx):
        if idx < 0 or idx >= 5:
            return
        path = self.enroll_paths[idx].get().strip()
        if not path or not os.path.isfile(path):
            return
        pil = Image.open(path)
        try:
            if getattr(pil, "is_animated", False):
                pil.seek(0)
        except Exception:
            pass
        pil = pil.convert('RGB')
        if self.preprocess_var.get():
            proc = preprocess_raw_to_stacked_rgb(pil)
        else:
            proc = pil.copy()

        def to_tk(im):
            max_w = 120
            w, h = im.size
            if w > max_w:
                ratio = max_w / float(w)
                im = im.resize((int(w*ratio), int(h*ratio)))
            return ImageTk.PhotoImage(im)

        # Update 2-row x 6-col grid at column idx
        self._grid_top_tk[idx] = to_tk(pil)
        self._grid_bot_tk[idx] = to_tk(proc)
        self.grid_top_widgets[idx].configure(image=self._grid_top_tk[idx])
        self.grid_bot_widgets[idx].configure(image=self._grid_bot_tk[idx])
        self.grid_name_widgets[idx].configure(text=self.enroll_names[idx].get().strip() or f"person{idx+1}")

    def update_all_enroll_previews(self):
        for i in range(5):
            self.update_enroll_preview(i)

    def update_all_previews(self):
        self.update_all_enroll_previews()
        self.update_query_preview()

    def run_demo(self):
        try:
            model_name = self.model_var.get()
            ckpt_path = self.ckpt_var.get()
            if not os.path.isfile(ckpt_path):
                messagebox.showerror("Error", f"Checkpoint not found: {ckpt_path}")
                return

            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.log(f"Device: {device}")
            self.log(f"Loading model: {model_name}")
            model, input_size = get_model(model_name, ckpt_path)
            model.to(device)
            self.log("Model loaded.")

            # Collect enrollment entries
            enrollment = []
            for i in range(5):
                name = self.enroll_names[i].get().strip()
                img_path = self.enroll_paths[i].get().strip()
                if name and img_path and os.path.isfile(img_path):
                    enrollment.append((name, img_path))
            if len(enrollment) == 0:
                messagebox.showwarning("Warning", "Please provide at least one enrollment image with name.")
                return

            if not self.query_path.get().strip() or not os.path.isfile(self.query_path.get().strip()):
                messagebox.showwarning("Warning", "Please select a query image.")
                return

            # Extract features
            self.log("Extracting enrollment features...")
            feats = {}
            for name, path in enrollment:
                if self.preprocess_var.get():
                    pil = Image.open(path)
                    try:
                        if getattr(pil, "is_animated", False):
                            pil.seek(0)
                    except Exception:
                        pass
                    pil = pil.convert('RGB')
                    pil_proc = preprocess_raw_to_stacked_rgb(pil)
                    x = build_transform(input_size)(pil_proc).unsqueeze(0)
                else:
                    x = load_tensor_image(path, input_size)
                feat = extract_feature(model, x, device, model_name)
                feats[name] = feat
                self.log(f"Enrolled: {name} <- {path}")

            self.log("Extracting query feature...")
            if self.preprocess_var.get():
                pil_q = Image.open(self.query_path.get().strip())
                try:
                    if getattr(pil_q, "is_animated", False):
                        pil_q.seek(0)
                except Exception:
                    pass
                pil_q = pil_q.convert('RGB')
                pil_qp = preprocess_raw_to_stacked_rgb(pil_q)
                xq = build_transform(input_size)(pil_qp).unsqueeze(0)
            else:
                xq = load_tensor_image(self.query_path.get().strip(), input_size)
            q = extract_feature(model, xq, device, model_name)

            # Compare by cosine similarity
            self.log("Computing similarities...")
            sims = []
            for name, f in feats.items():
                sim = float(np.dot(q, f))  # cosine after L2 norm
                self.log(f"Similarity to {name}: {sim:.4f}")
                sims.append((name, sim))

            sims.sort(key=lambda x: x[1], reverse=True)
            best_name, best_sim = sims[0]
            second_sim = sims[1][1] if len(sims) > 1 else -1.0
            margin = best_sim - second_sim if second_sim >= 0 else best_sim

            thr = float(self.threshold_var.get())
            self.log("---- Result ----")
            if best_sim < thr:
                self.log(f"Predicted identity: UNKNOWN  (best={best_sim:.4f}, second={second_sim:.4f}, margin={margin:.4f}, thr={thr:.3f})")
            else:
                self.log(f"Predicted identity: {best_name}  (similarity={best_sim:.4f}, second={second_sim:.4f}, margin={margin:.4f}, thr={thr:.3f})")
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = DemoGUI(root)
    root.mainloop()
