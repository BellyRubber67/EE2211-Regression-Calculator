import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr
import backend

class RegressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Regression Model App")
        self.root.geometry("900x750") 
        
        # --- Variables ---
        self.model_type = tk.StringVar(value="Linear")
        self.split_x = tk.StringVar(value="5")
        self.tree_depth = tk.StringVar(value="1")
        self.want_manual_split = tk.BooleanVar(value=False)
        self.want_w = tk.BooleanVar(value=True)
        self.want_pred = tk.BooleanVar(value=True)
        self.want_mse = tk.BooleanVar(value=False)
        self.want_train_mse = tk.BooleanVar(value=False)
        self.want_r = tk.BooleanVar(value=False)
        self.want_p = tk.BooleanVar(value=False)
        
        # OHE Variables
        self.want_ohe = tk.BooleanVar(value=False)
        self.ohe_format = tk.StringVar(value="raw")
        
        # Lists to hold the entry widgets for our matrices
        self.X_train_entries = []
        self.y_train_entries = []
        self.X_test_entries = []
        self.y_test_entries = []

        # History of previous inputs (snapshots)
        self.input_history = []  # list[dict]
        self._pending_prefill = None

        self.setup_ui()

    def setup_ui(self):
        # --- 1. Dimensions ---
        dim_frame = ttk.LabelFrame(self.root, text="1. Matrix Dimensions", padding=(10, 5))
        dim_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(dim_frame, text="Train Rows (m):").grid(row=0, column=0, padx=5, pady=5)
        self.train_m = ttk.Entry(dim_frame, width=5)
        self.train_m.insert(0, "3")
        self.train_m.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(dim_frame, text="Test Rows:").grid(row=0, column=4, padx=5, pady=5)
        self.test_m = ttk.Entry(dim_frame, width=5)
        self.test_m.insert(0, "2")
        self.test_m.grid(row=0, column=5, padx=5, pady=5)

        ttk.Label(dim_frame, text="Features (n):").grid(row=0, column=2, padx=5, pady=5)
        self.features_n = ttk.Entry(dim_frame, width=5)
        self.features_n.insert(0, "2")
        self.features_n.grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(dim_frame, text="Targets (Non-OHE):").grid(row=0, column=6, padx=5, pady=5)
        self.targets_n = ttk.Entry(dim_frame, width=5)
        self.targets_n.insert(0, "1")
        self.targets_n.grid(row=0, column=7, padx=5, pady=5)

        # --- 2. One-Hot Encoding Settings ---
        ohe_frame = ttk.LabelFrame(self.root, text="2. One-Hot Encoding Target Preparation", padding=(10, 5))
        ohe_frame.pack(fill="x", padx=10, pady=5)

        ttk.Checkbutton(ohe_frame, text="Enable One-Hot", variable=self.want_ohe, command=self.toggle_ohe_options).grid(row=0, column=0, padx=5)

        ttk.Label(ohe_frame, text="Format:").grid(row=0, column=1, padx=5)
        self.ohe_combo = ttk.Combobox(ohe_frame, textvariable=self.ohe_format, values=["raw", "encoded"], state="disabled", width=10)
        self.ohe_combo.grid(row=0, column=2, padx=5)
        self.ohe_combo.bind("<<ComboboxSelected>>", self.toggle_ohe_options)

        ttk.Label(ohe_frame, text="Classes (Columns):").grid(row=0, column=3, padx=5)
        self.ohe_classes = ttk.Entry(ohe_frame, width=5, state="disabled")
        self.ohe_classes.insert(0, "3")
        self.ohe_classes.grid(row=0, column=4, padx=5)

        ttk.Button(ohe_frame, text="Generate Input Grids", command=self.generate_grids).grid(row=0, column=5, padx=20)

        # --- 3. Model Selection & Outputs ---
        settings_frame = ttk.LabelFrame(self.root, text="3. Settings & Outputs", padding=(10, 5))
        settings_frame.pack(fill="x", padx=10, pady=5)

        ttk.Radiobutton(settings_frame, text="Linear", variable=self.model_type, value="Linear").grid(row=0, column=0, padx=5)
        ttk.Radiobutton(settings_frame, text="Polynomial", variable=self.model_type, value="Poly").grid(row=0, column=1, padx=5)
        ttk.Radiobutton(settings_frame, text="Decision Tree", variable=self.model_type, value="DecisionTree").grid(row=0, column=2, padx=5)

        ttk.Label(settings_frame, text="Order:").grid(row=0, column=3, padx=5)
        self.poly_order = ttk.Entry(settings_frame, width=5)
        self.poly_order.insert(0, "2")
        self.poly_order.grid(row=0, column=4, padx=5)

        ttk.Label(settings_frame, text="Lambda:").grid(row=0, column=5, padx=5)
        self.reg_factor = ttk.Entry(settings_frame, width=5)
        self.reg_factor.insert(0, "0.0")
        self.reg_factor.grid(row=0, column=6, padx=5)

        ttk.Separator(settings_frame, orient='vertical').grid(row=0, column=7, sticky='ns', padx=10)

        ttk.Label(settings_frame, text="Depth:").grid(row=0, column=8, padx=5)
        self.tree_depth_entry = ttk.Entry(settings_frame, textvariable=self.tree_depth, width=5, state="disabled")
        self.tree_depth_entry.grid(row=0, column=9, padx=5)

        self.manual_split_cb = ttk.Checkbutton(
            settings_frame,
            text="Manual split",
            variable=self.want_manual_split,
            command=self.toggle_model_options,
        )
        self.manual_split_cb.grid(row=0, column=10, padx=5)
        self.manual_split_cb.state(['disabled'])

        ttk.Label(settings_frame, text="Split x:").grid(row=0, column=11, padx=5)
        self.split_x_entry = ttk.Entry(settings_frame, textvariable=self.split_x, width=7, state="disabled")
        self.split_x_entry.grid(row=0, column=12, padx=5)

        # Put output toggles on a new row so they don't run off-screen on small windows
        ttk.Separator(settings_frame, orient='horizontal').grid(row=1, column=0, columnspan=13, sticky='ew', pady=(6, 2))

        ttk.Checkbutton(settings_frame, text="W Array", variable=self.want_w).grid(row=2, column=0, padx=5, pady=(2, 0), sticky="w")
        ttk.Checkbutton(settings_frame, text="Predictions", variable=self.want_pred).grid(row=2, column=1, padx=5, pady=(2, 0), sticky="w")

        self.mse_cb = ttk.Checkbutton(settings_frame, text="Test MSE", variable=self.want_mse)
        self.mse_cb.grid(row=2, column=2, padx=5, pady=(2, 0), sticky="w")

        self.train_mse_cb = ttk.Checkbutton(settings_frame, text="Train MSE", variable=self.want_train_mse)
        self.train_mse_cb.grid(row=2, column=3, padx=5, pady=(2, 0), sticky="w")

        ttk.Checkbutton(settings_frame, text="R", variable=self.want_r).grid(row=2, column=4, padx=5, pady=(2, 0), sticky="w")
        ttk.Checkbutton(settings_frame, text="P", variable=self.want_p).grid(row=2, column=5, padx=5, pady=(2, 0), sticky="w")

        self.model_type.trace_add("write", lambda *args: self.toggle_model_options())
        self.toggle_model_options()
        self.update_test_mse_availability()

        # --- 4. Matrix Input Area ---
        self.matrix_frame = ttk.Frame(self.root)
        self.matrix_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # --- 5. History ---
        history_frame = ttk.LabelFrame(self.root, text="5. Input History", padding=(10, 5))
        history_frame.pack(fill="x", padx=10, pady=5)

        self.history_list = tk.Listbox(history_frame, height=4)
        self.history_list.pack(side="left", fill="both", expand=True, padx=(0, 10))

        history_btns = ttk.Frame(history_frame)
        history_btns.pack(side="left", fill="y")
        ttk.Button(history_btns, text="Load Selected", command=self.load_selected_history).pack(fill="x", pady=(0, 5))
        ttk.Button(history_btns, text="Clear History", command=self.clear_history).pack(fill="x")

        # --- 6. Run Button & Output Area ---
        ttk.Button(self.root, text="RUN MODEL", command=self.run_model).pack(pady=5)
        
        self.result_text = tk.Text(self.root, height=10)
        self.result_text.pack(fill="both", expand=True, padx=10, pady=10)

    # UI Logic for Checkboxes/Dropdowns
    def toggle_model_options(self):
        is_tree = self.model_type.get() == "DecisionTree"

        # Decision tree controls only active in DecisionTree mode
        self.tree_depth_entry.config(state="normal" if is_tree else "disabled")
        if is_tree:
            self.manual_split_cb.state(['!disabled'])
        else:
            self.want_manual_split.set(False)
            self.manual_split_cb.state(['disabled'])

        # Manual split input only active when DecisionTree + manual split checked
        if is_tree and self.want_manual_split.get():
            self.split_x_entry.config(state="normal")
            # Manual split doesn't produce a w vector
            self.want_w.set(False)
        else:
            self.split_x_entry.config(state="disabled")

    def update_test_mse_availability(self):
        """Enable Test MSE only when there are test rows."""
        try:
            test_m = int(self.test_m.get())
        except Exception:
            test_m = 0

        if test_m > 0:
            self.mse_cb.state(['!disabled'])
        else:
            self.want_mse.set(False)
            self.mse_cb.state(['disabled'])

    def toggle_ohe_options(self, *args):
        if self.want_ohe.get():
            self.ohe_combo.state(['!disabled'])
            self.ohe_combo.config(state="readonly")
            
            if self.ohe_format.get() == "encoded":
                self.ohe_classes.state(['!disabled'])
                self.ohe_classes.config(state="normal")
            else:
                self.ohe_classes.state(['disabled'])
                self.ohe_classes.config(state="disabled")
        else:
            self.ohe_combo.state(['disabled'])
            self.ohe_combo.config(state="disabled")
            self.ohe_classes.state(['disabled'])
            self.ohe_classes.config(state="disabled")

    def _entries_to_strings(self, entries):
        if not entries:
            return None
        return [[e.get() for e in row] for row in entries]

    def _add_history_snapshot(self, reason):
        if not self.X_train_entries:
            return

        snapshot = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reason": reason,
            "dims": {
                "train_m": self.train_m.get().strip(),
                "test_m": self.test_m.get().strip(),
                "features_n": self.features_n.get().strip(),
                "targets_n": self.targets_n.get().strip(),
                "want_ohe": bool(self.want_ohe.get()),
                "ohe_format": self.ohe_format.get(),
                "ohe_classes": self.ohe_classes.get().strip(),
            },
            "data": {
                "X_train": self._entries_to_strings(self.X_train_entries),
                "y_train": self._entries_to_strings(self.y_train_entries),
                "X_test": self._entries_to_strings(self.X_test_entries),
                "y_test": self._entries_to_strings(self.y_test_entries),
            },
        }
        self.input_history.append(snapshot)
        self.refresh_history_list()

    def refresh_history_list(self):
        if not hasattr(self, "history_list"):
            return
        self.history_list.delete(0, tk.END)
        for idx, snap in enumerate(self.input_history):
            dims = snap.get("dims", {})
            desc = (
                f"{idx+1}. {snap.get('timestamp', '')} "
                f"[{snap.get('reason', '')}] "
                f"train={dims.get('train_m','?')}x{dims.get('features_n','?')} "
                f"test={dims.get('test_m','?')}x{dims.get('features_n','?')} "
                f"Y={dims.get('targets_n','?')}"
            )
            self.history_list.insert(tk.END, desc)

    def clear_history(self):
        self.input_history = []
        self.refresh_history_list()

    def load_selected_history(self):
        if not self.input_history:
            return
        sel = self.history_list.curselection()
        if not sel:
            messagebox.showinfo("History", "Select a snapshot to load.")
            return
        snap = self.input_history[sel[0]]

        dims = snap.get("dims", {})
        # Restore dimension + OHE settings (best-effort, keep current if missing)
        self.train_m.delete(0, tk.END); self.train_m.insert(0, dims.get("train_m", self.train_m.get()))
        self.test_m.delete(0, tk.END); self.test_m.insert(0, dims.get("test_m", self.test_m.get()))
        self.features_n.delete(0, tk.END); self.features_n.insert(0, dims.get("features_n", self.features_n.get()))
        self.targets_n.delete(0, tk.END); self.targets_n.insert(0, dims.get("targets_n", self.targets_n.get()))

        self.want_ohe.set(bool(dims.get("want_ohe", self.want_ohe.get())))
        self.ohe_format.set(dims.get("ohe_format", self.ohe_format.get()))
        self.ohe_classes.delete(0, tk.END); self.ohe_classes.insert(0, dims.get("ohe_classes", self.ohe_classes.get()))
        self.toggle_ohe_options()

        self._pending_prefill = snap.get("data", None)
        self.generate_grids()

    def _clip_2d(self, values, max_rows, max_cols):
        if not values:
            return None
        clipped = []
        for r in range(min(max_rows, len(values))):
            row = values[r] or []
            clipped.append([row[c] if c < len(row) else "" for c in range(min(max_cols, len(row)))])
        return clipped

    def generate_grids(self):
        # Save current inputs before regenerating (so nothing is lost)
        self._add_history_snapshot("resize")

        # Capture previous values for overlap-retention
        prev = {
            "X_train": self._entries_to_strings(self.X_train_entries),
            "y_train": self._entries_to_strings(self.y_train_entries),
            "X_test": self._entries_to_strings(self.X_test_entries),
            "y_test": self._entries_to_strings(self.y_test_entries),
        }

        for widget in self.matrix_frame.winfo_children():
            widget.destroy()

        try:
            m = int(self.train_m.get())
            test_m = int(self.test_m.get())
            n = int(self.features_n.get())
            
            # Determine Target Columns (Y)
            if self.want_ohe.get():
                if self.ohe_format.get() == "raw":
                    t = 1 # Raw input forces 1 column
                else:
                    t = int(self.ohe_classes.get()) # Encoded uses specified classes
            else:
                t = int(self.targets_n.get())

        except ValueError:
            messagebox.showerror("Error", "Please enter valid integers for dimensions.")
            return

        # Update checkbox availability based on whether test set exists
        self.update_test_mse_availability()

        # Decide what to prefill: explicit history-load wins, else retain overlap from current grid
        prefill = self._pending_prefill if self._pending_prefill is not None else prev
        self._pending_prefill = None

        X_train_prefill = self._clip_2d(prefill.get("X_train"), m, n) if prefill else None
        y_train_prefill = self._clip_2d(prefill.get("y_train"), m, t) if prefill else None
        X_test_prefill = self._clip_2d(prefill.get("X_test"), test_m, n) if prefill else None
        y_test_prefill = self._clip_2d(prefill.get("y_test"), test_m, t) if prefill else None

        self.X_train_entries = self.create_grid(self.matrix_frame, "Training X", rows=m, cols=n, grid_row=0, grid_col=0, initial_values=X_train_prefill)
        self.y_train_entries = self.create_grid(self.matrix_frame, "Training Y", rows=m, cols=t, grid_row=0, grid_col=1, is_y=True, initial_values=y_train_prefill)
        self.X_test_entries = self.create_grid(self.matrix_frame, "Test X", rows=test_m, cols=n, grid_row=1, grid_col=0, initial_values=X_test_prefill)
        self.y_test_entries = self.create_grid(self.matrix_frame, "Test Y", rows=test_m, cols=t, grid_row=1, grid_col=1, is_y=True, initial_values=y_test_prefill)

    def create_grid(self, parent, title, rows, cols, grid_row, grid_col, is_y=False, initial_values=None):
        frame = ttk.LabelFrame(parent, text=title)
        frame.grid(row=grid_row, column=grid_col, padx=10, pady=10, sticky="nw")
        
        entries = []
        for i in range(rows):
            row_entries = []
            for j in range(cols):
                e = ttk.Entry(frame, width=6)
                e.grid(row=i, column=j, padx=2, pady=2)

                # Prefill retained values (if any)
                if initial_values is not None and i < len(initial_values) and j < len(initial_values[i]):
                    val = initial_values[i][j]
                    if val is not None and str(val) != "":
                        e.insert(0, str(val))
                
                # Apply OHE Specific UI behaviors to Y targets
                if self.want_ohe.get() and is_y:
                    if self.ohe_format.get() == "raw":
                        if not e.get().strip():
                            e.insert(0, "class")
                        e.bind("<FocusIn>", lambda event, widget=e: self.clear_placeholder(event, widget))
                    elif self.ohe_format.get() == "encoded":
                        e.bind("<KeyRelease>", lambda event, row=i, col=j, widget_list=row_entries: self.auto_fill_ohe(event, row, col, widget_list))
                
                row_entries.append(e)
            entries.append(row_entries)
        return entries

    def clear_placeholder(self, event, widget):
        if widget.get() == "class":
            widget.delete(0, tk.END)

    def auto_fill_ohe(self, event, row, col, row_entries):
        widget = event.widget
        val = widget.get().strip()
        if val == '1':
            for idx, entry in enumerate(row_entries):
                if idx != col:
                    entry.delete(0, tk.END)
                    entry.insert(0, "0")

    def get_matrix_data(self, entries, is_y=False):
        if not entries:
            return None
        
        data = []
        for row in entries:
            row_data = []
            for e in row:
                val = e.get().strip()
                if val == "class" or not val:
                    row_data.append(0.0) 
                else:
                    try:
                        num = float(val)
                        if self.want_ohe.get() and is_y:
                            num = int(num) # Enforce integer parsing if OHE is enabled
                        row_data.append(num)
                    except ValueError:
                        row_data.append(0.0)
            data.append(row_data)
            
        return np.array(data)

    def calculate_mse(self, y_true, predictions, format_type):
        """Helper function to align shapes before calculating MSE"""
        y_comp = np.copy(y_true)
        preds_comp = np.copy(predictions)
        
        if format_type == "argmax":
            if y_comp.ndim > 1 and y_comp.shape > 1:
                y_comp = np.argmax(y_comp, axis=1)
            return np.mean((y_comp.flatten() - preds_comp.flatten()) ** 2)
            
        return np.mean((y_comp - preds_comp) ** 2)

    def calculate_r_x_to_y(self, X_train, y_train):
        """
        Pearson correlation (r) between X_train and y_train.

        This follows the simple interpretation: flatten both arrays and pass them
        directly into scipy.stats.pearsonr.

        Returns:
          r: float
          p: float
        """
        x = np.asarray(X_train).reshape(-1)
        y = np.asarray(y_train).reshape(-1)

        k = min(x.size, y.size)
        x = x[:k]
        y = y[:k]

        try:
            r, p = pearsonr(x, y)
        except Exception:
            r, p = np.nan, np.nan
        return r, p

    def run_model(self):
        self.result_text.delete(1.0, tk.END) 
        
        if not self.X_train_entries:
            messagebox.showwarning("Warning", "Please generate and fill the grids first.")
            return

        try:
            self._add_history_snapshot("run")
            # 1. Gather Data
            X_train = self.get_matrix_data(self.X_train_entries)
            y_train = self.get_matrix_data(self.y_train_entries, is_y=True)
            X_test = self.get_matrix_data(self.X_test_entries)
            y_test = self.get_matrix_data(self.y_test_entries, is_y=True)

            # For R: always use Training X vs Training Y as entered
            y_train_for_r = np.copy(y_train)

            # --- One Hot Encoding Backend Transformation ---
            if self.want_ohe.get() and self.ohe_format.get() == "raw":
                # FIX: Fit ONLY on y_train. handle_unknown='ignore' ensures that if 
                # y_test has empty cells (parsed as 0), it doesn't crash the encoder.
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoder.fit(y_train)
                
                y_train = encoder.transform(y_train)
                y_test = encoder.transform(y_test)

            order = int(self.poly_order.get())
            reg = float(self.reg_factor.get())

            # --- Determine Predict Format ---
            pred_format = "raw"
            if self.want_ohe.get():
                num_classes = y_train.shape[1]
                if num_classes == 2:
                    pred_format = "sign"
                elif num_classes > 2:
                    pred_format = "argmax"

            # 2. Call Midterm Backend
            if self.model_type.get() == "DecisionTree":
                # Decision tree modes require a single target (scalar y)
                if y_train_for_r.ndim > 1 and y_train_for_r.shape[1] != 1:
                    raise ValueError("Decision Tree requires Training Y to have exactly 1 column.")

                y_tree = y_train_for_r.reshape(-1)
                w = None

                if self.want_manual_split.get():
                    split_val = float(self.split_x.get())
                    split_info = backend.manual_split_tree(X_train, y_tree, split_val)
                    preds = backend.manual_split_predict(
                        X_test,
                        split_info["split_x"],
                        split_info["left_mean"],
                        split_info["right_mean"],
                    )
                else:
                    depth_val = int(float(self.tree_depth.get()))
                    if depth_val < 0:
                        raise ValueError("Decision Tree depth must be >= 0.")
                    if depth_val == 0:
                        # Root node only: constant prediction = mean(y_train)
                        preds = np.full((np.asarray(X_test).shape[0],), float(np.mean(y_tree)))
                    else:
                        tree = backend.regressionTree(X_train, y_tree, depth=depth_val)
                        preds = tree.predict(np.asarray(X_test) if np.asarray(X_test).ndim == 2 else np.asarray(X_test).reshape(-1, 1))
            elif self.model_type.get() == "Linear":
                w = backend.LinRegression(X_train, y_train)
                preds = backend.predict(X_test, w, order=1, format=pred_format)
            else:
                w = backend.PolyRegression(X_train, y_train, order=order, reg_factor=reg)
                preds = backend.predict(X_test, w, order=order, format=pred_format)

            # 3. Display Results
            if self.want_w.get():
                self.result_text.insert(tk.END, "--- W Array ---\n")
                self.result_text.insert(tk.END, f"{np.round(w, 4)}\n\n")

            if self.want_p.get() and self.model_type.get() in {"Linear", "Poly"}:
                p_order = 1 if self.model_type.get() == "Linear" else int(self.poly_order.get())
                poly = PolynomialFeatures(p_order)
                P_train = poly.fit_transform(X_train)
                P_test = poly.transform(X_test) if np.asarray(X_test).shape[0] > 0 else None

                self.result_text.insert(tk.END, f"--- P Matrix (PolynomialFeatures order={p_order}) ---\n")
                self.result_text.insert(tk.END, "P_train:\n")
                self.result_text.insert(tk.END, f"{np.round(P_train, 4)}\n")
                if P_test is not None:
                    self.result_text.insert(tk.END, "\nP_test:\n")
                    self.result_text.insert(tk.END, f"{np.round(P_test, 4)}\n")
                self.result_text.insert(tk.END, "\n")

            if self.want_pred.get():
                self.result_text.insert(tk.END, "--- Predicted Values (Test Data) ---\n")
                self.result_text.insert(tk.END, f"{np.round(preds, 4)}\n\n")

            if self.model_type.get() == "DecisionTree" and self.want_manual_split.get():
                self.result_text.insert(tk.END, "--- Manual Split Details (Training Data) ---\n")
                self.result_text.insert(tk.END, f"split_x = {split_info['split_x']}\n")
                self.result_text.insert(tk.END, f"n_left = {split_info['n_left']}, n_right = {split_info['n_right']}\n")
                self.result_text.insert(tk.END, f"left_mean = {split_info['left_mean']:.4f}, right_mean = {split_info['right_mean']:.4f}\n")
                self.result_text.insert(tk.END, f"weighted_train_mse = {split_info['mse_total']:.4f}\n\n")

            # Test MSE only when enabled and test set exists
            if self.want_mse.get() and np.asarray(X_test).shape[0] > 0:
                if self.model_type.get() == "DecisionTree":
                    y_test_tree = np.asarray(y_test)
                    if y_test_tree.ndim == 2 and y_test_tree.shape[1] == 1:
                        y_test_tree = y_test_tree.reshape(-1)
                    preds_tree = np.asarray(preds).reshape(-1)
                    k = min(y_test_tree.size, preds_tree.size)
                    mse = float(np.mean((y_test_tree[:k] - preds_tree[:k]) ** 2)) if k > 0 else float("nan")
                else:
                    mse = self.calculate_mse(y_test, preds, pred_format)
                self.result_text.insert(tk.END, "--- Test Data Mean Squared Error (MSE) ---\n")
                self.result_text.insert(tk.END, f"{mse:.4f}\n\n")

            if self.want_r.get():
                r, p = self.calculate_r_x_to_y(X_train, y_train_for_r)
                self.result_text.insert(tk.END, "--- Pearson Correlation (R): Training X vs Training Y ---\n")
                if np.isnan(r):
                    self.result_text.insert(tk.END, "R is undefined (constant input, insufficient data, or invalid data).\n\n")
                else:
                    self.result_text.insert(tk.END, f"R = {r:.4f} (p = {p:.4g})\n\n")
                
            # In Decision Tree mode, always show Training MSE (default)
            if self.model_type.get() == "DecisionTree" or self.want_train_mse.get():
                if self.model_type.get() == "DecisionTree":
                    if self.want_manual_split.get():
                        train_mse = float(split_info["mse_total"])
                    else:
                        depth_val = int(float(self.tree_depth.get()))
                        train_mse = backend.regressionTree_mse(X_train, y_tree, depth=depth_val)
                else:
                    if self.model_type.get() == "Linear":
                        train_preds = backend.predict(X_train, w, order=1, format=pred_format)
                    else:
                        train_preds = backend.predict(X_train, w, order=order, format=pred_format)
                    train_mse = self.calculate_mse(y_train, train_preds, pred_format)
                self.result_text.insert(tk.END, "--- Training Data Mean Squared Error (MSE) ---\n")
                self.result_text.insert(tk.END, f"{train_mse:.4f}\n\n")

        except Exception as e:
            messagebox.showerror("Calculation Error", f"An error occurred:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = RegressionApp(root)
    root.mainloop()