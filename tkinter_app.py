import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import backend

class RegressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Regression Model App")
        self.root.geometry("900x750") 
        
        # --- Variables ---
        self.model_type = tk.StringVar(value="Linear")
        self.want_w = tk.BooleanVar(value=True)
        self.want_pred = tk.BooleanVar(value=True)
        self.want_mse = tk.BooleanVar(value=False)
        self.want_train_mse = tk.BooleanVar(value=False)
        
        # OHE Variables
        self.want_ohe = tk.BooleanVar(value=False)
        self.ohe_format = tk.StringVar(value="raw")
        
        # Lists to hold the entry widgets for our matrices
        self.X_train_entries = []
        self.y_train_entries = []
        self.X_test_entries = []
        self.y_test_entries = []

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

        ttk.Label(settings_frame, text="Order:").grid(row=0, column=2, padx=5)
        self.poly_order = ttk.Entry(settings_frame, width=5)
        self.poly_order.insert(0, "2")
        self.poly_order.grid(row=0, column=3, padx=5)

        ttk.Label(settings_frame, text="Lambda:").grid(row=0, column=4, padx=5)
        self.reg_factor = ttk.Entry(settings_frame, width=5)
        self.reg_factor.insert(0, "0.0")
        self.reg_factor.grid(row=0, column=5, padx=5)

        ttk.Separator(settings_frame, orient='vertical').grid(row=0, column=6, sticky='ns', padx=10)

        ttk.Checkbutton(settings_frame, text="W Array", variable=self.want_w).grid(row=0, column=7)
        ttk.Checkbutton(settings_frame, text="Predictions", variable=self.want_pred).grid(row=0, column=8)
        
        self.mse_cb = ttk.Checkbutton(settings_frame, text="Test MSE", variable=self.want_mse, command=self.toggle_train_mse)
        self.mse_cb.grid(row=0, column=9)
        
        self.train_mse_cb = ttk.Checkbutton(settings_frame, text="Train MSE", variable=self.want_train_mse)
        self.train_mse_cb.grid(row=0, column=10)
        self.train_mse_cb.state(['disabled'])

        # --- 4. Matrix Input Area ---
        self.matrix_frame = ttk.Frame(self.root)
        self.matrix_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # --- 5. Run Button & Output Area ---
        ttk.Button(self.root, text="RUN MODEL", command=self.run_model).pack(pady=5)
        
        self.result_text = tk.Text(self.root, height=10)
        self.result_text.pack(fill="both", expand=True, padx=10, pady=10)

    # UI Logic for Checkboxes/Dropdowns
    def toggle_train_mse(self):
        if self.want_mse.get():
            self.train_mse_cb.state(['!disabled'])
        else:
            self.want_train_mse.set(False) 
            self.train_mse_cb.state(['disabled']) 

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

    def generate_grids(self):
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

        self.X_train_entries = self.create_grid(self.matrix_frame, "Training X", rows=m, cols=n, grid_row=0, grid_col=0)
        self.y_train_entries = self.create_grid(self.matrix_frame, "Training Y", rows=m, cols=t, grid_row=0, grid_col=1, is_y=True)
        self.X_test_entries = self.create_grid(self.matrix_frame, "Test X", rows=test_m, cols=n, grid_row=1, grid_col=0)
        self.y_test_entries = self.create_grid(self.matrix_frame, "Test Y", rows=test_m, cols=t, grid_row=1, grid_col=1, is_y=True)

    def create_grid(self, parent, title, rows, cols, grid_row, grid_col, is_y=False):
        frame = ttk.LabelFrame(parent, text=title)
        frame.grid(row=grid_row, column=grid_col, padx=10, pady=10, sticky="nw")
        
        entries = []
        for i in range(rows):
            row_entries = []
            for j in range(cols):
                e = ttk.Entry(frame, width=6)
                e.grid(row=i, column=j, padx=2, pady=2)
                
                # Apply OHE Specific UI behaviors to Y targets
                if self.want_ohe.get() and is_y:
                    if self.ohe_format.get() == "raw":
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

    def run_model(self):
        self.result_text.delete(1.0, tk.END) 
        
        if not self.X_train_entries:
            messagebox.showwarning("Warning", "Please generate and fill the grids first.")
            return

        try:
            # 1. Gather Data
            X_train = self.get_matrix_data(self.X_train_entries)
            y_train = self.get_matrix_data(self.y_train_entries, is_y=True)
            X_test = self.get_matrix_data(self.X_test_entries)
            y_test = self.get_matrix_data(self.y_test_entries, is_y=True)

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
            if self.model_type.get() == "Linear":
                w = backend.LinRegression(X_train, y_train)
                preds = backend.predict(X_test, w, order=1, format=pred_format)
            else:
                w = backend.PolyRegression(X_train, y_train, order=order, reg_factor=reg)
                preds = backend.predict(X_test, w, order=order, format=pred_format)

            # 3. Display Results
            if self.want_w.get():
                self.result_text.insert(tk.END, "--- W Array ---\n")
                self.result_text.insert(tk.END, f"{np.round(w, 4)}\n\n")

            if self.want_pred.get():
                self.result_text.insert(tk.END, "--- Predicted Values (Test Data) ---\n")
                self.result_text.insert(tk.END, f"{np.round(preds, 4)}\n\n")

            if self.want_mse.get():
                mse = self.calculate_mse(y_test, preds, pred_format)
                self.result_text.insert(tk.END, "--- Test Data Mean Squared Error (MSE) ---\n")
                self.result_text.insert(tk.END, f"{mse:.4f}\n\n")
                
            if self.want_train_mse.get():
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