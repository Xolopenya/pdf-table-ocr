"""Десктопный GUI (Tkinter/ttk) поверх ядра pipeline.process_pdf.

Тонкая оболочка: своей логики обработки НЕТ — выбирает ОДИН PDF, по кнопке зовёт
process_pdf в отдельном потоке (окно не виснет), показывает прогресс со счётчиками
вызовов/кэша и лог, по готовности — «Открыть Excel/JSON/папку». Ключи из .env.
Массовых/фоновых прогонов не запускает — только по кнопке и по одному файлу.

Запуск:  python run_gui.py
"""
from __future__ import annotations

import os
import queue
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

try:  # необязательный drag&drop; без него — кнопка «Выберите файл»
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
    _DND = True
except Exception:  # noqa: BLE001
    _DND = False

from .config import bootstrap_poppler, load_config
from .logging_setup import setup_logging

ENGINES = ["hybrid", "yandex", "qwen"]

# --- палитра / шрифты -------------------------------------------------------
BG = "#f4f6f8"
CARD = "#ffffff"
ACCENT = "#2d6cdf"
ACCENT_ACT = "#1e4fa0"
TEXT = "#1f2430"
MUTED = "#6b7280"
DROP_BORDER = "#b6c2d4"
UI_FONT = "Segoe UI" if sys.platform.startswith("win") else "Helvetica"


def _open_path(path) -> None:
    p = str(path)
    try:
        if sys.platform.startswith("win"):
            os.startfile(p)  # noqa: S606
        elif sys.platform == "darwin":
            os.system(f'open "{p}"')
        else:
            os.system(f'xdg-open "{p}"')
    except Exception as e:  # noqa: BLE001
        messagebox.showerror("Ошибка", f"Не удалось открыть: {e}")


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.q: queue.Queue = queue.Queue()
        self.result: dict | None = None
        self.busy = False
        self.cfg = load_config()
        setup_logging(self.cfg.get("logging.level", "INFO"))
        bootstrap_poppler(self.cfg.get("poppler_path") or None)

        root.title("Распознавание буровых журналов")
        root.geometry("820x640")
        root.minsize(720, 560)
        root.configure(bg=BG)
        self._init_style()

        outer = tk.Frame(root, bg=BG)
        outer.pack(fill="both", expand=True, padx=20, pady=18)

        tk.Label(outer, text="Буровые журналы → Excel", bg=BG, fg=TEXT,
                 font=(UI_FONT, 17, "bold")).pack(anchor="w")
        tk.Label(outer, text="Вставьте один PDF-файл и нажмите «Обработать». "
                 "Результат — Excel и JSON по всем скважинам файла.",
                 bg=BG, fg=MUTED, font=(UI_FONT, 10)).pack(anchor="w", pady=(2, 14))

        # --- зона drop / выбора файла ---
        self.drop = tk.Canvas(outer, height=104, bg=CARD, highlightthickness=0)
        self.drop.pack(fill="x")
        self.drop.bind("<Configure>", lambda e: self._redraw_drop())
        self.drop.bind("<Button-1>", lambda e: self._pick_file())
        self.input_var = tk.StringVar(value="")
        if _DND:
            self.drop.drop_target_register(DND_FILES)  # type: ignore
            self.drop.dnd_bind("<<Drop>>", self._on_drop)  # type: ignore

        # --- строка настроек ---
        row = tk.Frame(outer, bg=BG)
        row.pack(fill="x", pady=(14, 4))
        tk.Label(row, text="Движок:", bg=BG, fg=TEXT, font=(UI_FONT, 10)).pack(side="left")
        self.engine_var = tk.StringVar(value=self.cfg.get("engine", "hybrid"))
        ttk.Combobox(row, textvariable=self.engine_var, values=ENGINES, state="readonly",
                     width=12).pack(side="left", padx=(6, 18))
        self.debug_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="сохранять debug (overlay)",
                        variable=self.debug_var).pack(side="left")

        s = self.cfg.secrets
        keys = ("Yandex " + ("✓" if s.has_yandex() else "✗") + "   OpenRouter "
                + ("✓" if s.has_openrouter() else "✗"))
        tk.Label(row, text=keys, bg=BG, fg=MUTED, font=(UI_FONT, 9)).pack(side="right")

        # --- кнопки действия ---
        act = tk.Frame(outer, bg=BG)
        act.pack(fill="x", pady=(10, 8))
        self.run_btn = ttk.Button(act, text="Обработать", style="Accent.TButton",
                                  command=self._start)
        self.run_btn.pack(side="left")
        self.open_xlsx = ttk.Button(act, text="Открыть Excel", state="disabled",
                                    command=lambda: _open_path(self.result["xlsx"]))
        self.open_xlsx.pack(side="left", padx=(10, 4))
        self.open_json = ttk.Button(act, text="Открыть JSON", state="disabled",
                                    command=lambda: _open_path(self.result["json"]))
        self.open_json.pack(side="left", padx=4)
        self.open_dir = ttk.Button(act, text="Открыть папку", state="disabled",
                                   command=lambda: _open_path(Path(self.result["xlsx"]).parent))
        self.open_dir.pack(side="left", padx=4)

        # --- прогресс + статус ---
        self.prog = ttk.Progressbar(outer, mode="determinate")
        self.prog.pack(fill="x", pady=(6, 2))
        self.status = tk.Label(outer, text="Готово к работе.", bg=BG, fg=MUTED,
                               anchor="w", font=(UI_FONT, 9))
        self.status.pack(fill="x")

        # --- лог ---
        self.log = scrolledtext.ScrolledText(outer, height=14, wrap="word", state="disabled",
                                             font=("Consolas", 9), bg="#0f172a", fg="#d7e0ee",
                                             insertbackground="#d7e0ee", relief="flat")
        self.log.pack(fill="both", expand=True, pady=(10, 0))

        self._redraw_drop()
        self._logln("Готово. Перетащите PDF в рамку или кликните по ней / «Обработать».")
        self.root.after(100, self._poll)

    # ------------------------------------------------------------------ стиль
    def _init_style(self):
        st = ttk.Style()
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", background=BG, foreground=TEXT, font=(UI_FONT, 10))
        st.configure("TCombobox", padding=4)
        st.configure("TCheckbutton", background=BG, foreground=TEXT)
        st.configure("TButton", padding=(12, 7), font=(UI_FONT, 10))
        st.configure("Accent.TButton", padding=(20, 9), font=(UI_FONT, 11, "bold"),
                     background=ACCENT, foreground="white", borderwidth=0)
        st.map("Accent.TButton",
               background=[("active", ACCENT_ACT), ("disabled", "#a9b4c4")],
               foreground=[("disabled", "#eef1f6")])
        st.configure("TProgressbar", troughcolor="#e2e8f0", background=ACCENT,
                     borderwidth=0, thickness=10)

    def _redraw_drop(self):
        c = self.drop
        c.delete("all")
        w = c.winfo_width() or 700
        h = int(c["height"])
        c.create_rectangle(3, 3, w - 3, h - 3, dash=(6, 4), outline=DROP_BORDER, width=2)
        val = self.input_var.get()
        if val:
            c.create_text(w / 2, h / 2 - 8, text=Path(val).name, fill=TEXT,
                          font=(UI_FONT, 12, "bold"))
            c.create_text(w / 2, h / 2 + 14, text="кликните, чтобы выбрать другой",
                          fill=MUTED, font=(UI_FONT, 9))
        else:
            top = ("Перетащите PDF сюда" if _DND else "Кликните, чтобы выбрать PDF")
            c.create_text(w / 2, h / 2 - 8, text=top, fill=TEXT, font=(UI_FONT, 12))
            c.create_text(w / 2, h / 2 + 14, text="или нажмите, чтобы выбрать файл",
                          fill=MUTED, font=(UI_FONT, 9))

    # --------------------------------------------------------------- ввод
    def _pick_file(self):
        if self.busy:
            return
        p = filedialog.askopenfilename(title="Выберите PDF",
                                       filetypes=[("PDF", "*.pdf"), ("Все файлы", "*.*")])
        if p:
            self.input_var.set(p)
            self._redraw_drop()

    def _on_drop(self, event):
        raw = event.data.strip().strip("{}")
        if raw.lower().endswith(".pdf"):
            self.input_var.set(raw)
            self._redraw_drop()
        else:
            messagebox.showwarning("Не PDF", "Перетащите один PDF-файл.")

    # -------------------------------------------------------- лог/прогресс
    def _logln(self, msg: str):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._logln(payload)
                elif kind == "progress":
                    done, total, msg = payload
                    self.prog["maximum"] = max(total, 1)
                    self.prog["value"] = done
                    self.status.configure(text=msg)
                    self._logln(msg)
                elif kind == "done":
                    self.result = payload
                    self.busy = False
                    self.run_btn.configure(text="Обработать", state="normal")
                    for b in (self.open_xlsx, self.open_json, self.open_dir):
                        b.configure(state="normal")
                    self.status.configure(text="Готово.")
                elif kind == "error":
                    self.busy = False
                    self.run_btn.configure(text="Обработать", state="normal")
                    self.status.configure(text="Ошибка (см. лог).")
                    self._logln("ОШИБКА:\n" + payload)
                    messagebox.showerror("Ошибка обработки", payload.splitlines()[0])
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    # ------------------------------------------------------------- запуск
    def _start(self):
        if self.busy:
            return
        path = self.input_var.get().strip()
        if not path or not Path(path).exists():
            messagebox.showwarning("Нет файла", "Выберите существующий PDF-файл.")
            return
        if not Path(path).is_file() or Path(path).suffix.lower() != ".pdf":
            messagebox.showwarning("Только PDF", "Выберите ОДИН PDF-файл (не папку).")
            return
        self.busy = True
        self.result = None
        for b in (self.open_xlsx, self.open_json, self.open_dir):
            b.configure(state="disabled")
        self.run_btn.configure(text="Обработка…", state="disabled")
        self.prog["value"] = 0
        engine = self.engine_var.get()
        self._logln(f"─── Обработка: {Path(path).name}  (движок={engine}) ───")
        threading.Thread(target=self._work, args=(path, engine, self.debug_var.get()),
                         daemon=True).start()

    def _work(self, path: str, engine: str, debug: bool):
        from .pipeline.process import process_pdf

        def prog(done, total, msg):
            self.q.put(("progress", (done, total, msg)))
        try:
            res = process_pdf(Path(path), self.cfg, engine=engine,
                              save_debug=debug, progress=prog)
            self.q.put(("done", res))
        except Exception as e:  # noqa: BLE001
            import traceback
            self.q.put(("error", f"{e}\n{traceback.format_exc()}"))


def main():
    root = (TkinterDnD.Tk() if _DND else tk.Tk())
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
