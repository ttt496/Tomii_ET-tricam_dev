# -*- coding: utf-8 -*-
"""Simple Tkinter-based mock UI for selecting social attributes and visualizing
Big Five personality scores on a radar chart.

The left-hand side offers drop-downs for standard attributes and the ability
to append custom attributes. The right-hand side renders a radar chart using
mock scores derived from the chosen values.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Tuple

import numpy as np

# Matplotlib imports need to come after setting the backend to avoid issues on macOS.
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Mock data and simple heuristics for translating selections into scores.
# ---------------------------------------------------------------------------

ATTRIBUTES: Dict[str, List[str]] = {
    "国籍": ["日本", "米国", "英国", "その他"],
    "年齢": ["10代", "20代", "30代", "40代", "50代以上"],
    "性別": ["男性", "女性", "ノンバイナリー", "回答しない"],
    "職種": ["学生", "エンジニア", "デザイナー", "マネージャー", "研究者", "その他"],
}

BIG5_TRAITS: List[str] = [
    "Openness (開放性)",
    "Conscientiousness (誠実性)",
    "Extraversion (外向性)",
    "Agreeableness (協調性)",
    "Neuroticism (神経症傾向)",
]

BASE_PROFILE = np.array([0.52, 0.55, 0.5, 0.53, 0.48], dtype=float)

ATTRIBUTE_EFFECTS: Dict[str, Dict[str, np.ndarray]] = {
    "国籍": {
        "日本": np.array([-0.04, 0.05, -0.06, 0.06, 0.08]),
        "米国": np.array([0.07, -0.02, 0.08, 0.04, -0.05]),
        "英国": np.array([0.03, 0.04, 0.02, 0.03, -0.03]),
        "その他": np.zeros(5),
    },
    "年齢": {
        "10代": np.array([0.05, -0.05, 0.07, -0.02, 0.02]),
        "20代": np.array([0.06, -0.02, 0.05, 0.01, 0.0]),
        "30代": np.array([0.0, 0.04, 0.0, 0.02, -0.03]),
        "40代": np.array([-0.03, 0.05, -0.04, 0.04, -0.01]),
        "50代以上": np.array([-0.04, 0.06, -0.05, 0.05, -0.02]),
    },
    "性別": {
        "男性": np.array([0.01, 0.0, 0.04, -0.02, -0.03]),
        "女性": np.array([0.02, 0.03, -0.02, 0.05, 0.02]),
        "ノンバイナリー": np.array([0.04, 0.02, 0.02, 0.04, 0.01]),
        "回答しない": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    },
    "職種": {
        "学生": np.array([0.05, -0.04, 0.06, 0.02, 0.01]),
        "エンジニア": np.array([0.03, 0.05, -0.02, 0.01, -0.03]),
        "デザイナー": np.array([0.06, -0.01, 0.05, 0.03, 0.0]),
        "マネージャー": np.array([-0.01, 0.06, -0.01, 0.05, -0.02]),
        "研究者": np.array([0.02, 0.07, -0.03, 0.02, -0.04]),
        "その他": np.zeros(5),
    },
}

# ---------------------------------------------------------------------------
# Tkinter UI creation helpers.
# ---------------------------------------------------------------------------


def clamp_scores(scores: np.ndarray) -> np.ndarray:
    """Keep scores within a visually pleasing range for the radar chart."""
    return np.clip(scores, 0.1, 0.9)


def compute_profile(selections: Dict[str, str]) -> np.ndarray:
    profile = BASE_PROFILE.copy()
    for attribute, value in selections.items():
        effect_map = ATTRIBUTE_EFFECTS.get(attribute, {})
        profile += effect_map.get(value, np.zeros_like(profile))
    return clamp_scores(profile)


def add_custom_attribute(
    name_var: tk.StringVar,
    value_var: tk.StringVar,
    listbox: tk.Listbox,
    registry: List[Tuple[str, str]],
) -> None:
    name = name_var.get().strip()
    value = value_var.get().strip()

    if not name or not value:
        return

    registry.append((name, value))
    listbox.insert(tk.END, f"{name}: {value}")
    name_var.set("")
    value_var.set("")


# ---------------------------------------------------------------------------
# Main application assembly.
# ---------------------------------------------------------------------------


def launch_app() -> None:
    root = tk.Tk()
    root.title("社会的属性 × Big Five モック")
    root.geometry("960x600")

    root.columnconfigure(0, weight=2)
    root.columnconfigure(1, weight=3)
    root.rowconfigure(0, weight=1)

    control_frame = ttk.Frame(root, padding=20)
    control_frame.grid(row=0, column=0, sticky="nsew")
    control_frame.columnconfigure(1, weight=1)

    chart_frame = ttk.Frame(root, padding=20)
    chart_frame.grid(row=0, column=1, sticky="nsew")
    chart_frame.columnconfigure(0, weight=1)
    chart_frame.rowconfigure(0, weight=1)

    ttk.Label(control_frame, text="社会的属性", font=("Helvetica", 14, "bold")).grid(
        row=0, column=0, columnspan=2, sticky="w"
    )
    ttk.Label(
        control_frame,
        text="基本属性を選択し、必要に応じて追加属性を入力してください。",
        wraplength=280,
    ).grid(row=1, column=0, columnspan=2, pady=(4, 16), sticky="w")

    selections: Dict[str, str] = {k: v[0] for k, v in ATTRIBUTES.items()}
    attribute_vars: Dict[str, tk.StringVar] = {}

    def refresh_profile(_: tk.Event | None = None) -> None:
        updated = compute_profile(selections)
        plot_radar(updated)
        for trait, var in score_labels.items():
            index = BIG5_TRAITS.index(trait)
            var.set(f"{updated[index]:.2f}")

    current_row = 2
    for attribute, options in ATTRIBUTES.items():
        ttk.Label(control_frame, text=attribute).grid(
            row=current_row, column=0, sticky="w", pady=2
        )
        var = tk.StringVar(value=options[0])
        combo = ttk.Combobox(
            control_frame,
            textvariable=var,
            values=options,
            state="readonly",
        )
        combo.grid(row=current_row, column=1, sticky="ew", pady=2)

        def on_change(event: tk.Event, attr: str = attribute, variable: tk.StringVar = var) -> None:
            selections[attr] = variable.get()
            refresh_profile(event)

        combo.bind("<<ComboboxSelected>>", on_change)
        attribute_vars[attribute] = var
        current_row += 1

    ttk.Separator(control_frame).grid(
        row=current_row, column=0, columnspan=2, sticky="ew", pady=(12, 12)
    )
    current_row += 1

    ttk.Label(control_frame, text="その他の属性追加").grid(
        row=current_row, column=0, columnspan=2, sticky="w"
    )
    current_row += 1

    custom_name = tk.StringVar()
    custom_value = tk.StringVar()
    custom_entries: List[Tuple[str, str]] = []

    ttk.Entry(control_frame, textvariable=custom_name, width=18).grid(
        row=current_row, column=0, sticky="ew", pady=2
    )
    ttk.Entry(control_frame, textvariable=custom_value, width=18).grid(
        row=current_row, column=1, sticky="ew", pady=2
    )
    current_row += 1

    ttk.Button(
        control_frame,
        text="追加",
        command=lambda: add_custom_attribute(
            custom_name, custom_value, custom_listbox, custom_entries
        ),
    ).grid(row=current_row, column=0, columnspan=2, sticky="ew", pady=(4, 6))
    current_row += 1

    custom_listbox = tk.Listbox(control_frame, height=5)
    custom_listbox.grid(row=current_row, column=0, columnspan=2, sticky="nsew")
    control_frame.rowconfigure(current_row, weight=1)

    # ------------------------------------------------------------------
    # Radar chart setup.
    # ------------------------------------------------------------------
    figure = Figure(figsize=(5.5, 5), dpi=100)
    radar_ax = figure.add_subplot(111, polar=True)

    canvas = FigureCanvasTkAgg(figure, master=chart_frame)
    canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    chart_frame.rowconfigure(0, weight=1)

    scores_frame = ttk.Frame(chart_frame)
    scores_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
    scores_frame.columnconfigure(1, weight=1)

    ttk.Label(scores_frame, text="Big Five スコア", font=("Helvetica", 12, "bold")).grid(
        row=0, column=0, columnspan=2, sticky="w"
    )

    score_labels: Dict[str, tk.StringVar] = {}
    for index, trait in enumerate(BIG5_TRAITS, start=1):
        ttk.Label(scores_frame, text=trait).grid(row=index, column=0, sticky="w")
        var = tk.StringVar(value="0.00")
        ttk.Label(scores_frame, textvariable=var).grid(row=index, column=1, sticky="e")
        score_labels[trait] = var

    def plot_radar(scores: np.ndarray) -> None:
        labeled_scores = np.concatenate((scores, [scores[0]]))
        angles = np.linspace(0, 2 * np.pi, len(BIG5_TRAITS), endpoint=False)
        angled_scores = np.concatenate((angles, [angles[0]]))

        radar_ax.clear()
        radar_ax.set_theta_offset(np.pi / 2)
        radar_ax.set_theta_direction(-1)
        radar_ax.set_thetagrids(np.degrees(angles), BIG5_TRAITS)
        radar_ax.set_ylim(0, 1)

        radar_ax.plot(angled_scores, labeled_scores, color="#4C72B0", linewidth=2)
        radar_ax.fill(angled_scores, labeled_scores, color="#4C72B0", alpha=0.25)
        radar_ax.set_title("Big Five プロファイル", pad=20, fontsize=12)

        canvas.draw_idle()

    # Initialize with default selections.
    refresh_profile()

    root.mainloop()


if __name__ == "__main__":
    launch_app()
