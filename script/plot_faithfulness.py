#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import argparse
import datetime

import matplotlib.colors
import matplotlib.figure
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np

import config
import dataset


# method = ["LIME", "SHAP", "Tilia", "Tilia\\textsuperscript{+}", "Tilia\\textsubscript{10}\\textsuperscript{+}", "Tilia\\textsuperscript{+}\\textsubscript{15}"]
method = ["LIME$_5$", "SHAP$_5$", "Tilia$_{5}$", "Tilia$^{+}_{5}$", "Tilia$^{+}_{10}$", "Tilia$^{+}_{15}$"]


# palette = ["#a4e2c6", "#012696", "#e9eeb9", "#ed9874", "#ed9874", "#ed9874"]
# palette = ["#9797f8", "#f1faee", "#eebabb", "#d86967", "#d86967", "#d86967"]
palette = {
        # "fid": {"LR": "#58539f", "DT": "#d86967"},
        "bar": {"LR": "#00007f", "DT": "#db8bf4"},
        "fid": {"LR": "#00007f", "DT": "#db8bf4"},
        }

alpha = {
        "bar": 0.5,
        "fid": 0.8,
        }

text_offset = 1
line_width = 6

fidelity = {
        "sample": {
            "Sample = 5": [
                {"LR": 91.32, "DT": 48.97},
                {"LR": 82.56, "DT": 47.95},
                {"LR": 99.89, "DT": 99.79},
                {"LR": 99.89, "DT": 99.80},
                {"LR": 99.89, "DT": 99.81},
                {"LR": 99.91, "DT": 99.79},
                ],
            "Sample = 10": [
                {"LR": 92.41, "DT": 52.92},
                {"LR": 88.00, "DT": 46.74},
                {"LR": 99.91, "DT": 99.89},
                {"LR": 99.91, "DT": 99.89},
                {"LR": 99.91, "DT": 99.90},
                {"LR": 99.91, "DT": 99.89},
                ],
            "Sample = 200": [
                {"LR": 92.46, "DT": 60.62},
                {"LR": 82.68, "DT": 54.76},
                {"LR": 99.92, "DT": 99.87},
                {"LR": 99.92, "DT": 99.86},
                {"LR": 99.92, "DT": 99.88},
                {"LR": 99.92, "DT": 99.87},
                ],
            "Sample = 400": [
                {"LR": 92.25, "DT": 59.29},
                {"LR": 82.73, "DT": 54.87},
                {"LR": 99.92, "DT": 99.86},
                {"LR": 99.92, "DT": 99.86},
                {"LR": 99.92, "DT": 99.87},
                {"LR": 99.92, "DT": 99.87},
                ],
            },
        "method": {
            "LIME$_5$": [
                {"LR": 91.32, "DT": 48.97},
                {"LR": 92.41, "DT": 52.92},
                {"LR": 92.46, "DT": 60.62},
                {"LR": 92.25, "DT": 59.29},
                ],
            "SHAP$_5$": [
                {"LR": 82.56, "DT": 47.95},
                {"LR": 88.00, "DT": 46.74},
                {"LR": 82.68, "DT": 54.76},
                {"LR": 82.73, "DT": 54.87},
                ],
            "Tilia$_{5}$": [
                {"LR": 99.89, "DT": 99.79},
                {"LR": 99.91, "DT": 99.89},
                {"LR": 99.92, "DT": 99.87},
                {"LR": 99.92, "DT": 99.86},
                ],
            "Tilia$^{+}_{5}$": [
                {"LR": 99.89, "DT": 99.80},
                {"LR": 99.91, "DT": 99.89},
                {"LR": 99.92, "DT": 99.86},
                {"LR": 99.92, "DT": 99.86},
                ],
            "Tilia$^{+}_{10}$": [
                {"LR": 99.89, "DT": 99.81},
                {"LR": 99.91, "DT": 99.90},
                {"LR": 99.92, "DT": 99.88},
                {"LR": 99.92, "DT": 99.87},
                ],
            "Tilia$^{+}_{15}$": [
                {"LR": 99.91, "DT": 99.79},
                {"LR": 99.91, "DT": 99.89},
                {"LR": 99.92, "DT": 99.87},
                {"LR": 99.92, "DT": 99.87},
                ]
            }
        }

recall = {
        "sample": {
            "Sample = 5": [
                {"LR": {"v": 53.33, "b": 0, "c": 0}, "DT": {"v": 95.00, "b": 1, "c": 0}},
                {"LR": {"v": 46.67, "b": 0, "c": 0}, "DT": {"v": 85.50, "b": 0, "c": 0}},
                {"LR": {"v": 55.00, "b": 0, "c": 0}, "DT": {"v": 94.00, "b": 0, "c": 0}},
                {"LR": {"v": 45.83, "b": 0, "c": 0}, "DT": {"v": 95.00, "b": 1, "c": 0}},
                {"LR": {"v": 58.33, "b": 1, "c": 0}, "DT": {"v": 90.00, "b": 0, "c": 0}},
                {"LR": {"v": 58.33, "b": 1, "c": 0}, "DT": {"v": 90.00, "b": 0, "c": 0}},
                ],
            "Sample = 10": [
                {"LR": {"v": 67.22, "b": 0, "c": 0}, "DT": {"v": 96.67, "b": 1, "c": 1}},
                {"LR": {"v": 51.11, "b": 0, "c": 0}, "DT": {"v": 75.33, "b": 0, "c": 0}},
                {"LR": {"v": 70.56, "b": 0, "c": 0}, "DT": {"v": 94.67, "b": 0, "c": 0}},
                {"LR": {"v": 63.89, "b": 0, "c": 0}, "DT": {"v": 96.67, "b": 1, "c": 1}},
                {"LR": {"v": 72.22, "b": 1, "c": 0}, "DT": {"v": 93.33, "b": 0, "c": 0}},
                {"LR": {"v": 72.22, "b": 1, "c": 0}, "DT": {"v": 93.33, "b": 0, "c": 0}},
                ],
            "Sample = 200": [
                {"LR": {"v": 92.67, "b": 0, "c": 0}, "DT": {"v": 83.46, "b": 0, "c": 0}},
                {"LR": {"v": 70.18, "b": 0, "c": 0}, "DT": {"v": 83.99, "b": 0, "c": 0}},
                {"LR": {"v": 93.36, "b": 0, "c": 0}, "DT": {"v": 86.21, "b": 1, "c": 0}},
                {"LR": {"v": 92.96, "b": 0, "c": 0}, "DT": {"v": 85.52, "b": 0, "c": 0}},
                {"LR": {"v": 93.13, "b": 0, "c": 0}, "DT": {"v": 85.91, "b": 0, "c": 0}},
                {"LR": {"v": 93.99, "b": 1, "c": 0}, "DT": {"v": 85.49, "b": 0, "c": 0}},
                ],
            "Sample = 400": [
                {"LR": {"v": 94.06, "b": 0, "c": 0}, "DT": {"v": 86.79, "b": 0, "c": 0}},
                {"LR": {"v": 74.46, "b": 0, "c": 0}, "DT": {"v": 84.70, "b": 0, "c": 0}},
                {"LR": {"v": 94.38, "b": 0, "c": 0}, "DT": {"v": 88.18, "b": 0, "c": 0}},
                {"LR": {"v": 94.76, "b": 0, "c": 0}, "DT": {"v": 88.39, "b": 0, "c": 0}},
                {"LR": {"v": 94.62, "b": 0, "c": 0}, "DT": {"v": 89.51, "b": 1, "c": 0}},
                {"LR": {"v": 95.04, "b": 1, "c": 1}, "DT": {"v": 89.20, "b": 0, "c": 0}},
                ],
            },
        "method": {
            "LIME$_5$": [
                {"LR": {"v": 53.33, "b": 0, "c": 0}, "DT": {"v": 95.00, "b": 1, "c": 0}},
                {"LR": {"v": 67.22, "b": 0, "c": 0}, "DT": {"v": 96.67, "b": 1, "c": 1}},
                {"LR": {"v": 92.67, "b": 0, "c": 0}, "DT": {"v": 83.46, "b": 0, "c": 0}},
                {"LR": {"v": 94.06, "b": 0, "c": 0}, "DT": {"v": 86.79, "b": 0, "c": 0}},
                ],
            "SHAP$_5$": [
                {"LR": {"v": 46.67, "b": 0, "c": 0}, "DT": {"v": 85.50, "b": 0, "c": 0}},
                {"LR": {"v": 51.11, "b": 0, "c": 0}, "DT": {"v": 75.33, "b": 0, "c": 0}},
                {"LR": {"v": 70.18, "b": 0, "c": 0}, "DT": {"v": 83.99, "b": 0, "c": 0}},
                {"LR": {"v": 74.46, "b": 0, "c": 0}, "DT": {"v": 84.70, "b": 0, "c": 0}},
                ],
            "Tilia$_{5}$": [
                {"LR": {"v": 55.00, "b": 0, "c": 0}, "DT": {"v": 94.00, "b": 0, "c": 0}},
                {"LR": {"v": 70.56, "b": 0, "c": 0}, "DT": {"v": 94.67, "b": 0, "c": 0}},
                {"LR": {"v": 93.36, "b": 0, "c": 0}, "DT": {"v": 86.21, "b": 1, "c": 0}},
                {"LR": {"v": 94.38, "b": 0, "c": 0}, "DT": {"v": 88.18, "b": 0, "c": 0}},
                ],
            "Tilia$^{+}_{5}$": [
                {"LR": {"v": 45.83, "b": 0, "c": 0}, "DT": {"v": 95.00, "b": 1, "c": 0}},
                {"LR": {"v": 63.89, "b": 0, "c": 0}, "DT": {"v": 96.67, "b": 1, "c": 1}},
                {"LR": {"v": 92.96, "b": 0, "c": 0}, "DT": {"v": 85.52, "b": 0, "c": 0}},
                {"LR": {"v": 94.76, "b": 0, "c": 0}, "DT": {"v": 88.39, "b": 0, "c": 0}},
                ],
            "Tilia$^{+}_{10}$": [
                {"LR": {"v": 58.33, "b": 1, "c": 0}, "DT": {"v": 90.00, "b": 0, "c": 0}},
                {"LR": {"v": 72.22, "b": 1, "c": 0}, "DT": {"v": 93.33, "b": 0, "c": 0}},
                {"LR": {"v": 93.13, "b": 0, "c": 0}, "DT": {"v": 85.91, "b": 0, "c": 0}},
                {"LR": {"v": 94.62, "b": 0, "c": 0}, "DT": {"v": 89.51, "b": 1, "c": 0}},
                ],
            "Tilia$^{+}_{15}$": [
                {"LR": {"v": 58.33, "b": 1, "c": 0}, "DT": {"v": 90.00, "b": 0, "c": 0}},
                {"LR": {"v": 72.22, "b": 1, "c": 0}, "DT": {"v": 93.33, "b": 0, "c": 0}},
                {"LR": {"v": 93.99, "b": 1, "c": 0}, "DT": {"v": 85.49, "b": 0, "c": 0}},
                {"LR": {"v": 95.04, "b": 1, "c": 1}, "DT": {"v": 89.20, "b": 0, "c": 0}},
                ]
            }
        }


dataset = ["books"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", default=1, type=int, required=False, help="Save to local or not.")
    ap.add_argument("-f", "--format", default="png", required=False, help="Format of the output figure, png or pdf.")
    ap.add_argument("-g", "--group", default="sample", required=False, help="Way to group subplots, sample or method.")
    ap.add_argument("-r", "--regressor", default=None, required=False, help="Regressor, linear or tree.")
    ap.add_argument("-s", "--sample", default=-1, type=int, required=False, help="Number of samples to visualize.")
    ap.add_argument("-t", "--timestamp", default=None, required=False, help="Timestamp")
    args = ap.parse_args()

    if args.regressor == "linear":
        args.regressor = None

    if args.timestamp is None:
        args.timestamp = datetime.datetime.now().strftime("%Y%m%d")

    np.set_printoptions(precision=4, linewidth=200)

    if args.group == "sample":
        xpos = [1, 3, 6, 8, 11, 13, 16, 18, 21, 23, 26, 28]
        tick = [2, 7, 12, 17, 22, 27]

        for ds in dataset:
            output_dir = f"{config.ROOT_DIR}/output/{args.timestamp}/{ds}/fig"
            os.makedirs(output_dir, exist_ok=True)

            fig_w = 10
            fig_h = 7.5
            fig = plt.figure(figsize=(fig_w, fig_h), layout="constrained")

            ttl = fig.suptitle("Faithfulness $\\uparrow$")

            w_ratio =[1] * len(recall[args.group]["Sample = 5"]) * 2
            h_ratio = [2] * 2
            gs = fig.add_gridspec(
                    nrows=2, ncols=len(w_ratio),
                    width_ratios=w_ratio, height_ratios=h_ratio,
                    wspace=0.0, hspace=0.0,
                    )

            axes = {}
            axes["Sample = 5"] =fig.add_subplot(gs[0, 0:6])
            axes["Sample = 10"] =fig.add_subplot(gs[0, 6:])
            axes["Sample = 200"] =fig.add_subplot(gs[1, 0:6])
            axes["Sample = 400"] =fig.add_subplot(gs[1, 6:])

            axes["Sample = 5"].set_xticks([])
            axes["Sample = 5"].set_xticklabels([])
            axes["Sample = 5"].set_yticks([0, 50, 100])
            axes["Sample = 5"].set_yticklabels(["0", "50", "100"])

            axes["Sample = 10"].set_xticks([])
            axes["Sample = 10"].set_xticklabels([])
            axes["Sample = 10"].set_yticks([])
            axes["Sample = 10"].set_yticklabels([])

            axes["Sample = 200"].set_xticks(tick)
            axes["Sample = 200"].set_xticklabels(method)
            axes["Sample = 200"].set_yticks([0, 50, 100])
            axes["Sample = 200"].set_yticklabels(["0", "50", "100"])

            axes["Sample = 400"].set_xticks(tick)
            axes["Sample = 400"].set_xticklabels(method)
            axes["Sample = 400"].set_yticks([])
            axes["Sample = 400"].set_yticklabels([])

            lgd_lr = matplotlib.patches.Patch(facecolor=palette["bar"]["LR"], edgecolor="black", label="LR", alpha=alpha["bar"])
            lgd_dt = matplotlib.patches.Patch(facecolor=palette["bar"]["DT"], edgecolor="black", label="DT", alpha=alpha["bar"], hatch="..")
            axes["Sample = 400"].legend(
                    handles=[lgd_lr, lgd_dt],
                    loc="lower right",
                    fancybox=True,
                    framealpha=0.8,
                    shadow=True,
                    )

            for name, ax in axes.items():
                ax.set_aspect(0.15)
                ax.set_title(name)
                ax.set_ylim(0, 120)
                ax.margins(x=0, y=0)

                lr_v = [item["LR"] for item in fidelity[args.group][name]]
                dt_v = [item["DT"] for item in fidelity[args.group][name]]
                ax.plot(
                        [1, 6, 11, 16, 21, 26],
                        lr_v,
                        color=palette["fid"]["LR"],
                       label="LR",
                        marker="o", markerfacecolor="white",
                        markersize=line_width*2, markeredgewidth=max(line_width//2, 1),
                        linewidth=line_width,
                        alpha=alpha["fid"],
                        )
                ax.plot(
                        [3, 8, 13, 18, 23, 28],
                        dt_v,
                        color=palette["fid"]["DT"],
                        label="DT",
                        marker="s", markerfacecolor="white",
                        markersize=line_width*2, markeredgewidth=max(line_width//2, 1),
                        linewidth=line_width,
                        alpha=alpha["fid"])

                for idx, val in enumerate(recall[args.group][name]):
                    ax.bar(
                        xpos[idx * 2],
                        val["LR"]["v"],
                        width=2,
                        # color=palette[idx],
                        color=palette["bar"]["LR"],
                        edgecolor="black",
                        alpha=alpha["bar"],
                        )
                    # NOTE (2025-01-24): Only show best values.
                    if val["LR"]["b"]:
                        string = f"\\textbf{{{val['LR']['v']:.2f}}}" if val["LR"]["b"] else f"{val['LR']['v']:.2f}"
                        text = ax.text(
                            xpos[idx * 2],
                            val["LR"]["v"] + text_offset,
                            s = string,
                            fontsize=12 if val["LR"]["b"] else 8,
                            ha="center",
                            va="bottom",
                            )
                        if val["LR"]["c"]:
                            text.set_bbox(dict(facecolor="#d5d5d5", edgecolor="#d5d5d5", alpha=0.8, boxstyle="ellipse"))

                    ax.bar(
                        xpos[idx * 2 + 1],
                        val["DT"]["v"],
                        width=2,
                        # color=palette[idx],
                        color=palette["bar"]["DT"],
                        edgecolor="black",
                        alpha=alpha["bar"],
                        hatch="..",
                        )
                    # NOTE (2025-01-24): Only show best values.
                    if val["DT"]["b"]:
                        string = f"\\textbf{{{val['DT']['v']:.2f}}}" if val["DT"]["b"] else f"{val['DT']['v']:.2f}"
                        # text = f"\\colorbox{{gray!25}}{{{text}}}" if val["DT"]["c"] else text
                        text = ax.text(
                            xpos[idx * 2 + 1],
                            val["DT"]["v"] + text_offset,
                            s = string,
                            fontsize=12 if val["DT"]["b"] else 8,
                            ha="center",
                            va="bottom",
                            )
                        if val["DT"]["c"]:
                            text.set_bbox(dict(facecolor="#d5d5d5", edgecolor="#d5d5d5", alpha=0.8, boxstyle="round"))

            if args.debug:
                plt.show()
                exit()
            else:
                plt.savefig(f"{output_dir}/faithfulness_{args.group}_{fig_w}_{fig_h}.{args.format}", bbox_inches="tight")
    elif args.group == "method":
        xpos = [1, 3, 6, 8, 11, 13, 16, 18]

        xtick = [2, 7, 12, 17]
        xticklabel = ["5", "10", "200", "400"]

        ytick = [0, 50, 100]
        yticklabel = ["0", "50", "100"]
        # ytick = [40, 70, 100]
        # yticklabel = ["40", "70", "100"]

        for ds in dataset:
            output_dir = f"{config.ROOT_DIR}/output/{args.timestamp}/{ds}/fig"
            os.makedirs(output_dir, exist_ok=True)

            fig_w = 12
            fig_h = 6
            fig = plt.figure(figsize=(fig_w, fig_h), layout="constrained")
            fig.suptitle("Faithfulness $\\uparrow$")

            w_ratio =[1] * len(recall[args.group]["LIME$_5$"]) * 3
            h_ratio = [2] * 2
            gs = fig.add_gridspec(
                    nrows=2, ncols=len(w_ratio),
                    width_ratios=w_ratio, height_ratios=h_ratio,
                    wspace=0.0, hspace=0.0,
                    )

            axes = {}
            axes["LIME$_5$"] = fig.add_subplot(gs[0, 0:4])
            axes["SHAP$_5$"] = fig.add_subplot(gs[0, 4:8])
            axes["Tilia$_{5}$"] = fig.add_subplot(gs[0, 8:])
            axes["Tilia$^{+}_{5}$"] = fig.add_subplot(gs[1, 0:4])
            axes["Tilia$^{+}_{10}$"] = fig.add_subplot(gs[1, 4:8])
            axes["Tilia$^{+}_{15}$"] = fig.add_subplot(gs[1, 8:])

            axes["LIME$_5$"].set_xticks([])
            axes["LIME$_5$"].set_xticklabels([])
            axes["LIME$_5$"].set_yticks(ytick)
            axes["LIME$_5$"].set_yticklabels(yticklabel)

            axes["SHAP$_5$"].set_xticks([])
            axes["SHAP$_5$"].set_xticklabels([])
            axes["SHAP$_5$"].set_yticks(ytick)
            axes["SHAP$_5$"].set_yticklabels([])
            axes["SHAP$_5$"].yaxis.set_tick_params(length=0)

            axes["Tilia$_{5}$"].set_xticks([])
            axes["Tilia$_{5}$"].set_xticklabels([])
            axes["Tilia$_{5}$"].set_yticks(ytick)
            axes["Tilia$_{5}$"].set_yticklabels([])
            axes["Tilia$_{5}$"].yaxis.set_tick_params(length=0)

            axes["Tilia$^{+}_{5}$"].set_xticks(xtick)
            axes["Tilia$^{+}_{5}$"].set_xticklabels(xticklabel)
            axes["Tilia$^{+}_{5}$"].set_yticks(ytick)
            axes["Tilia$^{+}_{5}$"].set_yticklabels(yticklabel)

            axes["Tilia$^{+}_{10}$"].set_xticks(xtick)
            axes["Tilia$^{+}_{10}$"].set_xticklabels(xticklabel)
            axes["Tilia$^{+}_{10}$"].set_yticks(ytick)
            axes["Tilia$^{+}_{10}$"].set_yticklabels([])
            axes["Tilia$^{+}_{10}$"].yaxis.set_tick_params(length=0)

            axes["Tilia$^{+}_{15}$"].set_xticks(xtick)
            axes["Tilia$^{+}_{15}$"].set_xticklabels(xticklabel)
            axes["Tilia$^{+}_{15}$"].set_yticks(ytick)
            axes["Tilia$^{+}_{15}$"].set_yticklabels([])
            axes["Tilia$^{+}_{15}$"].yaxis.set_tick_params(length=0)

            # lgd_lr = matplotlib.patches.Patch(facecolor="#00007f", marker="o", label="LR", alpha=0.99)
            # lgd_dt = matplotlib.patches.Patch(facecolor="#db8bf4", marker="s", label="DT", alpha=0.99)
            # axes["Tilia$^{+}_{15}$"].legend(
            #         handles=[lgd_lr, lgd_dt],
            #         loc="lower right",
            #         fancybox=True,
            #         framealpha=0.8,
            #         shadow=True,
            #         )

            for name, ax in axes.items():
                ax.grid(axis="y")
                # ax.set_aspect(0.04)
                ax.set_title(name)
                ax.set_ylim(min(ytick), max(ytick)+20)
                ax.margins(x=0, y=0)

                x = range(len(recall[args.group][name]))
                lr_v = [item["LR"]["v"] for item in recall[args.group][name]]
                dt_v = [item["DT"]["v"] for item in recall[args.group][name]]

                lr_f = [item["LR"] for item in fidelity[args.group][name]]
                dt_f = [item["DT"] for item in fidelity[args.group][name]]

                ax.bar([1, 6, 11, 16], lr_v, color="#00007f", width=2, edgecolor="black", label="LR", alpha=alpha["bar"])
                ax.bar([3, 8, 13, 18], dt_v, color="#db8bf4", width=2, edgecolor="black", label="DT", alpha=alpha["bar"], hatch="..")

                ax.plot([1, 6, 11, 16], lr_f, color="#00007f", marker="o", markerfacecolor="white", markersize=10, linewidth=5, alpha=alpha["fid"])
                ax.plot([3, 8, 13, 18], dt_f, color="#db8bf4", marker="s", markerfacecolor="white", markersize=10, linewidth=5, alpha=alpha["fid"])

            axes["Tilia$^{+}_{15}$"].legend(
                    fancybox=True,
                    framealpha=0.8,
                    loc="lower right",
                    shadow=True,
                    )

                # for idx, val in enumerate(recall[args.group][name]):
                #     ax.bar(
                #         xpos[idx * 2],
                #         val["LR"]["v"],
                #         width=2,
                #         color=palette[idx],
                #         edgecolor="black",
                #         hatch="",
                #         )
                #     string = f"\\textbf{{{val['LR']['v']:.2f}}}" if val["LR"]["b"] else f"{val['LR']['v']:.2f}"
                #     text = ax.text(
                #         xpos[idx * 2],
                #         val["LR"]["v"] + 1,
                #         s = string,
                #         fontsize=10 if val["LR"]["b"] else 8,
                #         fontweight = "black" if val["LR"]["b"] else "normal",
                #         ha="center",
                #         va="bottom",
                #         )
                #     if val["LR"]["c"]:
                #         text.set_bbox(dict(facecolor="#d5d5d5", edgecolor="#d5d5d5", alpha=0.8, boxstyle="square"))

                #     ax.bar(
                #         xpos[idx * 2 + 1],
                #         val["DT"]["v"],
                #         width=2,
                #         color=palette[idx],
                #         edgecolor="black",
                #         hatch="..",
                #         )
                #     string = f"\\textbf{{{val['DT']['v']:.2f}}}" if val["DT"]["b"] else f"{val['DT']['v']:.2f}"
                #     # text = f"\\colorbox{{gray!25}}{{{text}}}" if val["DT"]["c"] else text
                #     text = ax.text(
                #         xpos[idx * 2 + 1],
                #         val["DT"]["v"] + 1,
                #         s = string,
                #         fontsize=10 if val["DT"]["b"] else 8,
                #         ha="center",
                #         va="bottom",
                #         )
                #     if val["DT"]["c"]:
                #         text.set_bbox(dict(facecolor="#d5d5d5", edgecolor="#d5d5d5", alpha=0.8, boxstyle="square"))

            if args.debug:
                plt.show()
                exit()
            else:
                plt.savefig(f"{output_dir}/faithfulness_{args.group}_{fig_w}_{fig_h}.{args.format}", bbox_inches="tight")
    else:
        raise ValueError(f"Invalid group: {args.group}")
