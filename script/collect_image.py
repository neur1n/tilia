#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import argparse
import colorsys
import copy
import datetime
import itertools
import keras
from keras import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import scipy.spatial.distance
import skimage.segmentation
import tensorflow as tf

import config
import lime.lime_image
import pandas as pd


# def generate_color(n):
#     cmap = plt.get_cmap("viridis")
#     colors = [cmap(i / n)[:3] for i in range(n)]
#     return colors


# def generate_color(n):
#     if n <= 1:
#         return [(1.0, 0.0, 0.0)]

#     hue_start = 0.0  # Red
#     hue_end = 270.0 / 360.0  # Violet

#     colors = [colorsys.hsv_to_rgb(hue_start + (hue_end - hue_start) * i / (n - 1), 1.0, 1.0) for i in range(n)]

#     return colors


def pairwise_cosine(boundaries):
    indices = []
    for (mask1, mask2) in itertools.combinations(boundaries, 2):
        indices.append(1 - scipy.spatial.distance.cosine(mask1.flatten(), mask2.flatten()))
    return indices


rainbow = [
        (1.0, 0.0, 0.0),  # Red
        (1.0, 0.5, 0.0),  # Orange
        (1.0, 1.0, 0.0),  # Yellow
        (0.0, 1.0, 0.0),  # Green
        (0.0, 0.0, 1.0),  # Blue
        (0.3, 0.0, 0.5),  # Indigo
        (0.6, 0.0, 1.0)   # Violet
        ]


dataset = [
        # "n02085936_Maltese_dog.JPEG.jpg",
        # "alligator.JPEG",
        # "beeflower.JPEG",
        "dog.JPEG",
        ]


patch = [5]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--topk", default=5, type=int, required=False, help="Top-k features.")
    ap.add_argument("-r", "--regressor", default=None, required=False, help="Regressor, linear or tree")
    ap.add_argument("-t", "--timestamp", default=None, required=False, help="Timestamp")
    args = ap.parse_args()

    if args.timestamp is None:
        args.timestamp = datetime.datetime.now().strftime("%Y%m%d")

    tf.config.run_functions_eagerly(False)
    black_box: keras.Model = keras.applications.InceptionV3(weights="imagenet")
    black_box.trainable = False  # Put the model in inference mode

    for name in dataset:
        output_dir = f"{config.ROOT_DIR}/output/{args.timestamp}/{name}/{args.regressor}"
        os.makedirs(output_dir, exist_ok=True)

        image: PIL.Image.Image = preprocessing.image.load_img(
                f"{config.ROOT_DIR}/dataset/{name}", target_size=(299, 299))

        input: np.ndarray = preprocessing.image.img_to_array(image)
        input = keras.applications.inception_v3.preprocess_input(input)

        score: np.ndarray = black_box.predict(np.expand_dims(input, axis=0))
        probability: list[tuple[str, str, float]] = \
                keras.applications.inception_v3.decode_predictions(score, top=args.topk)[0]
        print(f"{probability}\n\n")

        top_label = [p[0] for p in probability]
        top_desc = [p[1] for p in probability]
        top_prob = [p[2] for p in probability]

        exp_label = np.argmax(score)
        print("\n============================================================")
        print(f"Explaining  of {name} (ground truth: {top_label[0]} - {exp_label})...")

        max_depth = "adaptive" if args.regressor is not None else 0

        exp = lime.lime_image.LimeImageExplainer(random_state=config.SEED)

        df_exp = pd.DataFrame()
        for seed in config.SEED:
            with tf.device("/GPU:0"):
                exp_inst = exp.explain_instance(
                        image=input,
                        classifier_fn=black_box,
                        labels=top_label,
                        top_labels=args.topk,
                        num_samples=1000,
                        batch_size=100,
                        model_regressor=args.regressor,
                        max_depth=max_depth)
            for i, e in enumerate(exp_inst.local_exp[exp_label]):
                df_exp = pd.concat([df_exp, pd.DataFrame({"feature": e[0], "importance": e[1]}, index=[i])])
            # exp_label = max(exp_inst.top_labels, key=exp_inst.score.get)
            # seg = []
            # for e in exp_inst.local_exp[exp_label]:
            #     seg.append(e[0])
            # all_mask.append(np.array(sorted(seg[0:10])))
            # all_mask.append(np.array(seg[0:10]))
            print(f"\n{exp_label} -> {exp_inst.score[exp_label]}")
            # print(f"\n{exp_inst.top_labels[0]} -> {exp_inst.score[exp_inst.top_labels[0]]}")

            temp = np.zeros_like(input)
            mask = np.zeros(input.shape[:2])
            output = np.zeros_like(input)

            for p in patch:
                temp, mask = exp_inst.get_image_and_mask(exp_label, num_features=p)
                output = skimage.segmentation.mark_boundaries(
                        temp / 2 + 0.5, mask, color=(1, 0, 0))

                # output = np.zeros_like(input)
                # temp = np.zeros_like(input)
                # mask = np.zeros(input.shape[:2])
                # for i, l in enumerate(exp_inst.top_labels):
                #     temp, mask = exp_inst.get_image_and_mask(l, num_features=n_feature)
                #     if i == 0:
                #         output = skimage.segmentation.mark_boundaries(
                #                 temp / 2 + 0.5, mask, color=rainbow[i % len(rainbow)])
                #     else:
                #         output = skimage.segmentation.mark_boundaries(
                #                 output, mask, color=rainbow[i % len(rainbow)])

                # PIL.Image.fromarray((output * 255).astype(np.uint8)).save(
                #         f"{output_dir}/lbl{top_desc[0]}_feat{n_feature}_samp{n_sample}_rid{r}.png")

                df_exp.to_csv(f"{output_dir}/exp_lbl{top_desc[0]}_patch{p}_seed{seed}.csv", index=False)

                PIL.Image.fromarray((mask * 255).astype(np.uint8)).save(
                        f"{output_dir}/mask_lbl{top_desc[0]}_patch{p}_seed{seed}.png")
                PIL.Image.fromarray((output * 255).astype(np.uint8)).save(
                        f"{output_dir}/lbl{top_desc[0]}_patch{p}_seed{seed}.png")
