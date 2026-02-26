from io import BytesIO
from pathlib import Path
from typing import Optional, Sequence
from urllib.request import urlopen
from zipfile import ZipFile

import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

letters = [
    "Space",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]


def load(
    thr: int = 1,
    taxels: Optional[Sequence[int]] = None,
    custom_letters: Optional[Sequence[str]] = None,
):
    file_name = Path(f"./braille/data_braille_letters_th_{thr}.pkl")
    if not file_name.exists():
        resp = urlopen(
            "https://zenodo.org/records/7050094/files/reading_braille_data.zip"
        )
        with ZipFile(BytesIO(resp.read())) as zObject:
            zObject.extract(f"data_braille_letters_th_{thr}.pkl", path="./braille")

    data_dict = pd.read_pickle(file_name)

    # Extract data
    data_list: list[np.ndarray] = []
    label_list: list[int] = []

    nchan = len(data_dict["events"][1])  # number of channels per sensor

    max_events = 0
    for i, sample in enumerate(data_dict["events"]):
        for taxel in range(len(sample)):
            for event_type in range(len(sample[taxel])):
                events = sample[taxel][event_type]
                max_events = len(events) if len(events) > max_events else max_events

    for i, sample in enumerate(data_dict["events"]):
        events_array = np.full([nchan, max_events, 2], np.inf)
        for taxel in range(len(sample)):
            # loop over On and Off channels
            for event_type in range(len(sample[taxel])):
                events = sample[taxel][event_type]
                if events:
                    events_array[taxel, : len(events), event_type] = events  # ms

        if taxels is not None:
            events_array = np.reshape(
                np.transpose(events_array, (0, 2, 1))[taxels, :, :],
                (-1, events_array.shape[1]),
            )
            selected_chans = 2 * len(taxels)
        else:
            events_array = np.reshape(
                np.transpose(events_array, (0, 2, 1)), (-1, events_array.shape[1])
            )
            selected_chans = 2 * nchan

        lbl = data_dict["letter"][i]
        if custom_letters is not None:
            if lbl in custom_letters:
                data_list.append(events_array)
                label_list.append(custom_letters.index(lbl))
        else:
            data_list.append(events_array)
            label_list.append(letters.index(lbl))

    data = np.stack(data_list) * 100  # To ms
    labels = np.stack(label_list)
    nb_outputs = len(np.unique(labels))

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.30, shuffle=True, stratify=labels
    )

    x_test, x_validation, y_test, y_validation = train_test_split(
        x_test, y_test, test_size=0.33, shuffle=True, stratify=y_test
    )

    trainset = (jnp.asarray(x_train), jnp.asarray(y_train))
    testset = (jnp.asarray(x_test), jnp.asarray(y_test))
    valset = (jnp.asarray(x_validation), jnp.asarray(y_validation))

    return trainset, testset, valset, selected_chans, nb_outputs


def load_raw(
    taxels: Optional[Sequence[int]] = None,
    custom_letters: Optional[Sequence[str]] = None,
):
    file_name = Path("./braille/data_braille_letters_digits.pkl")
    if not file_name.exists():
        resp = urlopen(
            "https://zenodo.org/records/7050094/files/reading_braille_data.zip"
        )
        with ZipFile(BytesIO(resp.read())) as zObject:
            zObject.extract("data_braille_letters_digits.pkl", path="./braille")

    data_dict = pd.read_pickle(file_name)

    # Extract data
    data_list: list[np.ndarray] = []
    label_list: list[int] = []

    nchan = data_dict["taxel_data"][0].shape[1]  # number of channels per sensor

    for i, sample in enumerate(data_dict["taxel_data"]):
        if taxels is not None:
            sample = sample[:, taxels]
            selected_chans = len(taxels)
        else:
            selected_chans = nchan

        lbl = data_dict["letter"][i]
        if custom_letters is not None:
            if lbl in custom_letters:
                data_list.append(sample)
                label_list.append(custom_letters.index(lbl))
        else:
            data_list.append(sample)
            label_list.append(letters.index(lbl))

    data = np.stack(data_list)  # To ms
    labels = np.stack(label_list)
    nb_outputs = len(np.unique(labels))

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.30, shuffle=True, stratify=labels
    )

    x_test, x_validation, y_test, y_validation = train_test_split(
        x_test, y_test, test_size=0.33, shuffle=True, stratify=y_test
    )

    trainset = (jnp.asarray(x_train), jnp.asarray(y_train))
    testset = (jnp.asarray(x_test), jnp.asarray(y_test))
    valset = (jnp.asarray(x_validation), jnp.asarray(y_validation))

    return trainset, testset, valset, selected_chans, nb_outputs
