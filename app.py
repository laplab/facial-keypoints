import uuid
from os.path import join
from pathlib import Path

import click
import numpy as np
import pandas as pd
from keras.models import load_model
from skimage.io import imread, imsave
from skimage.transform import resize
from tqdm import tqdm

from augmenters import NoChangesAugmenter, FlipAugmenter, RotateAugmenter
from model import init_model
from utils import read_coords, center_circle


@click.command()
@click.argument('img_dir', metavar='img')
@click.argument('coords_file', metavar='coords')
@click.argument('dest_dir', metavar='dest')
def augment(img_dir, coords_file, dest_dir):
    """
    Augments images stored in IMG folder with coordinates
    from COORDS csv file and saves result in grayscale to DEST folder
    """
    coords_old = read_coords(coords_file)
    augmenters = [
        FlipAugmenter(),
        NoChangesAugmenter(),
        RotateAugmenter(radius=20, repeat=4)
    ]

    coords_gen = []
    for filename in tqdm(coords_old.keys(), desc='Augmenting data'):
        old_img = imread(join(img_dir, filename), as_grey=True)
        old_points = coords_old[filename]

        for aug in augmenters:
            for new_img, new_points in aug.augment(old_img, old_points):
                title = str(uuid.uuid4()) + '.jpg'
                new_points = new_points.astype(int).tolist()

                imsave(join(dest_dir, title), new_img)
                coords_gen.append([title] + new_points)

    df = pd.DataFrame(coords_gen,
                      columns=['filename', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6', 'x7',
                               'y7', 'x8', 'y8', 'x9', 'y9', 'x10', 'y10', 'x11', 'y11', 'x12', 'y12', 'x13', 'y13',
                               'x14', 'y14'])
    df.to_csv(join(dest_dir, 'coords.csv'))


@click.command()
@click.argument('img_dir', metavar='img')
@click.argument('coords_file', metavar='coords')
@click.argument('model_file', metavar='model')
def train(img_dir, coords_file, model_file):
    """
    Trains model on images from IMG folder with coordinates from COORDS
    csv file and saves trained model in hdf5 file MODEL
    """
    coords = read_coords(coords_file)

    X = []
    y = []
    for filename in tqdm(coords.keys(), desc='Reading data'):
        img = imread(join(img_dir, filename), as_grey=True)
        points = coords[filename]

        height, width = img.shape
        img = resize(img, (100, 100))
        points[::2] *= 100 / width
        points[1::2] *= 100 / height

        X.append(img)
        y.append(points)

    X = np.array(X).reshape((-1, 100, 100, 1))
    y = np.array(y)

    X = center_circle(X)

    model = init_model()
    model.fit(X, y, batch_size=128, epochs=1)
    model.save(model_file)


@click.command()
@click.argument('img_dir', metavar='img')
@click.argument('model_file', metavar='model')
@click.argument('coords_file', metavar='coords')
def predict(img_dir, model_file, coords_file):
    """
    Predicts facial keypoints coordinates for images from IMG folder
    using model from file MODEL and saves results in csv file COORDS
    """
    files = []
    X = []
    x_ratio = []
    y_ratio = []
    for img_path in tqdm(Path(img_dir).iterdir(), desc='Performing inference'):
        if not img_path.is_file():
            continue

        img = imread(str(img_path), as_grey=True)
        height, width = img.shape
        img = resize(img, (100, 100))

        files.append(img_path.name)
        X.append(img)
        x_ratio.append(width / 100)
        y_ratio.append(height / 100)

    X = np.array(X).reshape((-1, 100, 100, 1))
    x_ratio = np.array(x_ratio).reshape((-1, 1))
    y_ratio = np.array(y_ratio).reshape((-1, 1))

    X = center_circle(X)

    model = load_model(model_file)
    coords = model.predict(X)
    coords[:, ::2] *= x_ratio
    coords[:, 1::2] *= y_ratio

    df = pd.DataFrame.from_dict(dict(zip(files, coords.tolist())), orient='index')
    df.to_csv(coords_file)


@click.group()
def app():
    pass

app.add_command(augment)
app.add_command(train)
app.add_command(predict)

if __name__ == '__main__':
    app()
