from petastorm import TransformSpec
from torchvision import transforms
from PIL import Image
from functools import partial
import numpy as np
import io
import pandas as pd


# Preprocessing for converting bytes array to torch using TransformSpec in Petastorm
def transform_image(image_bytes: bytearray, resize_shape: tuple, resize: bool = False):
    if resize:
        return np.asarray(Image.open(io.BytesIO(image_bytes)).resize(resize_shape))
    return np.asarray(Image.open(io.BytesIO(image_bytes)))


def transform_row(
    pd_batch: pd.DataFrame,
    image_shape: tuple,
    input_img_column: str,
    output_img_column: str,
    index_column: str,
    resize: bool = False,
):
    pd_batch[output_img_column] = pd_batch[input_img_column].map(
        lambda x: transform_image(x, image_shape, resize)
    )
    pd_batch = pd_batch.filter([index_column, output_img_column])
    return pd_batch


def get_transform_spec(
    image_shape: tuple,
    input_img_column: str,
    output_img_column: str,
    index_column: str,
    resize: bool = False,
):
    # The output shape of the `TransformSpec` is not automatically known by petastorm,
    # so you need to specify the shape for new columns in `edit_fields` and specify the order of
    # the output columns in `selected_fields`.
    return TransformSpec(
        partial(
            transform_row,
            image_shape=image_shape,
            resize=resize,
            input_img_column=input_img_column,
            output_img_column=output_img_column,
            index_column=index_column,
        ),
        edit_fields=[
            (output_img_column, np.float32, (image_shape[1], image_shape[0], 3), False),
            (index_column, np.int32, (), False),
        ],
        selected_fields=[index_column, output_img_column],
    )


def tens_transform_row(resize_shape, pd_batch):
    """
    The input and output of this function must be pandas dataframes.
    Do data augmentation for the training dataset only.
    """
    transformers = [transforms.Lambda(lambda x: Image.open(io.BytesIO(x)))]
    transformers.extend(
        [
            transforms.Resize((resize_shape[0])),
            transforms.ToTensor(),
        ]
    )

    trans = transforms.Compose(transformers)

    pd_batch["image"] = pd_batch["panorama_img"].map(lambda x: trans(x).numpy())
    pd_batch = pd_batch.filter(["frame_index", "image"])
    return pd_batch


def get_tens_transform_spec(resize_shape=(1333, 2666)):
    # The output shape of the `TransformSpec` is not automatically known by petastorm,
    # so you need to specify the shape for new columns in `edit_fields` and specify the order of
    # the output columns in `selected_fields`.
    return TransformSpec(
        partial(tens_transform_row, resize_shape),
        edit_fields=[
            ("image", np.float32, (3, resize_shape[0], resize_shape[1]), False),
            ("frame_index", np.int32, (), False),
        ],
        selected_fields=["frame_index", "image"],
    )
