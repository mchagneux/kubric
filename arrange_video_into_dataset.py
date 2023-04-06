#%%
from typing import Iterator, Tuple, NamedTuple, Sequence
import jax 
import os 
import tensorflow_datasets as tfds
import shutil 
class Batch(NamedTuple):
  image: jax.Array  # [B, H, W, C]


# def load_dataset(split: str, batch_size: int, seed: int) -> Iterator[Batch]:

#   ds = (
#       tfds.load("binarized_mnist", split=split)
#       .shuffle(buffer_size=10 * batch_size, seed=seed)
#       .batch(batch_size)
#       .prefetch(buffer_size=5)
#       .repeat()
#   )

#   return ds



# path = 'datasets'

# def load_dataset(split: str, batch_size: int, seed: int) -> Iterator[Batch]:
#   ds = (
#      tfds.ImageFolder(path).as_dataset(split=split)
#       .shuffle(buffer_size=10 * batch_size, seed=seed)
#       .batch(batch_size)
#       .prefetch(buffer_size=5)
#       .repeat()
#       .as_numpy_iterator()
#   )
#   return map(lambda x: Batch(x["image"]), ds)


# ds = load_dataset(split='train',
#                   batch_size=10,
#                   seed=0)

# print(next(iter(ds))['image'])

date = '2023_04_05__08_33_12'

images_path = os.path.join('output', date, '0', 'images')
output_path = os.path.join('datasets', date)
train_split_path = os.path.join(output_path, 'train', 'dummy_label')

os.makedirs(train_split_path)

for image in os.listdir(images_path): 
  if 'rgba' in image:
    shutil.copy(os.path.join(images_path, image), os.path.join(train_split_path, image))


# builder = tfds.ImageFolder(output_path)
# print(builder.info)  # num examples, labels... are automatically calculated
# ds = builder.as_dataset(split='train', shuffle_files=True)
# tfds.show_examples(ds, builder.info)
#%%



