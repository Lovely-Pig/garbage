import os


def write_train_path(img_name: str):
    with open('train.txt', 'a') as f:
        f.write(f'data/custom/images/{img_name}\n')


def write_valid_path(img_name: str):
    with open('valid.txt', 'a') as f:
        f.write(f'data/custom/images/{img_name}\n')


filepath = 'images/'
filenames = os.listdir(path=filepath)


for filename in filenames:
    write_train_path(filename)


for filename in filenames[-16:]:
    write_valid_path(filename)
