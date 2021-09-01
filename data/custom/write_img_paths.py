
def write_train_path(num: int):
    with open('train.txt', 'w') as fp:
        for file_id in range(1, num + 1):
            fp.write(f'data/custom/images/{file_id}.jpg\n')


def write_valid_path(num: int):
    with open('valid.txt', 'w') as fp:
        for file_id in range(1, num + 1):
            fp.write(f'data/custom/images/{file_id}.jpg\n')


if __name__ == '__main__':
    write_train_path(num=122)
    write_valid_path(num=16)
