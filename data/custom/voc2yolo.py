import os
import xml.etree.ElementTree as ET


with open('classes.names', 'r') as fp:
    classes = fp.read().splitlines()


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars


def convert_file(file_id: int, input_dir: str, output_dir: str):
    print(f'{input_dir}/{file_id}.xml -> {output_dir}/{file_id}.txt')

    tree = ET.parse(f'{input_dir}/{file_id}.xml')
    root = tree.getroot()

    size = get_and_check(root, "size", 1)
    width = int(get_and_check(size, "width", 1).text)
    height = int(get_and_check(size, "height", 1).text)

    for obj in get(root, "object"):
        label = get_and_check(obj, 'name', 1).text
        label_id = classes.index(label)
        bndbox = get_and_check(obj, "bndbox", 1)
        xmin = int(get_and_check(bndbox, "xmin", 1).text)
        ymin = int(get_and_check(bndbox, "ymin", 1).text)
        xmax = int(get_and_check(bndbox, "xmax", 1).text)
        ymax = int(get_and_check(bndbox, "ymax", 1).text)
        x_center = (xmin + xmax) / 2 / width
        y_center = (ymin + ymax) / 2 / height
        o_width = (xmax - xmin) / width
        o_height = (ymax - ymin) / height

        with open(f'{output_dir}/{file_id}.txt', 'a') as fp:
            fp.write('%d %.6f %.6f %.6f %.6f\n' % (label_id, x_center, y_center, o_width, o_height))


def convert_files(num: int, input_dir: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    for file_id in range(1, num + 1):
        convert_file(file_id, input_dir, output_dir)


if __name__ == '__main__':
    convert_files(num=122, input_dir='labels2', output_dir='labels3')
