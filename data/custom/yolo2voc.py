import os
import cv2


with open('classes.names', 'r') as fp:
    classes = fp.read().splitlines()


def convert_file(file_id: int, input_dir: str, output_dir: str):
    image = cv2.imread(f'images/{file_id}.jpg')
    height, width, depth = image.shape

    with open(f'{output_dir}/{file_id}.xml', 'w') as fp_xml:
        fp_xml.write(f'''\
<annotation>
	<folder>images</folder>
	<filename>{file_id}.jpg</filename>
	<path>data/custom/images/{file_id}.jpg</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>{width}</width>
		<height>{height}</height>
		<depth>{depth}</depth>
	</size>
	<segmented>0</segmented>
''')

    with open(f'{input_dir}/{file_id}.txt', 'r') as fp_txt:
        lines = fp_txt.readlines()
        for line in lines:
            obj_info = line.strip().split(' ')
            label_id, x_center, y_center, o_width, o_height = obj_info
            label_id, x_center, y_center, o_width, o_height = int(label_id), float(x_center), float(y_center), float(o_width), float(o_height)
            label = classes[label_id]
            xmin = round(width * (x_center - 0.5 * o_width))
            ymin = round(height * (y_center - 0.5 * o_height))
            xmax = round(width * (x_center + 0.5 * o_width))
            ymax = round(height * (y_center + 0.5 * o_height))
            with open(f'{output_dir}/{file_id}.xml', 'a') as fp_xml:
                fp_xml.write(f'''\
    <object>
		<name>{label}</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>{xmin}</xmin>
			<ymin>{ymin}</ymin>
			<xmax>{xmax}</xmax>
			<ymax>{ymax}</ymax>
		</bndbox>
	</object>
''')

    with open(f'{output_dir}/{file_id}.xml', 'a') as fp_xml:
        fp_xml.write('</annotation>\n')


def convert_files(num: int, input_dir: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    for file_id in range(1, num + 1):
        convert_file(file_id, input_dir, output_dir)


if __name__ == '__main__':
    convert_files(num=122, input_dir='labels', output_dir='labels2')
