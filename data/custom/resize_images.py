import os
import cv2


def resize_image(img_id: int, input_dir: str, output_dir: str, max_height: int, max_width: int):
    print(f'{input_dir}/{img_id}.jpg -> {output_dir}/{img_id}.jpg')
    
    img = cv2.imread(f'{input_dir}/{img_id}.jpg')
    height, width, depth = img.shape
    for ratio in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        if height * ratio < max_height and width * ratio < max_width:
            break
    img = cv2.resize(img, dsize=None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(f'{output_dir}/{img_id}.jpg', img=img)


def resize_images(num: int, input_dir: str, output_dir: str, max_height: int, max_width: int):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    for img_id in range(1, num + 1):
        resize_image(img_id, input_dir, output_dir, max_height, max_width)


if __name__ == '__main__':
    resize_images(num=122, input_dir='images', output_dir='images2', max_height=1280, max_width=1280)
