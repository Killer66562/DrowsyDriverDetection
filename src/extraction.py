import numpy as np
import cv2
import os

from concurrent.futures import ProcessPoolExecutor


class FilePath(object):
    def __init__(self, dir: str, filename: str):
        self.dir = dir
        self.filename = filename


def process_single_image(read_path: str, write_path: str):
    img = cv2.imread(read_path)
    grayscaleImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurGrayscaleImg = cv2.GaussianBlur(grayscaleImg, (3, 3), 0)
    binaryImg = cv2.Canny(blurGrayscaleImg, 30, 100)

    cv2.imwrite(write_path, binaryImg)

def collect_image_paths(datasets_root_path: str) -> list[FilePath]:
    paths = os.listdir(datasets_root_path)
    collected = []

    for path in paths:
        abs_path = os.path.join(datasets_root_path, path)
        if abs_path.endswith((".jpg", ".jpeg", ".png")):
            collected.append(FilePath(dir=datasets_root_path, filename=path))
        elif os.path.isdir(abs_path):
            collected.extend(collect_image_paths(abs_path))

    return collected

def process_images(read_datasets_root_path: str, write_datasets_root_path: str, max_workers: int):
    if not os.path.exists(read_datasets_root_path):
        raise FileNotFoundError()
    if not os.path.exists(write_datasets_root_path):
        os.mkdir(write_datasets_root_path)
    
    paths = collect_image_paths(read_datasets_root_path)
    
    read_paths = [os.path.join(path.dir, path.filename) for path in paths]
    write_paths = []

    for path in paths:
        splited = path.dir.split("/")
        splited = splited[1:]

        write_dir = os.path.join(write_datasets_root_path, *splited)

        if not os.path.exists(write_dir):
            os.mkdir(write_dir)

        write_path = os.path.join(write_dir, path.filename)
        write_paths.append(write_path)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_single_image, read_paths, write_paths, chunksize=64)

def main():
    process_images("datasets", "datasets_processed", max_workers=6)

if __name__ == "__main__":
    main()

'''
#寻找轮廓
#也可以这么写：
#binary,contours, hierarchy = cv2.findContours(binaryImg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
#这样，可以直接用contours表示
h = cv2.findContours(binaryImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#提取轮廓
contours = h[0]
#打印返回值，这是一个元组
print(type(h))
#打印轮廓类型，这是个列表
print(type(h[1]))
#查看轮廓数量
print (len(contours))

#创建白色幕布
temp = np.ones(binaryImg.shape,np.uint8)*255
#画出轮廓：temp是白色幕布，contours是轮廓，-1表示全画，然后是颜色，厚度
cv2.drawContours(temp,contours,-1,(0,255,0),3)

cv2.imshow("contours",temp)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''