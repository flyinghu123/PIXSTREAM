import time, os, gzip, logging, json, hashlib
from dataclasses import dataclass

import cv2 as cv
import numpy as np
import pyperclip
from screeninfo import get_monitors
from tqdm import tqdm

formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
file_handler = logging.FileHandler(filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class HCODE(object):
    '''
    byte -> 8bit -> 0-255 -> 2 * 4bit -> 2 * (0-15) -> 2 * 8bit -> 2 * (0-255) -> 2byte
    '''
    def __init__(self):
        self.redundancy = 2

    def encode(self, data: np.ndarray):
        right = np.bitwise_and(data, 0x0F)
        left = np.right_shift(data, 4)
        data = np.concatenate((left, right)) * 16 + 7
        return data

    def decode(self, data: np.ndarray):
        data = data // 16
        left, right = np.split(data, 2)
        data = np.left_shift(left, 4) + right
        return data


@dataclass
class Package:  # md5, start, size, data
    start: int
    data: bytes
    max_size: int = 1024
    md5_length: int = 4
    def to_bytes(self):
        data = self.start.to_bytes(4, 'big', signed=True) + \
                len(self.data).to_bytes(4, 'big') + \
                self.data
        assert len(data) <= self.max_size - self.md5_length
        # pad
        data = data + b'\x00' * (self.max_size - self.md5_length - len(data))
        md5 = hashlib.md5(data).digest()[:self.md5_length]
        data = md5 + data
        return data
    
    def to_ndarray(self):
        return np.frombuffer(self.to_bytes(), np.uint8)
    
    @classmethod
    def from_ndarray(cls, data: np.ndarray):
        assert len(data) == cls.max_size
        data = data.tobytes()
        md5 = data[:cls.md5_length]
        if md5 != hashlib.md5(data[cls.md5_length:]).digest()[:cls.md5_length]:
            return None
        data = data[cls.md5_length:]
        start = int.from_bytes(data[:4], 'big', signed=True)
        size = int.from_bytes(data[4:8], 'big')
        data = data[8:size + 8]
        return Package(start, data)
    
    @classmethod
    def ndarray2packages(cls, datas: np.ndarray):
        def func(d):
            return np.frombuffer(hashlib.md5(d.tobytes()).digest()[:Package.md5_length], np.uint8)
        md5s, datas = np.split(datas, [Package.md5_length], axis=1)
        md5sc = np.apply_along_axis(func, 1, datas)
        datas = datas[np.all(md5s == md5sc, axis=1)]
        packages = []
        for data in datas:
            assert len(data) == cls.max_size
            data = data.tobytes()
            start = int.from_bytes(data[:4], 'big', signed=True)
            size = int.from_bytes(data[4:8], 'big')
            data = data[8:size + 8]
            packages.append(Package(start, data))
        return packages
    
    @classmethod
    def bytes2packages(cls, data: bytes):
        packages = {}
        max_data_size = cls.max_size - cls.md5_length - 4 - 4
        for i in range(0, len(data), max_data_size):
            package = Package(i, data[i:i+max_data_size])
            packages[i] = package.to_ndarray(), len(package.data)
        return packages

class Sender:
    def __init__(self, file_name, RECT, use_gzip: bool = True, md5_length: int = 4, package_size: int = 1024):
        assert os.path.exists(file_name), "File not found: %s" % file_name
        self.use_gzip = use_gzip
        self.file_name = file_name
        self.md5_length = md5_length
        self.hcode = HCODE()
        self.RECT = RECT
        self.capacity = self.RECT['height'] * self.RECT['width'] * 3
        self.package_size = package_size
        self.package_capacity = self.capacity // (package_size * self.hcode.redundancy)
        self.buffer = np.random.randint(0, 255, (self.package_capacity, self.package_size), np.uint8)

        Package.max_size = self.package_size
        Package.md5_length = self.md5_length
        
        with open(file_name, 'rb') as f:
            file_data = f.read()
            if self.use_gzip:
                file_data = gzip.compress(file_data)
                
        file_packages = Package.bytes2packages(file_data)
        info = {'md5': hashlib.md5(file_data).hexdigest()[:10],
                'file_name': os.path.basename(file_name),
                'use_gzip': self.use_gzip,
                'package_num': len(file_packages),
                'file_size': len(file_data)}
        info_bytes = json.dumps(info, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        self.send_size = len(info_bytes) + len(file_data)
        self.packages = {-1: (Package(-1, info_bytes).to_ndarray(), len(info_bytes))}
        self.packages.update(file_packages)
     
    def send_next(self):
        packages = list(self.packages.values())[:self.package_capacity]
        send_packages_ndarray = np.stack([package[0] for package in packages], 0)
        send_packages_ndarray = send_packages_ndarray[np.random.permutation(len(send_packages_ndarray))]
        buffer = np.pad(send_packages_ndarray, ((0, self.package_capacity-len(send_packages_ndarray)), (0, 0)), 'symmetric')
        data = buffer.flatten()
        data = self.hcode.encode(data)
        image = np.pad(data, (0, self.capacity - data.size), 'reflect').reshape(
            self.RECT['height'], self.RECT['width'], 3
        )
        return image
    
    def quit(self):
        cv.destroyAllWindows()
        exit()
        
    def run(self):
        start_time = time.time()
        windows_name = 'send'
        cv.namedWindow(
            windows_name,
            flags=(cv.WINDOW_NORMAL | cv.WINDOW_GUI_NORMAL | cv.WINDOW_FREERATIO))
        cv.setWindowProperty(windows_name, cv.WND_PROP_TOPMOST, 1.0)
        cv.setWindowProperty(windows_name, cv.WND_PROP_FULLSCREEN, 1.0)
        cv.resizeWindow(
            windows_name,
            self.RECT['width'],
            self.RECT['height'])
        cv.moveWindow(
            windows_name,
            self.RECT['left'],
            self.RECT['top'])
        logger.info(f"start send, file_name: {os.path.basename(self.file_name)}")
        last_key = 0
        with tqdm(total=self.send_size, unit_scale=True, unit_divisor=1024, desc='send') as pbar:
            while self.packages:
                image = self.send_next()
                cv.imshow(windows_name, image)
                while True:
                    key = cv.waitKey(10)
                    if key == 27: self.quit()
                    msg = pyperclip.paste().strip()
                    try: msg = json.loads(msg)
                    except: continue
                    if msg['key'] == last_key: continue
                    last_key = msg['key']
                    for finished_start in msg['finished']:
                        if finished_start not in self.packages:
                            continue
                        pbar.update(self.packages[finished_start][1])
                        del self.packages[finished_start]
                    break

        logger.info(f'end send, {time.time() - start_time:.2f} s')
        self.quit()

            
import win32clipboard
def get_clipboard_file():
    # 打开剪贴板
    win32clipboard.OpenClipboard()
    try:
        # 检查剪贴板中是否有 CF_HDROP 格式的数据
        if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_HDROP):
            data = win32clipboard.GetClipboardData(win32clipboard.CF_HDROP)
            for i in range(len(data)):
                file_path = data[i]
                if os.path.exists(file_path):
                    return file_path
    finally:
        # 关闭剪贴板
        win32clipboard.CloseClipboard()
    return None
if __name__ == "__main__":
    import os, sys
    fn = (len(sys.argv) > 1 and sys.argv[1].strip()) or pyperclip.paste().strip() or get_clipboard_file()
    if not fn:
        logger.warning("Usage: python client.py <filename>")
        exit(1)
    if not os.path.exists(fn):
        logger.warning("File not found: %s" % fn)
        exit(1)
    WIDTH, HEIGHT = 1024, 1024
    RECT = get_monitors()[0]
    RECT = {
        'left': RECT.width + RECT.x - WIDTH,
        'top': RECT.y,
        'width': WIDTH,
        'height': HEIGHT,
        'mon': int(RECT.name[-1])
    }
    Sender(fn, RECT).run()

