
import gzip, hashlib, json, logging, os, time
from dataclasses import dataclass
from datetime import datetime

import cv2 as cv
import mss
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
file_handler.setLevel(logging.DEBUG)
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
    
class Receiver:
    def __init__(self, RECT, md5_length: int = 4, package_size=1024):
        self.RECT = RECT
        self.md5_length = md5_length
        self.hcode = HCODE()
        self.capacity = self.RECT['height'] * self.RECT['width'] * 3
        self.package_size = package_size
        self.package_capacity = self.capacity // (package_size * self.hcode.redundancy)
        Package.max_size = self.package_size
        Package.md5_length = self.md5_length
        self.packages = {}
        self.file_info = {}
        self.sct = mss.mss()
        
        
    def read_data(self):
        image = np.asarray(self.sct.grab(self.RECT).pixels).astype(np.uint8)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        data = image.reshape(-1)[:self.package_size * self.package_capacity * self.hcode.redundancy]
        data = self.hcode.decode(data)
        data = data.reshape(-1, self.package_size)
        return data

            
    def receive(self):
        with tqdm(unit_scale=True, unit_divisor=1024, desc='receive') as pbar:
            while len(self.packages) != self.file_info.get('package_num', -1):
                finished = list()
                for package in Package.ndarray2packages(self.read_data()):
                    if package.start == -1:
                        # info
                        self.file_info = json.loads(package.data.decode('utf-8'))
                        pbar.total = self.file_info['file_size']
                    elif package.start not in self.packages:
                        self.packages[package.start] = package
                        pbar.update(len(package.data))
                    finished.append(package.start)
                if finished:
                    pyperclip.copy(json.dumps({'key': time.time(), 'finished': finished}, ensure_ascii=False, separators=(',', ':')))
                logger.debug(f'{len(self.packages)}/{self.file_info.get("package_num", -1)}')

    def run(self):
        start_time = time.time()
        logger.info(f'start receive')
        self.receive()
        logger.info(f'end receive, {time.time() - start_time:.2f} s')
        file_data = b''.join([package.data for _, package in sorted(self.packages.items())] )
        if self.file_info['md5'] != hashlib.md5(file_data).hexdigest()[:len(self.file_info['md5'])]:
            logger.error(f'file md5 error')
        if self.file_info['use_gzip']:
            file_data = gzip.decompress(file_data)
        with open(f'{datetime.now().strftime("%Y%m%d%H%M%S")}_' + self.file_info['file_name'], 'wb') as f:
            f.write(file_data)
        self.sct.close()

if __name__ == '__main__':
    WIDTH, HEIGHT = 1024, 1024
    RECT = get_monitors()[-1]
    RECT = {
        "left": RECT.x + RECT.width - WIDTH,
        "top": RECT.y,
        "width": WIDTH,
        "height": HEIGHT,
        "mon": int(RECT.name[-1])
    }
    import win32gui, win32con
    def windows_callback(hwnd, param):
        if win32gui.IsWindow(hwnd) and \
            win32gui.IsWindowEnabled(hwnd) and \
            win32gui.IsWindowVisible(hwnd) and \
            'mvdi' in win32gui.GetWindowText(hwnd):
            win32gui.SetForegroundWindow(hwnd)
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
    win32gui.EnumWindows(windows_callback, 0)
    pyperclip.copy('')
    receiver = Receiver(RECT)
    receiver.run()
