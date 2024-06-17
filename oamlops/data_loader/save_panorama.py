from os import makedirs, path
from moma_sqlite_data_handlers.handlers.panorama import Panorama
from moma_sqlite_data_handlers.handlers.camera import Camera
from .sqlite_data_provider import SqliteDataProvider
from PIL import Image
from time import time


class PanoramaProviderLocal:
    def __init__(self, session_path, zoom_level, camera_type):
        self.session_name = path.basename(session_path).split(".")[0]
        self.zoom_level = zoom_level
        self.compression = 95
        self.sqlite_data_provider = SqliteDataProvider(sqlite_path=session_path)
        self.camera = Camera(self.sqlite_data_provider.get_camera_by_type(camera_type))

    def get_panorama_count(self):
        return self.camera.frames_count

    def get_panorama(self, panorama_id):
        start = time()
        panorama_raw = self.sqlite_data_provider.get_tiles(self.camera.zoom_levels[self.zoom_level].table_name,
                                                           panorama_id)

        panorama = Panorama(self.camera, self.zoom_level, panorama_raw)
        return panorama.image_data

    def get_and_save_panoramas(self, q, stop_flag, out_dir=""):
        while not q.empty() or not stop_flag.is_set():
            panorama_id = q.get()
            panorama_raw = self.sqlite_data_provider.get_tiles(self.camera.zoom_levels[self.zoom_level].table_name,
                                                            panorama_id)

            panorama = Panorama(self.camera, self.zoom_level, panorama_raw)
            self.__save_bin(path.join(out_dir, f"pano_{panorama_id}.png"), panorama.image_data)
            q.task_done()

    def __save_bin(self, file_path, data):
        Image.fromarray(data).save(file_path)