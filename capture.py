import asyncio
import cv2
import tensorflow as tf
import numpy as np
from serial import Serial


class ImageGetter:
    def __init__(self, image_queue, rectangle_queue, camera, shutdown_event, max_queue_size):
        self.ser = Serial(port='COM3', baudrate=115200)
        self.image_queue = image_queue
        self.rectangle_queue = rectangle_queue
        self.camera = camera
        self.shutdown = shutdown_event
        self.max_queue_size = max_queue_size
        self.camera_width = 640
        self.camera_height = 480
        self.step = 2
        self.center_min = self.camera_width / 2 - self.camera_width / 10
        self.center_max = self.camera_width / 2 + self.camera_width / 10

    def set_shutdown(self):
        self.shutdown.set()

    async def start(self):
        return asyncio.create_task(self.get_images())

    async def get_images(self):
        while not self.shutdown.is_set():
            _, frame = self.camera.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = tf.image.resize(rgb, (224, 224))
            while self.image_queue.qsize() > self.max_queue_size:
                await self.image_queue.get()
            await self.image_queue.put(resized)
            if self.rectangle_queue.qsize() > 0:
                coordinates = await self.rectangle_queue.get()
                point1 = tuple(np.multiply(coordinates[:2], [self.camera_width, self.camera_height]).astype(int))
                point2 = tuple(np.multiply(coordinates[2:], [self.camera_width, self.camera_height]).astype(int))
                cv2.rectangle(frame, point1, point2, (255, 0, 0), 2)
                await self.move_camera(point1, point2)
            cv2.imshow('Tracker', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.shutdown.set()
            await asyncio.sleep(0)

        self.camera.release()
        cv2.destroyAllWindows()

    def get_direction(self, left, right):
        box_center = (left + right) / 2
        if box_center < self.center_min:
            return self.step
        elif box_center > self.center_max:
            return -self.step
        else:
            return 0

    async def move_camera(self, top_left=(0, 0), bottom_right=(0, 0)):
        direction = self.get_direction(top_left[0], bottom_right[0])
        if direction != 0:
            self.ser.write(bytes(str(direction), 'utf-8'))
