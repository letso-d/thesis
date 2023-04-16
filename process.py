import asyncio
import numpy as np


class ImageProcessor:
    def __init__(self, image_queue, rectangle_queue, model, shutdown_event, max_queue_size):
        self.image_queue = image_queue
        self.rectangle_queue = rectangle_queue
        self.model = model
        self.shutdown = shutdown_event
        self.max_queue_size = max_queue_size

    def set_shutdown(self):
        self.shutdown.set()

    async def start(self):
        return asyncio.create_task(self.process_images())

    async def process_images(self):
        while not self.shutdown.is_set():
            resized_image = await self.image_queue.get()
            yhat = self.model.predict(np.expand_dims(resized_image / 255, axis=0))
            predicted_coordinates = yhat[1][0]
            width = predicted_coordinates[2] - predicted_coordinates[0]
            height = predicted_coordinates[3] - predicted_coordinates[1]
            area = width * height
            if area > 0.03 and yhat[0] > 0.998:
                while self.rectangle_queue.qsize() > self.max_queue_size:
                    await self.rectangle_queue.put(predicted_coordinates)
