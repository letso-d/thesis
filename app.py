import asyncio

import cv2
from keras.models import load_model
from capture import ImageGetter
from process import ImageProcessor


async def main():
    model = load_model('object_tracking_network.h5')
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, 640)
    cam.set(4, 480)
    image_queue = asyncio.Queue()
    rectangle_queue = asyncio.Queue()
    shutdown = asyncio.Event()
    image_getter = ImageGetter(image_queue, rectangle_queue, camera=cam, shutdown_event=shutdown, max_queue_size=30)
    image_processor = ImageProcessor(image_queue, rectangle_queue, model, shutdown_event=shutdown, max_queue_size=30)
    image_processor_task = await image_processor.start()
    image_getter_task = await image_getter.start()
    await asyncio.gather(image_getter_task, image_processor_task)


if __name__ == '__main__':
    asyncio.run(main())
