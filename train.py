from preprocessor import ImagePreprocessor
from model import FaceLocalizerAndRecognizer
from localization_losses import naive_loss, intersection_over_union
import uuid

create_augmented_images = True
resize_images = True


def main():
    batch_size = 16
    filename = f'{str(uuid.uuid4())}.h5'
    image_preprocessor = ImagePreprocessor()
    if resize_images:
        image_preprocessor.resize_images('img', ['train', 'test', 'val'], 'resized_images')
    if create_augmented_images:
        image_preprocessor.create_and_process_labels()

    test_images = image_preprocessor.get_images('augmented_images\\test\\*.jpg')
    train_images = image_preprocessor.get_images('augmented_images\\train\\*.jpg')
    val_images = image_preprocessor.get_images('augmented_images\\val\\*.jpg')

    test_labels = image_preprocessor.get_labels('augmented_images\\test\\*.json')
    train_labels = image_preprocessor.get_labels('augmented_images\\train\\*.json')
    val_labels = image_preprocessor.get_labels('augmented_images\\val\\*.json')

    test = image_preprocessor.prepare_data(test_images, test_labels, batch_size)
    train = image_preprocessor.prepare_data(train_images, train_labels, batch_size)
    val = image_preprocessor.prepare_data(val_images, val_labels, batch_size)

    model_manager = FaceLocalizerAndRecognizer()
    model = model_manager.build_model()

    history = model.fit(train, epochs=1, validation_data=val, callbacks=model_manager.callbacks)
    model_manager.save_model(filename)


if __name__ == '__main__':
    main()
