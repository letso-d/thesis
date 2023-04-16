from keras.models import Model
import tensorflow as tf


class Localization(Model):
    def __init__(self, model, opt, classloss, localizationloss, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt

    def compile(self, **kwargs):
        super().compile(**kwargs)

    def train_step(self, data, **kwargs):
        x, y = data
        with tf.GradientTape() as tape:
            classes, coords = self.model(x, training=True)

            classification_loss = self.closs(y_pred=tf.cast(y[0], tf.float32), y_true=classes)
            localization_loss = self.lloss(tf.cast(y[1], tf.float32), coords)

            total_loss = localization_loss + 0.5 * classification_loss

            grad = tape.gradient(total_loss, self.model.trainable_variables)

        self.opt.apply_gradients(zip(grad, self.model.trainable_variables))

        return {"total_loss": total_loss, "class_loss": classification_loss, "regress_loss": localization_loss}

    def test_step(self, data, **kwargs):
        x, y = data

        classes, coords = self.model(x, training=False)

        classification_loss = self.closs(y_pred=tf.cast(y[0], tf.float32), y_true=classes)
        localization_loss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = localization_loss + 0.5 * classification_loss

        return {"total_loss": total_loss, "class_loss": classification_loss, "regress_loss": localization_loss}

    def call(self, x, **kwargs):
        return self.model(x, **kwargs)
