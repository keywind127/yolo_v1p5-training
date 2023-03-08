# YOLO Train

from tensorflow.keras.backend import clear_session 
from tensorflow.keras.optimizers import Adam 
import datetime, os, gc

from model import YoloModel
from dataset import YoloData
from metrics import YoloMetrics

model_savename = os.path.join(".", "models/models/yolo_v1_vgg19-{}.h5".format(datetime.datetime.now().strftime("%y%m%d_%H%M%S")))

if (__name__ == "__main__"):

    image_folder = os.path.join(".", "../voc-dataset.v1i.yolov5pytorch/train/images")

    label_folder = os.path.join(".", "../voc-dataset.v1i.yolov5pytorch/train/labels")

    lambda_noobj = 0.5

    lambda_coord = 5.0

    S = 7

    C = 20

    input_shape = (448, 448, 3)

    batch_size = 8

    epochs = 20

    num_images = 128

    learning_rate = 5e-5

    thresh_obj = 0.6

    thresh_iou = 0.6

    yolo_metrics = YoloMetrics(S, C, lambda_coord, lambda_noobj, thresh_obj, thresh_iou)

    train = 1

    if (train == 1):

        yolo_model = YoloModel.build_model(S, C, input_shape) 

        yolo_data = YoloData(S, C, input_shape)

        (quantity, generator) = yolo_data.initialize_generator(batch_size, image_folder, label_folder, num_images) 

        yolo_model.summary()

        yolo_model.compile(optimizer = Adam(learning_rate = learning_rate), loss = yolo_metrics.loss) 

        clear_session()
        gc.collect()

        for epoch in range(epochs):
            print("Epoch: {}/{}".format(epoch + 1, epochs))
            (images, labels) = next(generator())
            mean_average_precision = yolo_metrics.mean_average_precision(labels, yolo_model.predict(images))
            print("Mean Average Precision: {}".format(mean_average_precision))
            yolo_model.fit(generator(), batch_size = batch_size, epochs = 1, shuffle = True, steps_per_epoch = quantity)
            clear_session()
            gc.collect()

            yolo_model.save(model_savename)