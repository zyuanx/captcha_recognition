import os

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras


class CNN:
    def __init__(self):
        model = keras.models.Sequential()
        # 第1层卷积，卷积核大小为1*1，32个，20*80为待训练图片的大小
        model.add(keras.layers.Conv2D(32, (1, 1), padding="same", activation="relu"))
        model.add(keras.layers.MaxPool2D((2, 2), strides=2, padding="same"))

        # 第2层卷积，卷积核大小为1*1，64个
        model.add(keras.layers.Conv2D(64, (1, 1), padding="same", activation="relu"))
        model.add(keras.layers.MaxPool2D((2, 2), strides=2, padding="same"))

        model.add(keras.layers.Flatten())  # 会将三维的张量转为一维的向量
        model.add(keras.layers.Dense(104, activation=None))

        model.build(input_shape=[None, 20, 80, 3])
        # 打印定义的模型的结构
        model.summary()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=[self.custom_acc],
            loss=self.custom_loss,
        )
        self.model = model

    @staticmethod
    def custom_loss(y_true, y_pred):
        """
        自定义损失函数
        :param y_true:
        :param y_pred:
        :return:
        """
        loss_list = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        loss = tf.reduce_mean(loss_list)
        return loss

    @staticmethod
    def custom_acc(y_true, y_pred):
        """
        自定义评估函数
        :param y_true:
        :param y_pred:
        :return:
        """
        equal_list = tf.reduce_all(
            tf.equal(
                tf.argmax(tf.reshape(y_pred, shape=[-1, 4, 26]), axis=2),
                tf.argmax(tf.reshape(y_true, shape=[-1, 4, 26]), axis=2),
            ),
            axis=1,
        )
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))
        return accuracy


class DataSource:
    def __init__(self):
        self.csv_data = pd.read_csv(
            "./GenPics/labels.csv", names=["file_num", "chars"], index_col="file_num"
        )
        self.train_dataset = ""
        self.test_dataset = ""

    def filename2char(self, image_path):
        """
        图片路径到验证码值
        :param image_path:
        :return:
        """
        # print(image_path)
        num = image_path.split("/")[-1].split(".")[0]
        chars = self.csv_data.loc[int(num), "chars"]
        return chars

    def filename2label(self):
        """
        解析csv文件，建立文件名对应的标签值
        abcd -> [0,1,2,3]
        :return:
        """
        labels = []
        for label in self.csv_data["chars"]:
            tmp = []
            for letter in label:
                tmp.append(ord(letter) - ord("A"))
            labels.append(tmp)
        self.csv_data["labels"] = labels
        return labels

    @staticmethod
    def load_and_preprocess_from_path_label(path, label):
        """
        处理验证码与标签
        :param label:
        :param path:
        :return:
        """
        image = tf.io.read_file(path)
        image_decode = tf.image.decode_jpeg(image, channels=3)
        # image_decode = tf.image.rgb_to_grayscale(image_decode)
        image_decode = tf.image.resize(image_decode, [20, 80])
        # 归一化处理
        image_decode /= 255.0
        label_decode = tf.one_hot(label, depth=26)
        label_decode = tf.reshape(label_decode, shape=(104,))

        return image_decode, label_decode

    def read_picture(self):
        """
        读取验证码图片
        :return:
        """
        all_image_paths = ["./GenPics/" + str(i) + ".jpg" for i in range(6000)]
        # 图片顺序得到的标签值
        all_image_labels = self.filename2label()
        # 6000个样本，5900个作为训练集
        # 构建 图片-标签 数据的dataset
        dataset = tf.data.Dataset.from_tensor_slices(
            (all_image_paths[:5900], all_image_labels[:5900])
        )

        # 处理图片与标签
        dataset = dataset.map(self.load_and_preprocess_from_path_label)
        # 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据被充分打乱。
        train_dataset = dataset.shuffle(5900)
        train_dataset = train_dataset.repeat(3)
        train_dataset = train_dataset.batch(512)

        # 6000个样本，100个作为测试集
        # 构建test_dataset
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (all_image_paths[5900:], all_image_labels[5900:])
        )
        test_dataset = test_dataset.map(self.load_and_preprocess_from_path_label)
        test_dataset = test_dataset.shuffle(100)
        return train_dataset, test_dataset


class Train:
    def __init__(self):
        self.cnn = CNN()
        self.data = DataSource()

    def train(self):
        train_dataset, test_dataset = self.data.read_picture()
        if os.path.exists("simple.keras"):
            # 加载训练好的模型，注意需要传入自定义损失评估函数
            model = keras.models.load_model(
                "simple.keras",
                custom_objects={
                    "custom_loss": self.cnn.custom_loss,
                    "custom_acc": self.cnn.custom_acc,
                },
            )
        else:
            # 训练模型，大概3轮
            model = self.cnn.model
            for step, (x, y) in enumerate(train_dataset):
                # print(x.shape, y.shape)  # (512, 20, 80, 3) (512, 104)
                model.fit(x, y)
            model.save("simple.keras")
        for step, (x, y) in enumerate(test_dataset):
            # 因为模型输入shape为(-1,20,80,3)，所以对测试集输入的图片进行0纬扩展
            pred = model.predict(tf.expand_dims(x, axis=0))
            pred = tf.reshape(pred, shape=(-1, 4, 26))
            pred = tf.argmax(pred, axis=2).numpy()[0]
            pred = "-".join([chr(ord("A") + i) for i in pred])
            # 绘图显示预测值，共24个
            plt.subplot(6, 4, step + 1)
            plt.imshow(x)
            plt.xlabel(pred)
            if step == 23:
                plt.show()
                break


if __name__ == "__main__":
    app = Train()
    app.train()
