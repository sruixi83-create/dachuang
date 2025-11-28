import os

# 添加以下代码以抑制GPU相关警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, LambdaCallback
from tensorflow.keras.metrics import AUC, Precision, Recall

from CNN_Model import MultiLabelModel
from DataLoader import DataLoader, labels_list

# 定义全局参数
############################# Parameters ############################
# 是否测试已训练的模型 和 是否加载之前的权重
test_trained_model = True  # 设置为True时，程序将跳过训练步骤，直接使用已训练的模型
load_previous_weights = True  # 设置为True时，程序将加载已保存的模型权重
# test_trained_model = False
# load_previous_weights = False

# 训练样本数量
samples_to_train = 78468  # 3000  # 最大值: 78468
# 验证样本数量
samples_to_val = 3000  # 250  # 最大值: 11219
# 测试样本数量
samples_to_test = 22433  # 2000  # 最大值: 22433
# 训练轮数
epochs = 50
# 批次大小
batch_size = 64
# 图像形状
image_shape = (256, 256, 3)
# 模型学习率
model_learn_rate = 0.0001
# 模型架构
model_architecture = 'dense121'

# 每个批次结束后的空闲时间（秒）
idle_time_on_batch = 0.1
# 每个epoch结束后的空闲时间（秒）
idle_time_on_epoch = 5

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 从JSON文件中加载标签映射，用于将数字标签转换为中文描述
with open('labels_map.json', 'r', encoding='utf-8') as f:
    labels_map = json.load(f)


#####################################################################

# 加载模型
def load_model():
    # 打印加载数据的提示
    print('##### 正在加载数据 #####')  # Loading Data
    # 加载数据
    ################################################ Load Data ################################################
    # 创建DataLoader实例，用于加载训练、验证和测试数据
    data_loader = DataLoader(batch_size=batch_size,
                             img_shape=image_shape,
                             ntrain=samples_to_train,
                             nval=samples_to_val,
                             ntest=samples_to_test,
                             # plot_distribuition_debug=False,
                             plot_distribuition_debug=True,  # 设置为True时，程序将输出样本数量
                             plot_distribuition=False,
                             # plot_distribuition=True,  # 设置为True时，程序将输出数据分布图
                             augment_data=True,
                             # undersample=False,
                             undersample=True,  # 设置为True时，程序将启用下采样平衡数据
                             shuffle=True)

    # 获取训练、验证和测试数据生成器
    train_data = data_loader.load_train_generator()
    val_data = data_loader.load_validation_generator()
    test_data = data_loader.load_test_generator()

    # 创建神经网络模型
    ################################################ Create NN model ################################################
    if not test_trained_model:
        # 打印构建模型的提示
        print('##### 构建神经网络模型 #####')  # Building NN Model
        # 初始化模型
        model = MultiLabelModel(model2load=model_architecture,
                                percent2retrain=0.6,
                                image_dimensions=image_shape,
                                n_classes=len(labels_list)).get_model()

        # 如果需要加载之前的权重
        if load_previous_weights:
            print('##### 加载模型权重 #####')  # Loading Model Weights
            model.load_weights("model_weights.hdf5")

        # 定义优化器，使用Adam优化器，设置学习率和其他参数
        optimizer = Adam(learning_rate=model_learn_rate,
                         beta_1=0.9,
                         beta_2=0.999,
                         epsilon=1e-08,
                         decay=0.0,
                         amsgrad=False)

        # 编译模型，设置损失函数和评估指标
        model.compile(optimizer=optimizer, loss="binary_crossentropy",
                      metrics=['acc', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')])

        # 定义学习率调整策略，当验证损失不再改善时，减少学习率
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                                    mode='max',
                                                    patience=5,
                                                    factor=0.5,
                                                    min_lr=0.000001,
                                                    verbose=1,  # 0 不打印学习率更新信息、1 当学习率发生更新时打印提示信息
                                                    )

        # 定义早停策略，当验证损失不再改善时，提前停止训练
        early_stop = EarlyStopping(monitor="val_loss",
                                   mode="min",
                                   patience=15,
                                   verbose=1
                                   )

        # 定义模型权重保存策略，保存最佳模型权重
        checkpoint = ModelCheckpoint('model_weights.hdf5',
                                     monitor='val_loss',
                                     mode='min',
                                     save_best_only=True,
                                     save_weights_only=True,
                                     verbose=1,  # 0 不打印任何信息、1 打印模型保存的信息
                                     )

        # 定义空闲时间回调函数，在每个epoch和batch结束后，暂停一段时间，减少资源占用
        idle = LambdaCallback(on_epoch_end=lambda batch, logs: time.sleep(idle_time_on_epoch),
                              on_batch_end=lambda batch, logs: time.sleep(idle_time_on_batch))

        # 将模型保存为JSON文件
        with open("model.json", "w") as json_model:
            json_model.write(model.to_json())

        # 打印训练模型的提示
        print('##### 训练模型 #####')  # Training Model
        ########################################## Train Model ###############################################
        # 打印模型摘要
        model.summary()
        # 开始训练模型
        history = model.fit(train_data,
                            validation_data=val_data,
                            epochs=epochs,
                            steps_per_epoch=len(train_data),
                            verbose=1,  # 0 无输出（安静模式），1 显示进度条和每 step 的损失、指标，2 每个 epoch 结束后只输出一行摘要
                            callbacks=[learning_rate_reduction, early_stop, checkpoint, idle],
                            # use_multiprocessing=True,
                            # workers=4
                            )

        # 绘制训练过程中的损失和准确率曲线
        ############################# Check Loss and Accuracy graphics over training ########################
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))

        # 损失曲线
        ax[0].plot(history.history['loss'], label="训练损失 (Train Loss)", color='tab:blue')
        ax[0].plot(history.history['val_loss'], label="验证损失 (Val Loss)", color='tab:orange')
        ax[0].set_title('训练与验证损失')
        ax[0].set_ylabel("Loss")
        ax[0].grid(True)
        ax[0].legend(loc='best')

        # 准确率曲线
        ax[1].plot(history.history['acc'], label="训练准确率 (Train Accuracy)", color='tab:green')
        ax[1].plot(history.history['val_acc'], label="验证准确率 (Val Accuracy)", color='tab:red')
        ax[1].set_title('训练与验证准确率')
        ax[1].set_ylabel("Accuracy")
        ax[1].set_xlabel("Epoch")
        ax[1].grid(True)
        ax[1].legend(loc='best')

        plt.tight_layout()
        plt.savefig("training_history.png", dpi=300)  # 保存为高清图片
        plt.show()

    # 如果测试已训练的模型
    else:  # if use_trained_model:
        # 打印加载模型的提示
        print('##### 加载神经网络模型 #####')  # Loading NN Model
        from tensorflow.keras.models import model_from_json

        # 从JSON文件加载模型
        with open('model.json', 'r') as json_model:
            model = model_from_json(json_model.read())

        # 加载模型权重
        print('##### 加载模型权重 #####')  # Loading Model Weights
        model.load_weights("model_weights.hdf5")

        # 定义优化器
        optimizer = Adam(learning_rate=model_learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0,
                         amsgrad=False)
        # 编译模型
        model.compile(optimizer=optimizer, loss="binary_crossentropy",
                      metrics=['acc', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')])
    return model, test_data


def show_prediction(image_path, predicted_labels):
    img = cv2.imread(image_path)
    if img is None:
        print("无法加载图片，请检查路径是否正确！")
        return

    title = "预测结果: " + ", ".join([f"{zh_label} ({confidence:.2f})" for zh_label, confidence in predicted_labels])
    plt.figure(figsize=(8, 6))
    plt.title(title, fontsize=14)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# 预测单张图片
def predict_image(model, img_path, threshold=0.5):
    # 检查图片是否存在
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"无法找到图片文件：{img_path}")

    # 读取并预处理图片
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("OpenCV 无法读取图片，请检查路径是否正确！")
        img_resized = cv2.resize(img, (image_shape[0], image_shape[1]))
        img_normalized = img_resized.astype(np.float32) / 255.
        img_input = np.expand_dims(img_normalized, axis=0)
    except Exception as e:
        print(f"[错误] 图像预处理失败: {e}")
        return []

    # 模型预测
    try:
        prediction = model.predict(img_input)[0]
    except Exception as e:
        print(f"[错误] 模型预测失败: {e}")
        return []

    # 打印所有预测概率用于调试
    print("【DEBUG】预测概率分布：")
    for i, label in enumerate(labels_list):
        print(f"{label}: {prediction[i]:.4f}")

    # 过滤出置信度大于阈值的标签
    predicted_labels = [
        (labels_map[str(i)], float(pred)) for i, pred in enumerate(prediction) if pred > threshold
    ]

    if len(predicted_labels) == 0:
        predicted_labels = [("Normal", float(1 - np.sum(prediction)))]  # 如果全部低于阈值，默认视为正常

    return predicted_labels


# 主函数
if __name__ == "__main__":
    model, test_data = load_model()
    # 评估模型在测试集上的表现
    # print('##### 模型测试 #####')
    # test_loss, test_acc, test_auc, test_precision, test_recall = model.evaluate(test_data)
    #
    # print(f"测试损失: {test_loss:.4f}")  #衡量模型预测结果与真实标签之间的差异
    # print(f"测试准确率: {test_acc:.4f}")  # 正确预测的样本数占总样本数的比例
    # print(f"AUC: {test_auc:.4f}")
    # print(f"精确率: {test_precision:.4f}")  # 预测为正类的样本中，实际为正类的比例
    # print(f"召回率: {test_recall:.4f}")  # 所有实际为正类的样本中，被模型正确识别出来的比例
    # print('##### 模型测试结束 #####')

    # 输入图片路径
    image_path = input("请输入X光图片的路径：").strip()
    if not image_path:
        print("未输入任何图片路径，程序退出。")
        exit()

    full_image_path = os.path.join('./database/', image_path)
    if not os.path.exists(full_image_path):
        print(f"图片路径不存在: {full_image_path}")
        exit()

    result = predict_image(model, full_image_path)

    print("检测到以下病灶类型及其置信度：")
    for label, confidence in result:
        print(f"- {label}: {confidence:.2f}")

    show_prediction(full_image_path, result)
