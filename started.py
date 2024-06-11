import math

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import cv2
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Controller:
    def __init__(self):
        print("\n--Основные----------------------------------------------------------")
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.sim.setStepping(True)
        self.get_NecessaryCoppeliaSimSceneObjectsID()
        self.max_velocity = 3.1415/180*60
        self.threshold = 0.3
        self.delta_time = 0.1
        self.isVerbose = not False
        self.sim.handleVisionSensor(self.sim.handle_all)
        print("\n--Загрузка нейросети----------------------------------------------------------")
        self.class_labels = {0: '0_LeftTr', 1: '1_RightTr', 2: '2_Square', 3: "3_Circle", 4: "4_Rhomb"}
        self.modelNN_loaded = tf.keras.models.load_model('5figures-28x28pix-ConvNN-TF2.14.0.keras')
        print(self.modelNN_loaded.summary())
        print("\n--Инициализация окон отображения OpenCV----------------------------------------------------------")
        self.cvWindow_Front_1 = "Front Original VisualSensor image"
        self.cvWindow_Front_2 = "Front Resized image"
        self.cvWindow_Front_3 = "Front Classify result"
        cv2.namedWindow(self.cvWindow_Front_1, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.cvWindow_Front_2, cv2.WINDOW_NORMAL)  # Создать окно OpenCV
        cv2.namedWindow(self.cvWindow_Front_3, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.cvWindow_Front_1, cv2.WND_PROP_TOPMOST, 1)
        #cv2.setWindowProperty(self.cvWindow_Front_2, cv2.WND_PROP_TOPMOST, 1)  # Отображение окна OpenCV поверх других окон
        cv2.setWindowProperty(self.cvWindow_Front_3, cv2.WND_PROP_TOPMOST, 1)
        self.yellow_mark_detected = False  # Флаг для отслеживания состояния жёлтой метки

    def get_NecessaryCoppeliaSimSceneObjectsID(self):
        self.left_motor_handle = self.sim.getObject('/leftMotor')
        self.right_motor_handle = self.sim.getObject('/rightMotor')
        self.left_cam_handle = self.sim.getObject('/vis_sensor_L1')
        self.right_cam_handle = self.sim.getObject('/vis_sensor_R1')
        self.rightFloorSing_cam_handle = self.sim.getObject('/vis_sensor_Floor')
        self.rightVerticalSing_cam_handle = self.sim.getObject('/vis_sensor_Vert')

        assert self.left_motor_handle != -1, "Left motor not found"
        assert self.right_motor_handle != -1, "Right motor not found"
        assert self.left_cam_handle != -1, "Left camera not found"
        assert self.right_cam_handle != -1, "Right camera not found"
        assert self.rightFloorSing_cam_handle != -1, "Right FloorSing camera not found"
        assert self.rightVerticalSing_cam_handle != -1, "Right VerticalSing camera not found"

        print("All Objects found")
        print("- left_motor_handle", self.left_motor_handle)
        print("- right_motor_handle", self.right_motor_handle)
        print("- left_cam_handle", self.left_cam_handle)
        print("- right_cam_handle", self.right_cam_handle)
        print("- rightFloorSing_cam_handle", self.rightFloorSing_cam_handle)
        print("- rightVerticalSing_cam_handle", self.rightVerticalSing_cam_handle)

    def get_camera_image(self, cam_handle):
        print("--Получить кадр из симуляции Coppelia----------------------------------------------------------")
        imgCoppelia, resolution = self.sim.getVisionSensorImg(cam_handle, 0b00000000)
        if self.isVerbose:
            print("Тип данных:", type(imgCoppelia))
            print("Разрешение: {}".format(resolution))
            print(f"Кол-во значений (H*W*Col) = {len(imgCoppelia)}")
            print("Содержимое (первые 10 эл-тов):\n", imgCoppelia[:10])
            print()
        imgNumpy = np.frombuffer(imgCoppelia, dtype=np.uint8)
        if self.isVerbose:
            print("Тип данных:", type(imgNumpy))
            print('Размерность: {}'.format(imgNumpy.shape))
            print()
        imgNumpy = imgNumpy.reshape(resolution[1], resolution[0], 3)
        imgNumpy = np.flipud(imgNumpy)
        imgNumpy = cv2.cvtColor(imgNumpy, cv2.COLOR_RGB2BGR)
        if self.isVerbose:
            print("Тип данных:", type(imgNumpy))
            print('Размерность: {}'.format(imgNumpy.shape))
            height = imgNumpy.shape[0]
            width = imgNumpy.shape[1]
            channels = imgNumpy.shape[2]
            print('Height   : {}'.format(height))
            print('Width    : {}'.format(width))
            print('Channels : {}'.format(channels))
            if imgNumpy.shape[0] > 5:
                print("Содержимое (первые 5 эл-тов):\n", imgNumpy[0, :5])
            else:
                print("Содержимое полное:\n", imgNumpy)
        return imgNumpy

    def classifyFrontVertSing(self):
        image_Coppelia_NumpyCV = self.get_camera_image(self.rightVerticalSing_cam_handle)
        image_preproc_Tensor = self.preprocess_image(image_Coppelia_NumpyCV, newH=28, newW=28)
        answerDict = self.recognize_shape(self.modelNN_loaded, image_preproc_Tensor)

        image_out_NumpyCV = image_Coppelia_NumpyCV.copy()
        cv2.putText(img=image_out_NumpyCV, text=answerDict['labelName'],
                    org=(10, 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

        cv2.imshow(self.cvWindow_Front_1, image_Coppelia_NumpyCV)
        cv2.imshow(self.cvWindow_Front_2, image_Coppelia_NumpyCV)
        cv2.imshow(self.cvWindow_Front_3, image_out_NumpyCV)

    def preprocess_image(self, image, newH, newW):
        print("\n--Предобработка изображения----------------------------------------------------------")
        if self.isVerbose: print('Тип данных:  ', type(image))
        if self.isVerbose: print(f'Размерность:  {image.shape}')

        image = cv2.resize(image, (newH, newW))
        if self.isVerbose: print(f'Размерность:  {image.shape}')

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.isVerbose: print(f'Размерность:  {image.shape}')
        image = np.expand_dims(image, axis=-1)
        if self.isVerbose: print(f'Размерность:  {image.shape}')

        if self.isVerbose: print("Нормализация. ДО.    Макс. зн.:", np.max(image))
        image = image.astype('float32') / 255.0
        if self.isVerbose: print("Нормализация. ПОСЛЕ. Макс. зн.:", np.max(image))

        image = tf.expand_dims(image, 0)
        if self.isVerbose: print('Тип данных:  ', type(image))
        if self.isVerbose: print(f'Размерность:  {image.shape}')

        return image

    def get_average_brightness(self, image):
        if self.isVerbose: print(f'Размерность:  {image.shape}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.isVerbose: print(f'Размерность:  {image.shape}')
        return np.mean(image)/255.0

    def recognize_shape(self, modelNN, image):
        print("\n--Классификация нейронкой----------------------------------------------------------")
        y_predictions = modelNN.predict(image)
        y_predicted_classNum = np.argmax(y_predictions)
        y_predicted_label = self.class_labels[y_predicted_classNum]

        return {"probabilityAllClass": y_predictions,
                "classNumber": y_predicted_classNum,
                "labelName": y_predicted_label}

    def startMainFunc(self, timeToWork=10):
        self.sim.startSimulation()
        print(f"Запуск симуляции на {timeToWork} сек. ")

        while (t := self.sim.getSimulationTime()) < timeToWork and not self.sim.getSimulationStopping():
            self.detect_yellow_mark()

            if self.yellow_mark_detected:
                print("Пауза для чтения знака дорожного движения.")
                self.stop_vehicle()
                self.classifyFrontVertSing()  # Чтение знака дорожного движения
                self.yellow_mark_detected = False  # Сброс флага после чтения знака
                self.resume_vehicle()  # Продолжить движение после чтения знака
            else:
                dir=self.CallDirectionLineFollower()
                self.speed_velocity_for_dir(dir)

            k = cv2.waitKey(10)
            if k == ord('q') & 0xFF:
                break

            print(f'Simulation time: {t:.2f} [s]')
            self.sim.step()

        self.sim.stopSimulation()
        print("Остановка симуляции.")

    # def moveStraight(self):
    #     self.sim.setJointTargetVelocity(self.left_motor_handle, self.max_velocity)
    #     self.sim.setJointTargetVelocity(self.right_motor_handle, self.max_velocity)

    # def follow_line(self):
    #     print("\n--Следование по линии----------------------------------------------------------")
    #     image = self.get_camera_image(self.rightFloorSing_cam_handle)
    #     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #
    #     lower_white = np.array([0, 0, 200])
    #     upper_white = np.array([180, 30, 255])
    #
    #     mask = cv2.inRange(hsv, lower_white, upper_white)
    #     edges = cv2.Canny(mask, 50, 150)
    #     lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    #
    #     if lines is not None:
    #         for line in lines:
    #             x1, y1, x2, y2 = line[0]
    #             cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #
    #         avg_x = np.mean([(x1 + x2) / 2 for x1, y1, x2, y2 in lines[:, 0]])
    #         center_x = image.shape[1] / 2
    #         deviation = center_x - avg_x
    #
    #         kp = 0.1
    #         velocity_adjustment = kp * deviation
    #
    #         left_velocity = self.max_velocity - velocity_adjustment
    #         right_velocity = self.max_velocity + velocity_adjustment
    #
    #         self.sim.setJointTargetVelocity(self.left_motor_handle, left_velocity)
    #         self.sim.setJointTargetVelocity(self.right_motor_handle, right_velocity)
    #     else:
    #         print("Линии не найдены. Продолжение движения прямо.")
    #         self.moveStraight()
    def CallDirectionLineFollower(self):
        '''Движение по трассе. По линии'''
        # Получение изображений с камер
        left_image_NumpyCV = self.get_camera_image(self.left_cam_handle)
        right_image_NumpyCV = self.get_camera_image(self.right_cam_handle)

        # Обработка изображений
        left_brightness = self.get_average_brightness(left_image_NumpyCV)
        right_brightness = self.get_average_brightness(right_image_NumpyCV)

        # Вычисление ошибки управления
        error = 0 - (right_brightness - left_brightness)
        print("LineFollower error=", error)
        Kp=75
        direction = Kp * error
        return direction
        # Вычисление скоростей колес для движения по линии
    def speed_velocity_for_dir(self,direction):
        if direction>100:
            direction=100
        elif direction<-100:
            direction=-100
        koeff=abs(direction)*(-0.013)+1.0

        if direction == 0:
            self.sim.setJointTargetVelocity(self.left_motor_handle, self.max_velocity)
            self.sim.setJointTargetVelocity(self.right_motor_handle, self.max_velocity)
        elif direction>0:
            self.sim.setJointTargetVelocity(self.left_motor_handle, self.max_velocity)
            self.sim.setJointTargetVelocity(self.right_motor_handle, self.max_velocity*koeff)
        elif direction < 0:
            self.sim.setJointTargetVelocity(self.left_motor_handle,self.max_velocity * koeff)
            self.sim.setJointTargetVelocity(self.right_motor_handle, self.max_velocity)

    def detect_yellow_mark(self):
        print("\n--Распознавание жёлтой метки на дороге----------------------------------------------------------")
        image = self.get_camera_image(self.rightFloorSing_cam_handle)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        result = cv2.bitwise_and(image, image, mask=mask)

        cv2.imshow('Original Floor Image', image)
        cv2.imshow('Yellow Mark Detection', result)

        yellow_percentage = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
        print(f"Процент жёлтого на изображении: {yellow_percentage * 100:.2f}%")

        if yellow_percentage > 0.05:
            print("Жёлтая метка обнаружена на дороге!")
            self.yellow_mark_detected = True
        else:
            print("Жёлтая метка не обнаружена.")
            self.yellow_mark_detected = False

    def stop_vehicle(self):
        self.sim.setJointTargetVelocity(self.left_motor_handle, 0)
        self.sim.setJointTargetVelocity(self.right_motor_handle, 0)

    def resume_vehicle(self):
        self.sim.setJointTargetVelocity(self.left_motor_handle, self.max_velocity)
        self.sim.setJointTargetVelocity(self.right_motor_handle, self.max_velocity)

def main():
    myCtrl = Controller()
    myCtrl.startMainFunc(timeToWork=100)

if __name__ == "__main__":
    main()
