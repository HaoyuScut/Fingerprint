from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from finger import *
import cv2
import enhance
import thinning
import feature
from PIL import Image,ImageQt
from pylab import *


global img

class MainWindow(Ui_MainWindow,QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        self.setupUi(self)

    def select(self):
        global img
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.tif;;*.png;;All Files(*)")
        if imgName:
            print(imgName)
            img = cv2.imdecode(np.fromfile(imgName, dtype=np.uint8), cv2.IMREAD_COLOR)
            print(type(img))
            if len(img.shape) > 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度图

            rows, cols = np.shape(img)
            aspect_ratio = np.double(rows) / np.double(cols)

            new_rows = 300  # randomly selected number
            new_cols = new_rows / aspect_ratio

            img = cv2.resize(img, (int(new_cols), int(new_rows)))
            jpg = QtGui.QPixmap(imgName).scaled(self.pic_origin.width(), self.pic_origin.height())
            self.pic_origin.setPixmap(jpg)

    def pil2_pixmap(self,pil_img):
        print("PIL格式转QPixmap格式")
        pixmap = ImageQt.toqpixmap(pil_img)
        return pixmap

    def enhance(self):
        global img
        print(type(img))
        print('img.shape', img.shape)
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 转灰度图
            cv2.imshow("original image", img)
            cv2.waitKey(0)
        img = enhance.image_enhance(img)

        print(img.shape)
        img_enhance = Image.fromarray(uint8(img))

        img_enhance = ImageQt.toqpixmap(img_enhance).scaled(self.pic_enhance.width(), self.pic_enhance.height())
        self.pic_enhance.setPixmap(img_enhance)


    def thinning(self):
        global img
        img = thinning.image_thinning(img)
        img_thinning = Image.fromarray(uint8(img))

        img_thinning = ImageQt.toqpixmap(img_thinning).scaled(self.pic_thinning.width(), self.pic_thinning.height())
        self.pic_thinning.setPixmap(img_thinning)


    def feature(self):
        global img
        img,features_endpoint,features_crosspoint = feature.image_feature(img)

        img_feature = Image.fromarray(uint8(img))

        img_feature = ImageQt.toqpixmap(img_feature).scaled(self.pic_feature.width(), self.pic_feature.height())
        self.pic_feature.setPixmap(img_feature)

        # features_endpoint = str(features_endpoint)
        # features_crosspoint = str(features_crosspoint)

        for m in range(len(features_endpoint)):
            self.le_endpoint.append(str(features_endpoint[m]))


        for m in range(len(features_crosspoint)):
            self.le_crosspoint.append(str(features_crosspoint[m]))




if __name__ =='__main__':
    app = QtWidgets.QApplication(sys.argv)
    mywindow = MainWindow()
    mywindow.show()
    sys.exit(app.exec_())
