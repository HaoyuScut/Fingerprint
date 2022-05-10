from PyQt5.QtWidgets import QTableWidgetItem,QFileDialog
from PyQt5.QtCore import *
from finger_add import *
import cv2
import enhance
import thinning
import feature1
from PIL import Image,ImageQt
from pylab import *


global img

class MainWindow(Ui_MainWindow,QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        self.setupUi(self)
        #表格显示设置
        self.tb_endpoint.setColumnCount(3)
        self.tb_crosspoint.setColumnCount(5)
        self.tb_endpoint.setHorizontalHeaderLabels(['特征点种类','特征点坐标','斜率（rad）'])
        self.tb_crosspoint.setHorizontalHeaderLabels(['特征点种类', '特征点坐标', '斜率1（rad）','斜率2（rad）','斜率3（rad）'])
        # self.tb_endpoint.resizeRowsToContents()
        # self.tb_endpoint.resizeColumnsToContents()
        # self.tb_crosspoint.resizeRowsToContents()
        # self.tb_crosspoint.resizeColumnsToContents()

    def select(self):
        global img
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.tif;;*.png;;All Files(*)")
        if imgName:
            print(imgName)
            img = cv2.imdecode(np.fromfile(imgName, dtype=np.uint8), cv2.IMREAD_COLOR)
            print(type(img))
            print('origin_type:',img.dtype)
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

        # print(type(img))
        # print('img.shape', img.shape)
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 转灰度图
            # cv2.imshow("original image", img)
            # cv2.waitKey(0)

        img = enhance.image_enhance(img)
        # print(img.shape)
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
        img,features_endpoint,features_crosspoint = feature1.image_feature(img)

        img_feature = Image.fromarray(uint8(img))

        img_feature = ImageQt.toqpixmap(img_feature).scaled(self.pic_feature.width(), self.pic_feature.height())
        self.pic_feature.setPixmap(img_feature)

        # 使用lineedit显示数据
        # features_endpoint = str(features_endpoint)
        # features_crosspoint = str(features_crosspoint)
        # for m in range(len(features_endpoint)):
        #     self.le_endpoint.append(str(features_endpoint[m]))
        #
        #
        # for m in range(len(features_crosspoint)):
        #     self.le_crosspoint.append(str(features_crosspoint[m]))

        # 使用tablewidget显示数据
        endpoint = np.array(features_endpoint)
        crosspoint = np.array(features_crosspoint)

        #创建表头
        tbend_row_num = endpoint.shape[0]
        tbcross_row_num = crosspoint.shape[0]
        tbend_row = []
        tbcross_row = []
        for i in range(1,tbend_row_num):
            tbend_row.append(str(i))
        for j in range(1, tbcross_row_num):
            tbcross_row.append(str(j))
        print(tbcross_row)
        print(tbend_row)

        self.tb_endpoint.setRowCount(tbend_row_num)
        self.tb_crosspoint.setRowCount(tbcross_row_num)

        self.tb_endpoint.setVerticalHeaderLabels(tbend_row)
        self.tb_crosspoint.setVerticalHeaderLabels(tbcross_row)

        end_final = []
        cross_final = []
        for m in range(0,tbend_row_num):
            end_final.append((str(endpoint[m][0]),str(endpoint[m][1]),str(endpoint[m][2])))
        for m in range(0,tbcross_row_num):
            cross_final.append((str(crosspoint[m][0]),str(crosspoint[m][1]),str(crosspoint[m][2][0]),str(crosspoint[m][2][1]),str(crosspoint[m][2][2])))


        print(end_final)
        print(cross_final)
        row = 0
        for tup in end_final:
            col = 0
            for item in tup:
                oneitem = QTableWidgetItem(item)
                self.tb_endpoint.setItem(row, col, oneitem)
                oneitem.setTextAlignment(Qt.AlignCenter|Qt.AlignCenter)
                col += 1
            row += 1

        row = 0
        for tup in cross_final:
            col = 0
            for item in tup:
                oneitem = QTableWidgetItem(item)
                self.tb_crosspoint.setItem(row, col, oneitem)
                oneitem.setTextAlignment(Qt.AlignCenter | Qt.AlignCenter)
                col += 1
            row += 1


if __name__ =='__main__':
    app = QtWidgets.QApplication(sys.argv)
    mywindow = MainWindow()
    mywindow.show()
    sys.exit(app.exec_())
