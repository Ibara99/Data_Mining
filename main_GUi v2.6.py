import sys, csv
from gui5 import *
from PySide.QtGui import *
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import numpy as np

class progress(QWidget):
    def __init__ (self):
        super(progress, self).__init__()
        
        self.dialog = QDialog()
        self.dialog.resize(361, 23)
        layout = QHBoxLayout(self.dialog)
        
        self.progressBar = QProgressBar()
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.progressBar.minimum = 1
        self.progressBar.maximum = 100

        #self.cancelBtn = QPushButton("Cancel")
        #self.cancelBtn.clicked.connect(self.cancelProgress)

        layout.addWidget(self.progressBar)#, 0,0, 1,1)
        #layout.addWidget(self.cancelBtn)
        

        #self
        self.dialog.setWindowTitle("Loading . . . ")
        self.dialog.show()
    def cancelProgress(self):
        self.dialog.close()
        QMessageBox.warning(self, "Warning!", "Proses canceled")
    def updateProgres(self, value):
        self.progressBar.setValue(value)
    def closeProgres(self):
        self.dialog.close()
class MyForm(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.ui = Ui_Dialog() #memnggil import
        self.ui.setupUi(self) #inisiasi gui
        self.data=[[]]
        #Tab 1
        self.ui.pushButton.clicked.connect(self.openFile)
        self.ui.pushButton_4.clicked.connect(self.okTab0)

        #Tab 2
        self.ui.radioMissingValue.clicked.connect(self.disableLineEdit)
        self.ui.radioNormal.clicked.connect(self.disableLineEdit)
        self.ui.radioDummy.clicked.connect(self.disableLineEdit)
        self.ui.radioOutlier.clicked.connect(self.EnableLineEdit)
        self.ui.ButtonOkPreproses.clicked.connect(self.preprosesOK)

        #Tab 3
        self.ui.pushButton_2.clicked.connect(self.klasifikasi)
        self.ui.pushButton_5.clicked.connect(self.test)
        self.ui.K_foldingCheck.clicked.connect(self.disSpin)
        #self.head=[]
        
        self.ui.pushButton_3.clicked.connect(self.pca)
    def test(self):
        var = []
        for x in self.dataPakai[0]:
            value, ok = QInputDialog.getText(self, "Masukkan Parameter", x)
            var.append(value)
            if not ok : break
        self.widget = QWidget()
        self.ui.tableWidget_3.clear()#
    def is_numeric(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    def coba(self):
        print("bener?")
    def openFile(self):
        
        #Pilih file
        fileName, ok= QFileDialog.getOpenFileName(self, "Open Text Files",
                                                 "c:/", "Open files(*.txt *.csv *.data)")# ;; CSV(*.csv)")
        if ok: separator, ok = QInputDialog.getText(self, "Input separator per coloum", "Separator : ")

        if ok:
            self.dataAwal = []
            #Cek tipe file, txt atau csv?
            self.ui.lineEdit.setText(fileName)
            tipe = fileName.split(".")
            print(fileName)
            
            #Ini unutk open
            if tipe[-1] == "csv":
                with open(fileName, newline='') as csvfile:
                    spamreader = csv.reader(csvfile, delimiter='-', quotechar='"')
                    for row in spamreader:
                        self.dataAwal.append(row[0].split(separator))
                        
                csvfile.close()
            elif tipe[-1] == "txt" :
                a = progress()
                
                file=open(fileName)            
                l=file.read()
                l=l.split('\n')
                for x in range(len(l)):
                    l[x]=l[x].split(separator)

                    a.updateProgres(int(x/len(l) *100))
                a.closeProgres()
                file.close()
                self.dataAwal=l
            elif tipe[-1] == "data":
                a = progress()
                
                file=open(fileName)
                l=file.read()
                l=l.split('\n')
                for x in range(len(l)):
                    l[x]=l[x].split(separator)
                    a.updateProgres(int(x/len(l) *100))
                    if x<5:print(l[x])
                a.closeProgres()
                file.close()
                self.dataAwal=l

            while self.dataAwal[-1] == [""]: self.dataAwal.pop(-1)
            #print(self.dataAwal[-10:])

            #self.head=self.dataAwal.pop(0)

            #Set combo box di sini
            self.ui.comboBox.clear();self.ui.comboBox_2.clear()
            for x in self.dataAwal[0]:
                self.ui.comboBox.addItem(x)
                self.ui.comboBox_2.addItem(x)
            self.ui.comboBox.addItem("NONE")
            self.ui.comboBox_2.addItem("NONE")

        
        
    def okTab0(self):
        try:
            #Memisahkan antara kolom id, kolom kelas, dan kolom data
            indexId = self.ui.comboBox.currentIndex()
            indexClass = self.ui.comboBox_2.currentIndex()
            self.id=[]; self.target=[]; self.dataPakai=[]
            for baris in self.dataAwal:
                temp=[]
                for kolom in range (len(baris)):
                    if kolom == indexId:
                        self.id.append(baris[indexId])
                    elif kolom == indexClass:
                        self.target.append(baris[indexClass])
                    else:
                        temp.append(baris[kolom])
                self.dataPakai.append(temp)
            #print(self.dataPakai[0])

            #pengecekan tipe data
            eror = False
            if self.ui.radioButton.isChecked(): #Numeric
                for i in range (len(self.dataPakai[1])):
                    #print(self.dataPakai[1][i])
                    if not self.is_numeric(self.dataPakai[1][i]) and self.dataPakai[1][i] !="" and self.dataPakai[1][i] !=" " :
                        eror=True
                    if eror : break
            elif self.ui.radioButton_2.isChecked():   #Ordinal
                for i in range (len(self.dataPakai[1])):
                    if self.is_numeric(self.dataPakai[1][i]):
                        eror=True
                    if eror : break
            #Jika data benar, maka menampilkan ke tabel
            if not eror:
                if self.target==[]:
                    self.ui.tableWidget.setColumnCount(len(self.dataPakai[1]))
                    self.ui.tableWidget.setHorizontalHeaderLabels(self.dataPakai[0])
                else:
                    self.ui.tableWidget.setColumnCount(len(self.dataPakai[1])+1)
                    self.ui.tableWidget.setHorizontalHeaderLabels(self.dataPakai[0]+["Class"])

                self.ui.tableWidget.setRowCount(len(self.dataPakai)-1)
                
                if self.id != []:
                    self.ui.tableWidget.setVerticalHeaderLabels(self.id[1:])
                #print(self.target[0])
                prog = progress()
                #prog.updateProgres(int(x/len(self.dataPakai)*100))
                #prog.closeProgres()
                for x in range (len(self.dataPakai)-1):
                    prog.updateProgres(int(x/len(self.dataPakai)*100))
                    for y in range(len(self.dataPakai[x])): 
                        self.ui.tableWidget.setItem(x,y, QtGui.QTableWidgetItem(self.dataPakai[x+1][y]))
                    if self.target != []:
                        self.ui.tableWidget.setItem(x,len(self.dataPakai[x]), QtGui.QTableWidgetItem(self.target[x+1]))
                prog.closeProgres()
                self.ui.spinBox.setMaximum(len(self.dataPakai[0])-1)
                self.ui.spinBox.setMinimum(1)

                for x in self.ui.semuaTable[1:]:
                    x.setHorizontalHeaderLabels(["" for i in range(1000)])
                    x.setVerticalHeaderLabels(["" for i in range(1000)])
                    x.setColumnCount(0)
                    x.setRowCount(0)
                
            else:
                if self.ui.radioButton.isChecked():
                    userInfo = QMessageBox.warning(self, "Warning!",
                            "Tipe data bukan Numeric")
                else:
                    userInfo = QMessageBox.warning(self, "Warning!",
                            "Tipe data bukan Ordinal")
        except AttributeError:
            userInfo = QMessageBox.warning(self, "Warning!",
                            " Terjadi kesalahan input data")
        except :
                userInfo = QMessageBox.warning(self, "Warning!",
                                str(sys.exc_info()[1]))
    def disSpin(self):
        self.ui.spinBox_2.setEnabled(self.ui.K_foldingCheck.isChecked())
    def disableLineEdit(self):
        self.ui.lineEdit_r.setDisabled(True)
        self.ui.lineEdit_Phi.setDisabled(True)
    def EnableLineEdit(self):
        self.ui.lineEdit_r.setDisabled(False)
        self.ui.lineEdit_Phi.setDisabled(False)
    
    def preprosesOK(self):
        try:
            if self.ui.radioMissingValue.isChecked():
                prog = progress()
                #prog.updateProgres(int(baris/len(data)*100))
                #prog.closeProgres()
                for baris in range(len(self.dataPakai)):
                    prog.updateProgres(int(baris/len(self.dataPakai)*100))
                    for kolom in range(len(self.dataPakai[baris])):
                        if self.dataPakai[baris][kolom] == "" \
                           or self.dataPakai[baris][kolom] == " ":
                            if self.target == []: adaKelas = False
                            else:   adaKelas = True
                            self.dataPakai[baris][kolom] = self.missingValue(baris, kolom,
                                                                     self.dataPakai, adaKelas)
                prog.closeProgres()
                #return True
            elif self.ui.radioNormal.isChecked():
                prog = progress()
                #baru dinormalisasi
                header = self.dataPakai[0]
                data = preprocessing.normalize(data)
                tempBaris=[]
                for baris in data:
                    prog.updateProgres(int(baris/len(data)*100))
                    tempKolom=[]
                    for kolom in baris:
                        tempKolom.append(str(kolom))
                    tempBaris.append(tempKolom)
                self.dataPakai = [header] + tempBaris
                #print(self.dataPakai[:10])
                prog.closeProgres()
            elif self.ui.radioOutlier.isChecked():
                #Kalau outlier, jalankan ini ini
                if self.is_numeric(self.ui.lineEdit_r.text()) \
                   and self.is_numeric(self.ui.lineEdit_Phi.text()) :
                    self.outlier = self.outlierDetect(self.dataPakai,
                                                 float(self.ui.lineEdit_r.text()),
                                                 float(self.ui.lineEdit_Phi.text()))

                    self.showOutlier()
                    userInfo = QMessageBox.question(self, "Confirmation",
                        "Terdapat "+str(self.outlier.count("Outlier"))+" Outlier pada data. \n Ingin menghapusnya?",
                        QMessageBox.Yes |QMessageBox.No)
                    if userInfo == QMessageBox.Yes:
                        for baris in range(len(self.outlier)-1, -1, -1):
                            if self.outlier[baris] == "Outlier":
                                self.dataPakai.pop(baris+1)
                                if self.id != []: self.id.pop(baris+1)
                                self.target.pop(baris+1)
                else:
                    userInfo = QMessageBox.warning(self, "Warning!",
                            "Tolong inputkan r dan phi dengan benar!")
            elif  self.ui.radioDummy.isChecked():
                prog = progress()
                
                lbl = preprocessing.LabelEncoder()
                hot = preprocessing.OneHotEncoder(sparse=False)
                pisah = []
                pisah = [[] for i in range(len(self.dataPakai[0]))]
                for baris in range(len(self.dataPakai)):
                    for fitur in range(len(self.dataPakai[baris])):
                        pisah[fitur].append(self.dataPakai[baris][fitur])

                            
                tmp = [[] for i in range(len(pisah[0]))]
                for x in range(len(pisah)):
                    print(x)
                    if self.is_numeric(pisah[x][1]):
                        for y in range(len(pisah[x])):
                            tmp[y].append(pisah[x][y])
                    else:
                        a = lbl.fit_transform(pisah[x][1:])
                        #print("b")
                        b = hot.fit_transform(a.reshape(-1,1))
                        #print("c")
                        c = np.append(lbl.classes_, b).reshape(len(pisah[x]), len(lbl.classes_))
                        #print("a")
                        for y in range(len(c)):
                            for z in c[y]:
                                tmp[y].append(z)
                self.dataPakai = tmp
                "Encodernya ngga ketemu :'( "
                prog.closeProgres()
            prog = progress()
                
            #Sama kayak yang load data
            if self.target==[]:
                self.ui.tableWidget_2.setColumnCount(len(self.dataPakai[1]))
                self.ui.tableWidget_2.setHorizontalHeaderLabels(self.dataPakai[0])
            else:
                self.ui.tableWidget_2.setColumnCount(len(self.dataPakai[1])+1)
                self.ui.tableWidget_2.setHorizontalHeaderLabels(self.dataPakai[0]+["Class"])

            self.ui.tableWidget_2.setRowCount(len(self.dataPakai)-1)
            
            if self.id != []:
                self.ui.tableWidget_2.setVerticalHeaderLabels(self.id[1:])
            #print(self.target[0])
            for x in range (len(self.dataPakai)-1):
                prog.updateProgres(int(x/len(self.dataPakai)*100))
                for y in range(len(self.dataPakai[x])): 
                    self.ui.tableWidget_2.setItem(x,y, QtGui.QTableWidgetItem(self.dataPakai[x+1][y]))
                if self.target != []:
                    self.ui.tableWidget_2.setItem(x,len(self.dataPakai[x]), QtGui.QTableWidgetItem(self.target[x+1]))
            
            prog.closeProgres()

            userInfo = QMessageBox.information(self, "Info!",
                            "Data berhasil diproses :)")
        except ValueError:
            userInfo = QMessageBox.warning(self, "Warning!",
                            "Terdapat Missing Value")
        except AttributeError:
            userInfo = QMessageBox.warning(self, "Warning!",
                            "Tolong Inputkan Data Terlebih Dahulu")
        except :
                userInfo = QMessageBox.warning(self, "Warning!",
                                str(sys.exc_info()[1]))
    
    def showOutlier(self):
        self.widget = QWidget()
        self.widget.setWindowTitle("Outlier")
        self.widget.move(50, 50)
        tabel = QTableWidget(self.widget)

        
        #Sama kayak yang load data
        if self.target==[]:
            tabel.setColumnCount(len(self.dataPakai[1]))
            tabel.setHorizontalHeaderLabels(self.dataPakai[0])
        else:
            tabel.setColumnCount(len(self.dataPakai[1])+1)
            tabel.setHorizontalHeaderLabels(self.dataPakai[0]+["Class"])

        tabel.setRowCount(self.outlier.count("Outlier"))
        count = 0; id_tampung=[]
        for x in range (len(self.dataPakai)-1):
            
            if self.outlier[x] == "Outlier":
                #print(self.outlier[x])
                for y in range(len(self.dataPakai[x])):
                        tabel.setItem(count,y, QtGui.QTableWidgetItem(self.dataPakai[x+1][y]))
                if self.target != []:
                    tabel.setItem(count,len(self.dataPakai[x]), QtGui.QTableWidgetItem(self.target[x+1]))
                if self.id !=[]:
                    id_tampung.append(self.id[x+1])
                count += 1
        if self.id != []:
            tabel.setVerticalHeaderLabels(id_tampung)
            
        layout = QVBoxLayout()
        layout.addWidget(tabel)
        self.widget.setLayout(layout)

        self.widget.show()
    def missingValue(self, BarisIndex, AttrIndex, data, Kelas):
        total=0
        count=0
        if Kelas:
            for baris in range(len(data)):
                if baris != BarisIndex \
                   and self.target[baris] == self.target[BarisIndex]:
                    #print(baris, BarisIndex)
                    #print(self.target[baris][-1], self.target[BarisIndex][-1])
                    total += float(data[baris][AttrIndex])
                    count += 1
        else:
            for baris in range(len(data)):
                prog.updateProgres(int(baris/len(data)*100))
                total += data[baris][AttrIndex]
                count += 1
        Rata_rata = total / count
        return "%.3f"%Rata_rata
                    
    def dist(self, a,b, file):
        "a, b is row"
        dis = 0
        for x in range (len(file[0])):
            temp = float(file[a][x]) - float(file[b][x])
            if temp<0:
                temp*=-1
            dis += temp
        return dis

    def outlierDetect(self, file, r, k):
        outlier = []
        prog = progress()
        #prog.updateProgres(int(baris/len(data)*100))
        #prog.closeProgres()
        for i in range(1, len (file)):      #file[0] is header
            count = 0
            for j in range (1, len(file)):  #Dicari distance per baris,
                prog.updateProgres(int(i/len(file)*100))
                if i != j and self.dist(i,j, file)<r:  #yg kurang dari r
                    count += 1
                    if count / (len(file)-1) >= k : #k adalah phi
                                                    #Klo jumlahnya lebih dari k
                        outlier += ['Not']              #Bukan Outliers
                        break
            if count / (len(file)-1)<k: outlier += ['Outlier'] #Sebaliknya
        prog.closeProgres()
        return outlier
    def confusion(self):
        self.widget = QWidget()
        self.widget.setWindowTitle("Confusion Matrik")
        self.widget.move(50, 50)
        data = np.array(self.dataPakai[1:], dtype=float)
        tabel = QTableWidget(self.widget)
        kelas = GaussianNB()
        kelas = kelas.fit(data, self.target[1:])
        #print(kelas.classes_)
        self.matrik = confusion_matrix(self.y_test, self.y_pred, kelas.classes_)# self.clf.classes_)
        
        #Sama kayak yang load data
        if self.target==[]:
            userInfo = QMessageBox.warning(self, "Warning!",
                            "Data tidak memiliki Class")
        else:
            tabel.setColumnCount(len(self.matrik))
            tabel.setHorizontalHeaderLabels(kelas.classes_)

        tabel.setRowCount(len(self.matrik))
        tabel.setVerticalHeaderLabels(kelas.classes_)
        
        #count = 0; id_tampung=[]
        for x in range (len(self.matrik)):
            for y in range(len(self.matrik[x])):
                print(self.matrik[x][y], end=' ')
                tabel.setItem(x,y, QtGui.QTableWidgetItem(str(self.matrik[x][y])))
            print()
        layout = QVBoxLayout()
        layout.addWidget(tabel)
        self.widget.setLayout(layout)
        self.widget.show()
        
    def klasifikasi(self):
        try:
            prog = progress()
            #print(self.ui.radioButton_5.isChecked(), self.ui.radioButton_3.isChecked())
            '''
            pisah_fitur = [ [] for i in range(len(self.dataPakai[0]))]
            for baris in range(len(self.dataPakai)):
                for fitur in range(len(self.dataPakai[baris])):
                    pisah_fitur[fitur].append(self.dataPakai[baris][fitur])
                    
            encode = preprocessing.LabelEncoder()
            temp=[]
            for x in range(len(pisah_fitur)):
                if self.is_numeric(pisah_fitur[x][1]):
                    temp.append(pisah_fitur[x][1:])
                else:
                    a = np.array([pisah_fitur[x][0]])
                    temp.append(encode.fit_transform(pisah_fitur[x][1:]))

            data = [ [] for i in range(len(temp[0]))]
            for baris in range(len(temp)):
                for fitur in range(len(temp[baris])):
                    data[fitur].append(temp[baris][fitur])
            data = np.array(data, dtype=float)
            self.sample = data
            '''
            data = np.array(self.dataPakai[1:], dtype=float)
            target = self.target[1:]

            if self.ui.K_foldingCheck.isChecked():
                kf = KFold(n_splits=3, shuffle=False)
                self.KFolding_index = []
                for i in kf.split(data):
                    self.KFolding_index.append(i)

                self.x_train=[]; self.y_train=[]
                for i in self.KFolding_index[self.ui.spinBox_2.value()][0]:
                    self.x_train.append(data[i])
                    self.y_train.append(target[i])
                    
                self.x_test = []; self.y_test=[]
                for i in self.KFolding_index[self.ui.spinBox_2.value()][1]:
                    self.x_test.append(data[i])
                    self.y_test.append(target[i])
                
            else:
                self.x_train, self.x_test = data, data
                self.y_train, self.y_test = target, target
            #print(len(self.y_train), len(self.x_train))
            #print(len(self.y_test), len(self.x_test))

            if self.ui.naiveRadio.isChecked():
                self.clf = GaussianNB()
            elif self.ui.treeRadio.isChecked():
                self.clf = tree.DecisionTreeClassifier()
            elif self.ui.backRadio.isChecked():
                self.clf = MLPClassifier([100,50], solver='lbfgs', max_iter=500, alpha= 0.0001)
            elif self.ui.normalRadio.isChecked():
                self.clf = LogisticRegression(solver="saga", multi_class = "multinomial")
            
            #if self.ui.K_foldingCheck.isChecked():
            #len(self.x_train
            print(self.x_train[0])
            print(len(self.x_train), len( self.y_train))
            #GaussianNB.fit(
            self.y_pred = self.clf.fit(self.x_train, self.y_train)
            self.y_pred = self.y_pred.predict(np.array(self.x_test))
            

            #akurasi = (target == self.y_pred).sum() / len(target) *100
            akurasi = self.clf.fit(self.x_train, self.y_train).score(self.x_test, self.y_test) *100
            self.ui.label_5.setText(str(akurasi)+"%")

            #Sama kayak yang load data
            if self.target==[]:
                raise ValueError("Data tidak memiliki Kelas")
            else:
                self.ui.tableWidget_3.setColumnCount(len(self.x_test[0])+2)
                self.ui.tableWidget_3.setHorizontalHeaderLabels(self.dataPakai[0]+["Class"]+["Prediction"])

            self.ui.tableWidget_3.setRowCount(len(self.x_test))
            
            #if self.id != []:
             #   self.ui.tableWidget_3.setVerticalHeaderLabels(self.id[1:])
            #print(self.target[0])
             
            for x in range (len(self.x_test)):
                for y in range(len(self.x_test[x])):
                    prog.updateProgres(int(x/len(self.x_test)*100))
                    self.ui.tableWidget_3.setItem(x,y, QtGui.QTableWidgetItem(str(self.x_test[x][y])))
                if self.target != []:
                    self.ui.tableWidget_3.setItem(x,len(self.dataPakai[x]), QtGui.QTableWidgetItem(str(self.y_test[x])))
                self.ui.tableWidget_3.setItem(x,len(self.dataPakai[x])+1, QtGui.QTableWidgetItem(str(self.y_pred[x])))

            self.confusion()
            prog.closeProgres()
        except AttributeError:
            userInfo = QMessageBox.warning(self, "Warning!",
                            "Tolong Inputkan Data Terlebih Dahulu")
        #except :
         #       userInfo = QMessageBox.warning(self, "Warning!",
          #                      str(sys.exc_info()[1]))

    def pca(self):
        try:
            prog = progress()
            data = np.array(self.dataPakai[1:], float)
            target = np.array(self.target[1:])

            pca = PCA(n_components=self.ui.spinBox.value())
            pca.fit(data)
            data = pca.transform(data)
            header = [[]]
            for x in range(self.ui.spinBox.value()):
                header[0].append(str(x+1))
            

            tempBaris=[]
            for baris in data:
                tempKolom=[]
                for kolom in baris:
                    tempKolom.append(str(kolom))
                tempBaris.append(tempKolom)
            self.dataPakai = header + tempBaris
            #print(header)
            #print(self.dataPakai[:5])
            if self.target==[]:
                raise ValueError("Data tidak memiliki Kelas")
            else:
                self.ui.tableWidget_4.setColumnCount(self.ui.spinBox.value()+1)
                self.ui.tableWidget_4.setHorizontalHeaderLabels(self.dataPakai[0]+["Class"])

            self.ui.tableWidget_4.setRowCount(len(data))
            
            if self.id != []:
                self.ui.tableWidget_4.setVerticalHeaderLabels(self.id[1:])
            #print(data[0][0])
            for x in range (len(data)-1):
                prog.updateProgres(int(x/len(data)*100))
                for y in range(len(data[x])): 
                    self.ui.tableWidget_4.setItem(x,y, QtGui.QTableWidgetItem(str(data[x][y])))
                if self.target != []:
                    self.ui.tableWidget_4.setItem(x,len(data[x]), QtGui.QTableWidgetItem(self.target[x+1]))

            if len(self.dataPakai[0]) > 1:
                self.ui.spinBox.setMaximum(len(self.dataPakai[0])-1)
            prog.closeProgres()
        except ValueError:
                userInfo = QMessageBox.warning(self, "Warning!",
                                "Tidak bisa memproses data selain numeric")
        except :
                userInfo = QMessageBox.warning(self, "Warning!",
                                str(sys.exc_info()[1]))
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myapp = MyForm()
    myapp.show()
    sys.exit(app.exec_()) 








































































































'''
iris = datasets.load_iris()
X = iris.data
y = iris.target

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)
'''
