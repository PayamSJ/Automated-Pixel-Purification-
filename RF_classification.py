from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle 
from sklearn import model_selection
import numpy as np
from osgeo import gdal
import pickle 

### Model Training

#  labels for the classes
def Iris_label(s):
    it={b'grasslands':0, b'Tree':1, b'bare_soil':2, b'residential_roofs':3, b'industrial_roofs':4, b'Water':5}
    return it[s]

path=r"data.txt"
SavePath = r"model.pickle"

n_estimators = 100 # ntrees = 100
max_features = 12  # mtry = 12

data=np.loadtxt(path, dtype=float, delimiter=',', converters={144:Iris_label} )
#  converters={7:Iris_label}

x,y=np.split(data,indices_or_sections=(144,),axis=1)
x=x[:,0:144] # nuber of bands
train_data,test_data,train_label,test_label = model_selection.train_test_split(x,y, random_state=1, train_size=0.7,test_size=0.3)

classifier = RandomForestClassifier(n_estimators=n_estimators, 
                               bootstrap = True,
                               max_features = 12)
classifier.fit(train_data, train_label.ravel())

print("On the given ntrees = {} and mtry = {}, the model is trained.".format(n_estimators, max_features))
print("oob score on the training set",classifier.score(train_data,train_label))
print("oob score on the test set",classifier.score(test_data,test_label))

file = open(SavePath, "wb")
pickle.dump(classifier, file)

file.close()

### Model Prediction

def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName+"Open file failed")
    return dataset

def writeTiff(im_data,im_geotrans,im_proj,path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) 
        dataset.SetProjection(im_proj) 
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset
    
RFpath = r"model.pickle"
Prisma_Path = r"Main_CNMF_Image.tif"
SavePath = r"Export_CNMF.tif"

dataset = readTif(Prisma_Path)
Tif_width = dataset.RasterXSize
Tif_height = dataset.RasterYSize
Tif_geotrans = dataset.GetGeoTransform()
Tif_proj = dataset.GetProjection()
Prisma_data = dataset.ReadAsArray(0,0,Tif_width,Tif_height)

file = open(RFpath, "rb")

rf_model = pickle.load(file)

file.close()

data = np.zeros((Prisma_data.shape[0],Prisma_data.shape[1]*Prisma_data.shape[2]))
for i in range(Prisma_data.shape[0]):
    data[i] = Prisma_data[i].flatten() 
data = data.swapaxes(0,1)

pred = rf_model.predict(data)

pred = pred.reshape(Prisma_data.shape[1],Prisma_data.shape[2])
pred = pred.astype(np.uint8)

writeTiff(pred,Tif_geotrans,Tif_proj,SavePath)
