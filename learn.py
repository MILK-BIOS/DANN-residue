from osgeo import gdal
import numpy as np

# 打开TIFF文件
ds = gdal.Open("E:/SJ/soil import remove-22.23/SM-VWC/SMlishu.tif")

# 读取数据
data = ds.ReadAsArray()

# 将-9999和0替换为NaN
data[data == -9999] = np.nan
data[data == 0] = np.nan

# 关闭文件
ds = None

# 创建副本并保存修改后的数据
driver = gdal.GetDriverByName("GTiff")
out_ds = driver.CreateCopy("your_processed_image.tif", ds, strict=0)

# 写入修改后的数据
out_ds.GetRasterBand(1).WriteArray(data)

# 关闭输出文件
out_ds = None



