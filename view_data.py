import zarr

# 打开 Zarr 数组（假设数据集在当前目录下）
z = zarr.open('./dataset/data/eepose', mode='r')

# 读取整个数组
data = z[:]
pass
# # 或者读取部分数据（支持切片）
# subset = z[0:100, 50:200]  # 例如读取前100行，第50到199列

# print("Shape:", z.shape)
# print("Chunks:", z.chunks)
# print("Dtype:", z.dtype)
# print("Data:", data)