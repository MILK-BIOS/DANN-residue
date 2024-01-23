import pandas as pd

# 加载数据集
dataset = pd.read_csv("D:/学习/代码/DANN/dataset/20samples.csv")  # 请替换为你的数据集文件路径
print(dataset.columns)
# 检查标签列是否包含 NaN
nan_labels = dataset['product'].isna().sum()
print(f'Number of NaN values in the label column: {nan_labels}')

# 查找包含缺失值的行
rows_with_missing_values = dataset[dataset.isnull().any(axis=1)]

# 打印包含缺失值的行
print(rows_with_missing_values)

# 查找包含缺失值的行的索引
indexes_with_missing_values = rows_with_missing_values.index

# 打印包含缺失值的行的索引
print(indexes_with_missing_values)

# 删除包含缺失值的行
dataset = dataset.dropna()

# 保存修改后的数据集到新的 CSV 文件
dataset.to_csv('D:/学习/代码/DANN/dataset/20samples.csv', index=False)


