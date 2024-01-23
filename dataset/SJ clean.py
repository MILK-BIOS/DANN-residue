import pandas as pd


# 读取数据
def read_data(file_path):
    df = pd.read_csv(file_path)  # 假设数据是以 CSV 格式存储的，你可以根据实际情况选择其他格式
    return df


# 数据清理
def data_cleaning(df):
    # 根据常识去除异常值
    df_filtered_first = df[((df['VHresidue'] < 0) & (df['VVresidue'] < 0) & (df['product'] < 1200))]

    # 计算每一列的均值和标准差
    column_means = df_filtered_first.mean()
    column_stds = df_filtered_first.std()

    # 定义异常值阈值
    upper_thresholds = column_means + 3 * column_stds
    lower_thresholds = column_means - 3 * column_stds

    # 根据阈值去除异常值行
    df_filtered_second = df_filtered_first[
        ~((df_filtered_first > upper_thresholds) | (df_filtered_first < lower_thresholds)).any(axis=1)]

    return df_filtered_second


# 输出清理后的数据
def output_cleaned_data(df_cleaned, output_file_path):
    df_cleaned.to_csv(output_file_path, index=False)  # 假设你想要将清理后的数据以 CSV 格式保存，你可以根据需求选择其他格式


# 例子
file_path = 'lishu_2020_random_points1.csv'  # 请替换为你的数据文件路径
output_file_path = 'lishu_2020_random_points1_cleaned.csv'  # 请替换为你想要保存清理后数据的路径

# 读取数据
data = read_data(file_path)

# 数据清理
cleaned_data = data_cleaning(data)

# 输出清理后的数据
output_cleaned_data(cleaned_data, output_file_path)
