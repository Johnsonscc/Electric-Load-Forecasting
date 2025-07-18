import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 读取数据集
df = pd.read_csv('../dataset/huihedata_hour_asc.csv')

# 处理缺失值
df.fillna(method='ffill', inplace=True)  # 使用前向填充方法填充缺失值

# 去除逗号并将电力负荷值转换为浮点数
df['Elia Grid Load [MW]'] = df['Elia Grid Load [MW]'].astype(float)

# 将时间列转换为日期时间格式
# 将时间列转换为日期时间格式，并使用 dayfirst 参数
df['Datetime (UTC)'] = pd.to_datetime(df['Datetime (UTC)'], format='%Y-%m-%d %H:%M' )

# 按升序对数据进行排序
df.sort_values('Datetime (UTC)', ascending=True, inplace=True)

# 只保留 Elia Grid Load 列
df = df[['Elia Grid Load [MW]']]

# 数据归一化
scaler = MinMaxScaler()
df['Elia Grid Load [MW]'] = scaler.fit_transform(df[['Elia Grid Load [MW]']])

# 保存处理后的数据集
df.to_csv('../dataset/processed_dataset.csv', index=False)
print('Data Preparation Complete')