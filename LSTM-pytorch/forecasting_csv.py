import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.io.common import file_path_to_url
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
import holidays
import os
import joblib
from datetime import datetime, timedelta

# 设置随机种子确保可复现性
tf.random.set_seed(42)
np.random.seed(42)

file_path='./dataset/huihedata_hour_asc.csv'

# ----------------------
# 1. 数据加载与预处理
# ----------------------
def load_and_preprocess_data(file_path):
    """加载CSV数据并进行预处理"""
    print(f"加载数据: {file_path}")
    # 读取CSV数据
    df = pd.read_csv(file_path, parse_dates=['timestamp'])

    # 确保数据按时间排序
    df = df.sort_values('timestamp')

    # 设置时间索引
    df.set_index('timestamp', inplace=True)

    # 检查是否有重复时间戳
    if df.index.duplicated().any():
        print("发现重复时间戳，进行合并...")
        df = df.groupby(df.index).mean()

    # 确保每小时一个数据点 - 创建完整时间序列
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq='H'
    )
    df = df.reindex(full_range)

    # 处理缺失值 - 线性插值
    print(f"原始数据点: {len(df)}")
    print(f"缺失值数量: {df['power'].isna().sum()}")
    df['power'].interpolate(method='linear', inplace=True)

    # 检查最终数据
    print(f"处理后数据点: {len(df)}")
    print(f"时间范围: {df.index.min()} 到 {df.index.max()}")

    return df


# ----------------------
# 2. 特征工程
# ----------------------
def create_features(df, country='CN'):
    """创建时间特征和节假日特征"""
    print("创建特征工程...")

    # 基本时间特征
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year

    # 是否为周末
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    # 节假日标记（使用holidays库）
    print("标记节假日...")
    years = df.index.year.unique()
    country_holidays = holidays.CountryHoliday(country, years=years.tolist())
    df['is_holiday'] = df.index.map(lambda x: 1 if x in country_holidays else 0)

    # 周期性特征编码
    print("创建周期性特征...")
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # 滞后特征 (过去24小时和168小时)
    print("创建滞后特征...")
    df['lag_24h'] = df['power'].shift(24)
    df['lag_168h'] = df['power'].shift(168)

    # 移动平均特征
    print("创建移动平均特征...")
    df['rolling_24h_mean'] = df['power'].rolling(window=24, min_periods=1).mean().shift(1)
    df['rolling_168h_mean'] = df['power'].rolling(window=168, min_periods=1).mean().shift(1)

    # 删除初始的NaN行
    initial_nan_count = df.isna().sum().sum()
    df.dropna(inplace=True)
    print(f"删除包含NaN的行: {initial_nan_count} -> 剩余行: {len(df)}")

    return df


# ----------------------
# 3. 数据准备
# ----------------------
def prepare_data(df, n_past=168, n_future=24):
    """准备训练数据集"""
    print("准备训练数据...")

    # 选择特征列
    feature_cols = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                    'day_sin', 'day_cos', 'is_weekend', 'is_holiday',
                    'lag_24h', 'lag_168h', 'rolling_24h_mean', 'rolling_168h_mean']

    # 目标列
    target_col = 'power'

    print(f"使用特征: {feature_cols}")
    print(f"目标变量: {target_col}")

    # 数据归一化
    print("数据归一化...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[feature_cols + [target_col]])

    # 分离特征和目标
    X = scaled_data[:, :-1]
    y = scaled_data[:, -1]

    # 创建时间窗口数据集
    print(f"创建时间窗口 (历史 {n_past}小时, 预测 {n_future}小时)...")
    X_data, y_data = [], []
    for i in range(n_past, len(scaled_data) - n_future):
        X_data.append(X[i - n_past:i])
        y_data.append(y[i:i + n_future])

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    print(f"总样本数: {len(X_data)}")

    # 划分训练集和测试集 (按时间顺序)
    split_index = int(0.8 * len(X_data))
    X_train, X_test = X_data[:split_index], X_data[split_index:]
    y_train, y_test = y_data[:split_index], y_data[split_index:]

    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

    return X_train, X_test, y_t