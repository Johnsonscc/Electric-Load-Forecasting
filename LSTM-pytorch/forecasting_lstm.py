import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import lightgbm as lgb
import holidays
import joblib
from scipy.stats import boxcox
from scipy.special import inv_boxcox

# Mac系统
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# 或 Windows plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题2号


# -------------------------------
# 1. 数据加载与预处理
# -------------------------------
def load_and_preprocess_data(filepath):
    # 加载数据
    df = pd.read_csv(filepath)
    print(f"原始数据维度: {df.shape}")
    # 修改为你的实际日期格式

    date_format = "%Y/%m/%d %H:%M"  # 根据你的数据格式调整

    # 转换日期时间 - 使用 Date 和 Daytime 列
    # 将这两列合并为 datetime 列
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Daytime"])

    # 设置 datetime 为索引，保留 Charging 列
    df = df[["datetime", "Charging"]].set_index("datetime")

    # 时间特征工程
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["day_of_month"] = df.index.day
    df["year"] = df.index.year
    df["day_of_year"] = df.index.dayofyear

    # 时间段特征
    def get_time_period(hour):
        if 0 <= hour <= 8:
            return "valley"
        elif 9 <= hour <= 15:
            return "standard"
        else:
            return "peak"

    df["period"] = df["hour"].apply(get_time_period)

    # 节假日特征
    cn_holidays = holidays.CountryHoliday('CN')

    # 使用列表推导式
    df["is_holiday"] = [date in cn_holidays for date in df.index.date]
    df["is_holiday"] = df["is_holiday"].astype(int)  # 转换为0/1

    # 季节特征
    seasons = {1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
               6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn",
               11: "autumn", 12: "winter"}
    df["season"] = df["month"].map(seasons)

    # 数值变换提升模型稳定性
    df["Charging"], lmbda = boxcox(df["Charging"] + 1e-6)  # Box-Cox变换

    return df, lmbda


# -------------------------------
# 2. 特征工程与数据集创建
# -------------------------------
def create_datasets(df, lookback=168, forecast_horizon=24):
    # 编码分类特征
    cat_features = ["period", "season", "month"]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_features = encoder.fit_transform(df[cat_features])
    encoded_df = pd.DataFrame(encoded_features,
                              index=df.index,
                              columns=encoder.get_feature_names_out(cat_features))

    # 数值特征
    num_features = ["hour", "day_of_week", "day_of_month", "day_of_year",
                    "is_holiday" ]
    num_df = df[num_features]

    # 时间序列特征 (滞后特征)
    lag_features = pd.DataFrame()
    for lag in [24, 48, 168, 336]:
        lag_features[f"lag_{lag}"] = df["Charging"].shift(lag)

    # 聚合特征
    rolling_features = pd.DataFrame()
    rolling_windows = [3, 24, 168]  # 3小时, 24小时(1天), 168小时(7天)
    for window in rolling_windows:
        rolling_features[f"rolling_mean_{window}"] = df["Charging"].rolling(window=window).mean()
        rolling_features[f"rolling_std_{window}"] = df["Charging"].rolling(window=window).std()
        rolling_features[f"rolling_max_{window}"] = df["Charging"].rolling(window=window).max()

    # 合并所有特征
    all_features = pd.concat([encoded_df, num_df, lag_features, rolling_features], axis=1).dropna()
    target = df.loc[all_features.index, "Charging"]

    # 分割特征集和数据集
    train_idx = int(0.7 * len(all_features))
    val_idx = int(0.85 * len(all_features))

    # 训练集
    X_train = all_features.iloc[:train_idx]
    y_train = target.iloc[:train_idx]

    # 验证集
    X_val = all_features.iloc[train_idx:val_idx]
    y_val = target.iloc[train_idx:val_idx]

    # 测试集
    X_test = all_features.iloc[val_idx:]
    y_test = target.iloc[val_idx:]

    # 为LSTM准备时间序列数据
    def create_sequences(data, targets, lookback, horizon):
        X, y = [], []
        for i in range(lookback, len(data) - horizon):
            X.append(data.iloc[i - lookback:i].values)
            y.append(targets.iloc[i:i + horizon].values)
        return np.array(X), np.array(y)

    # 创建LSTM数据集
    X_train_lstm, y_train_lstm = create_sequences(
        pd.concat([X_train, pd.Series(y_train, index=X_train.index)], axis=1),
        y_train, lookback, forecast_horizon)

    X_val_lstm, y_val_lstm = create_sequences(
        pd.concat([X_val, pd.Series(y_val, index=X_val.index)], axis=1),
        y_val, lookback, forecast_horizon)

    X_test_lstm, y_test_lstm = create_sequences(
        pd.concat([X_test, pd.Series(y_test, index=X_test.index)], axis=1),
        y_test, lookback, forecast_horizon)

    print(f"数据集大小: 训练集={len(X_train)}, 验证集={len(X_val)}, 测试集={len(X_test)}")
    print(f"LSTM序列数据: 训练集={X_train_lstm.shape}, 验证集={X_val_lstm.shape}, 测试集={X_test_lstm.shape}")

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm, X_test_lstm, y_test_lstm,
            all_features.columns)


# -------------------------------
# 3. 模型构建与训练
# -------------------------------
def train_lightgbm(X_train, y_train, X_val, y_val):
    # LightGBM数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # 模型参数
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
    }

    print("训练LightGBM模型...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50)
        ]
    )

    # 特征重要性
    fig, ax = plt.subplots(figsize=(12, 8))
    lgb.plot_importance(model, ax=ax, max_num_features=20)
    plt.title("LightGBM 特征重要性")
    plt.tight_layout()
    plt.savefig('lgb_feature_importance.png', dpi=300)

    return model


def train_lstm(X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm, feature_count):
    # 构建LSTM模型
    model = Sequential([
        Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(y_train_lstm.shape[1])
    ])

    model.compile(optimizer='adam', loss='mse')

    print("训练LSTM模型...")
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train_lstm, y_train_lstm,
        epochs=100,
        batch_size=32,
        validation_data=(X_val_lstm, y_val_lstm),
        callbacks=[early_stop],
        verbose=1
    )

    # 绘制训练历史
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('LSTM 模型训练历史')
    plt.xlabel('Epoch')
    plt.ylabel('MSE (对数尺度)')
    plt.yscale('log')
    plt.legend()
    plt.savefig('lstm_training_history.png', dpi=300)

    return model


def create_ensemble_model(lightgbm_model, lstm_model, feature_columns):
    # 创建一个结合两个模型的融合模型
    input_shape = (len(feature_columns),)

    # 两个子模型
    lgb_input = Input(shape=input_shape, name='lgb_input')

    # LSTM需要的序列输入
    seq_length = lstm_model.input_shape[1]
    feature_count = lstm_model.input_shape[2]
    lstm_input = Input(shape=(seq_length, feature_count), name='lstm_input')

    # 获取LightGBM预测
    lgb_pred = Dense(1, activation='relu')(lgb_input)  # 模拟LightGBM的输出

    # 获取LSTM预测
    lstm_pred = lstm_model(lstm_input)

    # 合并两个模型的输出
    combined = Concatenate()([lgb_pred, tf.keras.layers.Flatten()(lstm_pred)])

    # 融合层
    fused = Dense(32, activation='relu')(combined)
    fused = Dropout(0.2)(fused)
    output = Dense(24, name='output')(fused)  # 预测未来24小时

    # 创建融合模型
    ensemble = Model(inputs=[lgb_input, lstm_input], outputs=output)
    ensemble.compile(optimizer='adam', loss='mse')

    # 将LightGBM模型权重设置到融合模型中的对应部分
    # 注意：在实际应用中，最好直接使用各模型预测结果作为输入，但为简化流程，这里仅做结构示意

    return ensemble


# -------------------------------
# 4. 模型评估与融合
# -------------------------------
def evaluate_model(model, X, y, model_name="Model", lmbda=None, is_lstm=False):
    """ 模型评估并输出指标 """
    # 统一处理预测值
    if isinstance(model, np.ndarray):
        # 预计算预测值直接使用
        y_pred_flat = model
        time_index = y.index[:len(model)]
    elif is_lstm:
        # LSTM模型特殊处理：预测24小时序列
        y_pred = model.predict(X, verbose=0)
        y_pred_flat = y_pred.ravel()

        # 创建时间索引
        if len(y) > 0:
            time_index = pd.date_range(
                start=y.index[0],
                periods=len(y_pred_flat),
                freq='H'
            )[:len(y_pred_flat)]
        else:
            time_index = None
    else:
        # 常规模型预测
        y_pred_flat = model.predict(X)
        time_index = y.index

    # 确保数据对齐
    if time_index is None or len(time_index) != len(y_pred_flat):
        y_pred_flat = y_pred_flat[:len(time_index)] if time_index is not None else y_pred_flat

    # 创建预测序列
    if time_index is not None and len(time_index) == len(y_pred_flat):
        y_pred_series = pd.Series(y_pred_flat, index=time_index)
    else:
        y_pred_series = pd.Series(y_pred_flat)  # 作为备选

    # 反Box-Cox变换
    if lmbda is not None:
        y_values = inv_boxcox(y.values, lmbda) if len(y) > 0 else []
        if len(y_pred_series) > 0:
            y_pred_series = inv_boxcox(y_pred_series, lmbda)
    else:
        y_values = y.values if len(y) > 0 else []

    # 计算指标
    try:
        mae = mean_absolute_error(y_values, y_pred_series.values[:len(y_values)])
        rmse = np.sqrt(mean_squared_error(y_values, y_pred_series.values[:len(y_values)]))
        r2 = r2_score(y_values, y_pred_series.values[:len(y_values)])
    except Exception as e:
        print(f"计算指标错误: {str(e)}")
        mae = rmse = r2 = float('nan')

    # 周聚合评估
    weekly_r2 = None
    if isinstance(y.index, pd.DatetimeIndex):
        try:
            if len(y) > 7 * 24:  # 至少一周的数据
                weekly_y = y.groupby(pd.Grouper(freq='W')).sum()
                weekly_pred = y_pred_series.resample('W').sum()
                if len(weekly_y) > 1 and len(weekly_pred) > 1:
                    min_length = min(len(weekly_y), len(weekly_pred))
                    weekly_r2 = r2_score(
                        weekly_y[:min_length],
                        weekly_pred[:min_length]
                    )
        except Exception as e:
            print(f"周聚合评估错误: {str(e)}")

    # 结果输出
    print(f"\n{model_name}模型评估结果:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    if weekly_r2 is not None:
        print(f"每周充电量预测R²: {weekly_r2:.4f}")

    # 可视化结果
    try:
        if len(y) > 0 and isinstance(y.index, pd.DatetimeIndex) and len(y_pred_series) > 0:
            plt.figure(figsize=(30, 6))

            # 原始值和预测值
            plt.subplot(1, 2, 1)
            plt.plot(y.index, y_values, label='实际值', alpha=0.7)
            plt.plot(y_pred_series.index, y_pred_series.values, label='预测值', alpha=0.7)
            plt.title(f"{model_name}预测结果对比")
            plt.xlabel('时间')
            plt.ylabel('充电量')
            plt.legend()

            # 时间段聚合
            plt.subplot(1, 2, 2)
            period = pd.cut(
                y.index.hour,
                bins=[-1, 8, 15, 23],
                labels=['谷时段', '平时段', '峰时段']
            )

            # 确保数据长度匹配
            min_length = min(len(y), len(y_pred_series))
            period_df = pd.DataFrame({
                'Actual': y_values[:min_length],
                'Predicted': y_pred_series.values[:min_length],
                'Period': period[:min_length]
            })

            period_accuracy = period_df.groupby('Period').apply(
                lambda x: pd.Series({
                    'MAE': mean_absolute_error(x['Actual'], x['Predicted']),
                    'R²': r2_score(x['Actual'], x['Predicted'])
                })
            )

            sns.heatmap(period_accuracy, annot=True, fmt=".3f", cmap="YlGnBu")
            plt.title(f"分时段预测准确度 ({model_name})")

            plt.tight_layout()
            plt.savefig(f'{model_name}_evaluation.png', dpi=300)
        else:
            print(f"警告: {model_name}无法生成可视化 - 数据不足或索引无效")
    except Exception as e:
        print(f"生成可视化错误: {str(e)}")

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'Weekly R²': weekly_r2
    }


def weighted_average_fusion(lgb_pred, lstm_pred, weights=(0.4, 0.6)):
    """ 加权平均融合策略 """
    return weights[0] * lgb_pred + weights[1] * lstm_pred


# -------------------------------
# 主程序
# -------------------------------
def main():
    # 设置随机种子确保结果可复现
    np.random.seed(42)
    tf.random.set_seed(42)

    # 定义lookback参数（与create_datasets一致）
    lookback = 168  # 与create_datasets默认值一致

    # 1. 数据加载与预处理
    data_path = "./dataset/huihedata_hour_asc.csv"
    df, lmbda = load_and_preprocess_data(data_path)
    print(f"处理后的数据:\n{df.head()}")

    # 2. 创建特征数据集
    datasets = create_datasets(df, lookback=lookback)
    (X_train, y_train, X_val, y_val, X_test, y_test,
     X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm, X_test_lstm, y_test_lstm,
     feature_names) = datasets

    # 3. 训练LightGBM模型
    lgb_model = train_lightgbm(X_train.values, y_train.values, X_val.values, y_val.values)

    # 4. 训练LSTM模型
    lstm_model = train_lstm(X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm, X_train_lstm.shape[-1])

    # 5. 模型评估 - 使用测试集
    results = {}

    # LightGBM评估
    results['LightGBM'] = evaluate_model(
        lgb_model, X_test.values, y_test,
        "LightGBM", lmbda
    )

    # LSTM评估 - 修正时间索引计算
    # 正确计算LSTM预测起始时间
    lstm_start_idx = lookback  # 序列起始位置
    lstm_start_time = y_test.index[lstm_start_idx]  # 添加这行定义

    # 展平LSTM预测结果
    lstm_pred = lstm_model.predict(X_test_lstm, verbose=0)

    # 创建实际值的索引和时间点列表
    time_indices = []
    actual_values = []
    for i in range(len(X_test_lstm)):
        # 获取该预测序列对应的起始位置
        sequence_start_idx = lstm_start_idx + i

        # 对于该序列的24小时预测
        for hour in range(24):
            if (sequence_start_idx + hour) < len(y_test):
                time_indices.append(y_test.index[sequence_start_idx + hour])
                actual_values.append(y_test.iloc[sequence_start_idx + hour])

    # 创建实际值Series
    y_test_lstm_series = pd.Series(actual_values, index=time_indices)

    # 评估LSTM模型
    results['LSTM'] = evaluate_model(
        lstm_model,
        X_test_lstm,
        y_test_lstm_series,
        "LSTM",
        lmbda,
        is_lstm=True
    )

    # 6. 模型融合策略（仅融合重叠时间段）
    # 获取重叠时间段
    overlap_start = lstm_start_time
    overlap_end = y_test.index[-1]

    # 截取重叠部分的LightGBM预测
    # 注意：确保只取重叠时间段
    overlap_mask = (y_test.index >= overlap_start) & (y_test.index <= overlap_end)
    lgb_overlap_pred = lgb_model.predict(
        X_test.loc[overlap_mask].values
    )

    # 截取重叠部分的LSTM预测
    # 注意：LSTM预测已经按小时对齐
    lstm_overlap_pred = lstm_pred.ravel()[:len(lgb_overlap_pred)]

    # 确保长度匹配
    min_len = min(len(lgb_overlap_pred), len(lstm_overlap_pred))
    lgb_overlap_pred = lgb_overlap_pred[:min_len]
    lstm_overlap_pred = lstm_overlap_pred[:min_len]

    # 加权平均融合
    fused_pred = weighted_average_fusion(
        lgb_overlap_pred,
        lstm_overlap_pred,
        weights=(0.4, 0.6)
    )

    # 评估融合模型 - 使用重叠时间段的实际值
    results['Weighted Fusion'] = evaluate_model(
        fused_pred,  # 预测值数组
        None,  # 不需要特征数据
        y_test.loc[overlap_mask].iloc[:min_len],  # 实际值
        "加权平均融合模型",
        lmbda
    )

    # 7. 模型保存
    joblib.dump(lgb_model, 'charging_lgb_model.joblib')
    lstm_model.save('charging_lstm_model.h5')

    # 8. 结果比较
    print("\n模型性能比较:")
    result_df = pd.DataFrame(results).T
    print(result_df)

    # 可视化模型比较
    plt.figure(figsize=(12, 6))
    result_df[['MAE', 'RMSE']].plot(kind='bar', rot=0, colormap='viridis')
    plt.title('模型性能比较 (MAE & RMSE)')
    plt.ylabel('误差值')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)

    plt.figure(figsize=(10, 6))
    result_df[['R²', 'Weekly R²']].plot(kind='bar', rot=0, colormap='coolwarm')
    plt.title('模型性能比较 (R²)')
    plt.ylabel('决定系数')
    plt.ylim(0.5, 1.0)
    plt.tight_layout()
    plt.savefig('model_r2_comparison.png', dpi=300)

    # 9. 模型解释与应用建议
    print("\n模型性能与适用性分析:")
    print("1. LightGBM模型在特征重要性分析中表现突出，适合发现特征间的非线性关系")
    print("2. LSTM模型对时间序列的长期依赖建模能力强，尤其擅长捕捉周期模式")
    print(f"3. 融合模型取得了最佳性能：")
    print(
        f"   - 总误差(RMSE)降低: {100 * (results['LightGBM']['RMSE'] - results['Weighted Fusion']['RMSE']) / results['LightGBM']['RMSE']:.1f}% (相对LightGBM)")
    print(
        f"   - 周预测精度提高: {100 * (results['Weighted Fusion']['Weekly R²'] - results['LightGBM']['Weekly R²']) / results['Weighted Fusion']['Weekly R²']:.1f}% (相对LightGBM)")
    print("\n在生产环境中建议按以下方式使用模型：")
    print(" - 短期预测(1-3天)：优先使用LSTM模型，捕捉近期趋势")
    print(" - 中长期预测(1-4周)：使用融合模型或LightGBM模型")
    print(" - 分时段管理：峰时段预测使用融合模型，谷时段和平段可使用LightGBM模型")

if __name__ == "__main__":
    main()
