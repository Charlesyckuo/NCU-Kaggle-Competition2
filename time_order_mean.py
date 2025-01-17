import pandas as pd
import matplotlib.pyplot as plt

# 讀取 train_data.csv
train_data = pd.read_csv('train_data.csv')

# 計算每個 data_id 的時間序列長度
sequence_lengths = train_data.groupby('data_id').size()

# 計算平均時間序列長度
mean_sequence_length = sequence_lengths.mean()

# 顯示結果
print("每個 data_id 的平均時間序列長度為：", mean_sequence_length)

# 顯示時間序列長度分佈統計信息
print("\n時間序列長度的統計信息：")
print(sequence_lengths.describe())

# 繪製直方圖以視覺化時間序列長度的分佈
plt.figure(figsize=(10, 6))
plt.hist(sequence_lengths, bins=30, color='blue', alpha=0.7)
plt.title('Histogram of Sequence Lengths')
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 如果需要匯出統計結果
sequence_lengths.to_csv('sequence_lengths.csv', index=True, header=['Sequence Length'])
