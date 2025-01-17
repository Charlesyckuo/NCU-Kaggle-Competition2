# Kaggle Competition 2 113522042
## Project Structure
- Data Files
    - Code_and_data
        - train_data.csv
        - test_data.csv
        - sample_submission.csv
        - sequence_lengths.csv
        - submission.csv
        - final.py
        - time_oder_mean.py

## Requirements
- 可以直接導入 kaggle2.yml
- torch==2.5.1
- torchvision==0.20.1
- pandas==2.2.2
- numpy==1.26.4
- scikit-learn==1.5.1
## 程式碼運行及說明
- 統計時間序列長度
    - 執行 "time_order_mean.py"
    - mean     2495.300766
    - std       389.603178
    - min       623.000000
    - 25%      2287.750000
    - 50%      2553.500000
    - 75%      2756.000000
    - max      4696.000000
    - 可以自己決定要設定final.py中的SEQ_LEN為多少
    - 會輸出sequence_lengths.csv可以看到每個id的時間步數
- 訓練與預測
    - 執行 "final.py"
    - 可以自行調整EPOCHS，建議設置為110
    - 使用gpu進行訓練因此非常快
    - 輸出預測為submission.csv
    - 提交submission.csv
## 注意事項
- 使用gpu進行訓練，須注意自身cuda以及pytorch的板本是否相容
- conda環境仍有保留 可以直接導入 kaggle2.yml
