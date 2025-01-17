import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ---------------------
# 1. DATA PROCESSING
# ---------------------

# Load data
train_data = pd.read_csv('./train_data.csv')
train_info = pd.read_csv('./train_info.csv')
test_data = pd.read_csv('./test_data.csv')

# Process sequences with padding
def process_time_series(data, seq_len=None):
    grouped = data.groupby('data_id')
    sequences = []
    ids = []
    for data_id, group in grouped:
        seq = group[['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']].values
        if seq_len:
            # Pad or truncate to fixed length
            if len(seq) > seq_len:
                seq = seq[:seq_len]
            elif len(seq) < seq_len:
                seq = np.pad(seq, ((0, seq_len - len(seq)), (0, 0)), 'constant')
        sequences.append(seq)
        ids.append(data_id)
    return np.array(sequences), ids

# Standardize features
def standardize(sequences):
    scaler = StandardScaler()
    flat = sequences.reshape(-1, sequences.shape[-1])
    scaled = scaler.fit_transform(flat).reshape(sequences.shape)
    return scaled, scaler

# Fixed sequence length
SEQ_LEN = 2500
X_train_seq, train_ids = process_time_series(train_data, seq_len=SEQ_LEN)
X_train_seq, scaler = standardize(X_train_seq)

X_test_seq, test_ids = process_time_series(test_data, seq_len=SEQ_LEN)
X_test_seq = scaler.transform(X_test_seq.reshape(-1, X_test_seq.shape[-1])).reshape(X_test_seq.shape)

# Labels for training
y_gender = train_info['gender'].values
y_hold_racket = train_info['hold racket handed'].values
y_play_years = train_info['play years'].values
y_level = train_info['level'].values

# Dataset and DataLoader
class FullSequenceDataset(Dataset):
    def __init__(self, sequences, gender, hold_racket, play_years, level):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.gender = torch.tensor(gender, dtype=torch.float32).unsqueeze(1)
        self.hold_racket = torch.tensor(hold_racket, dtype=torch.float32).unsqueeze(1)
        self.play_years = torch.tensor(play_years, dtype=torch.long)
        self.level = torch.tensor(level, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (self.sequences[idx], self.gender[idx],
                self.hold_racket[idx], self.play_years[idx], self.level[idx])

train_dataset = FullSequenceDataset(X_train_seq, y_gender, y_hold_racket, y_play_years, y_level)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ---------------------
# 2. MODEL DESIGN
# ---------------------

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        se = self.global_pool(x).view(b, c)
        se = self.fc2(self.relu(self.fc1(se))).view(b, c, 1)
        return x * self.sigmoid(se)
    
class EnhancedCNNTransformerWithoutPE(nn.Module):
    def __init__(self, input_dim, cnn_filters, transformer_dim, nhead, num_layers, seq_len):
        super(EnhancedCNNTransformerWithoutPE, self).__init__()
        
        # SEBlock before conv1
        self.initial_se = SEBlock(input_dim)  # SEBlock with input_dim channels
        
        # Enhanced CNN Layers
        self.conv1 = nn.Conv1d(input_dim, cnn_filters, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.se1 = SEBlock(cnn_filters)
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(cnn_filters)
        self.se2 = SEBlock(cnn_filters)
        self.conv3 = nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(cnn_filters * 2)
        self.se3 = SEBlock(cnn_filters * 2)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.19)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Transformer Encoder for Global Dependencies
        transformer_seq_len = seq_len // 8
        self.input_layer = nn.Linear(cnn_filters * 2, transformer_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=nhead, dim_feedforward=transformer_dim * 4, dropout=0.2
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output Layers for Each Task
        self.fc_gender = nn.Linear(transformer_dim, 1)
        self.fc_hold_racket = nn.Linear(transformer_dim, 1)
        self.fc_play_years = nn.Linear(transformer_dim, 3)
        self.fc_level = nn.Linear(transformer_dim, 3)
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Initial SEBlock
        x = x.permute(0, 2, 1)  # Change to (batch, channels, seq_len) for SEBlock
        x = self.initial_se(x)
        
        # Enhanced CNN Layers
        x = self.se1(self.dropout(self.relu(self.bn1(self.conv1(x)))))
        x = self.se2(self.pool(self.dropout(self.relu(self.bn2(self.conv2(x))))))
        x = self.se3(self.pool(self.relu(self.bn3(self.conv3(x)))))
        
        # Transformer Layers
        x = x.permute(0, 2, 1)  # Back to (batch, seq_len, channels)
        x = self.input_layer(x)
        x = self.transformer_encoder(x)  # Directly pass to Transformer without Positional Encoding

        # SEBlock after Transformer (optional)
        #x = self.se3(x)  # You can add another SEBlock here if necessary

        # Prediction for each task
        x_mean = x.mean(dim=1)
        gender_out = self.sigmoid(self.fc_gender(x_mean))
        hold_racket_out = self.sigmoid(self.fc_hold_racket(x_mean))
        play_years_out = self.softmax(self.fc_play_years(x_mean))
        level_out = self.softmax(self.fc_level(x_mean))

        return gender_out, hold_racket_out, play_years_out, level_out


# Initialize the enhanced model without positional encoding
model = EnhancedCNNTransformerWithoutPE(
    input_dim=6, cnn_filters=96, transformer_dim=128, nhead=16, num_layers=4, seq_len=SEQ_LEN 
).cuda() 

# ---------------------
# 3. TRAINING
# ---------------------

# Loss functions
bce_loss = nn.BCELoss()  # Binary Cross Entropy for gender and hold_racket
ce_loss = nn.CrossEntropyLoss()  # Cross Entropy for play_years and level

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
# 定义 ReduceLROnPlateau
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='min',  # 监控验证损失下降
    factor=0.5,  # 学习率下降倍率
    patience=5,  # 等待 5 个 epoch 后降低学习率
    min_lr=1e-6,  # 最小学习率
    verbose=True
)

# Training loop
EPOCHS = 110
# Training loop
# Training loop with ReduceLROnPlateau
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for sequences, gender, hold_racket, play_years, level in train_loader:
        sequences, gender, hold_racket, play_years, level = (
            sequences.cuda(), gender.cuda(), hold_racket.cuda(), play_years.cuda(), level.cuda()
        )
        optimizer.zero_grad()
        gender_out, hold_racket_out, play_years_out, level_out = model(sequences)
        loss = (bce_loss(gender_out, gender) +
                bce_loss(hold_racket_out, hold_racket) +
                ce_loss(play_years_out, play_years) +
                ce_loss(level_out, level))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Validation phase (example validation loss calculation)
    val_loss = total_loss / len(train_loader)
    
    # 更新学习率调度器
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

# ---------------------
# 4. PREDICTION & SUBMISSION
# ---------------------

# Generate predictions for test data
model.eval()
test_dataset = torch.tensor(X_test_seq, dtype=torch.float32).cuda()
with torch.no_grad():
    gender_pred, hold_racket_pred, play_years_pred, level_pred = model(test_dataset)

# Create submission file
submission = pd.DataFrame({
    'data_id': test_ids,
    'gender': gender_pred.cpu().numpy().flatten(),
    'hold racket handed': hold_racket_pred.cpu().numpy().flatten(),
    'play years_0': play_years_pred.cpu().numpy()[:, 0],
    'play years_1': play_years_pred.cpu().numpy()[:, 1],
    'play years_2': play_years_pred.cpu().numpy()[:, 2],
    'level_0': level_pred.cpu().numpy()[:, 0],
    'level_1': level_pred.cpu().numpy()[:, 1],
    'level_2': level_pred.cpu().numpy()[:, 2]
})

submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")
