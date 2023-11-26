import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

# ダミーデータ
X_train = torch.randn((100, 10))
y_train = torch.randint(0, 2, (100,))

# TensorDatasetを作成
train_dataset = TensorDataset(X_train, y_train)

# DataLoaderを作成
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# シンプルなニューラルネットワークの定義
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        x = self.fc(x)
        return x


# モデルのインスタンス化
model = SimpleNN()

# 損失関数の定義
criterion = nn.CrossEntropyLoss()

# 最適化アルゴリズムの定義
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 訓練ループの例
for epoch in range(100):
    for inputs, labels in train_dataloader:
        # 勾配を初期化
        optimizer.zero_grad()

        # フォワードパス
        outputs = model(inputs)

        # 損失の計算
        loss = criterion(outputs, labels)

        # バックワードパスとパラメータの更新
        loss.backward()
        optimizer.step()


# モデルの保存
torch.save(model.state_dict(), "model.pth")

# モデルの読み込み
loaded_model = SimpleNN()
loaded_model.load_state_dict(torch.load("model.pth"))


# モデルの評価モードに切り替える
loaded_model.eval()

# テストデータを用意
X_test = torch.randn((20, 10))
y_test = torch.randint(0, 2, (20,))

# テストデータで予測
with torch.no_grad():
    test_outputs = loaded_model(X_test)
    _, predicted_labels = torch.max(test_outputs, 1)

# 精度の計算
accuracy = torch.sum(predicted_labels == y_test).item() / len(y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
