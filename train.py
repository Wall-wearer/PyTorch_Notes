import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from PIL import Image


# Check if MPS is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Hyperparameters
batch_size = 512
num_workers = 0  # Set to 0 for macOS to avoid threading issues
lr = 1e-4
epochs = 200
# Data Transforms
image_size = 28


data_transform = transforms.Compose([
    transforms.ToPILImage(),    # let the picture transfer to a PIL image. With the shape of (28, 28) or (1, 28, 28)
    transforms.Resize(image_size),  # let the previous PIL image become (28, 28)
    transforms.ToTensor()   # make the previous image with values normalized to [0, 1]
])


# Custom Dataset Class
class FMDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.images = df.iloc[:, 1:].values.astype(np.uint8)
        self.labels = df.iloc[:, 0].values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28, 1)
        label = int(self.labels[idx])
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.tensor(image / 255., dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        return image, label


# Load DataFrames
train_df = pd.read_csv("train_images/fashion-mnist_train.csv")
test_df = pd.read_csv("train_images/fashion-mnist_test.csv")

# Create Datasets
train_data = FMDataset(train_df, data_transform)
test_data = FMDataset(test_df, data_transform)

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),     # 全连接层，接受64x4x4个特征输入，得到512个特征输出
            # nn.Linear(input, output), input here is the size of the picture 28pixel* 28
            # output: hyperparameter I selected in batch_size = 256
            nn.ReLU(),      # 调用ReLU，采用线性整流
            # ReLU gets input from previous layer.
            nn.Linear(512, 10)
            # Here, the input should be same with the output in last step.
            # And the output should be same with my target output. Here, I want to make 10 classes of output.
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc(x)
        return x


model = Net().to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Track training progress
train_losses = []
val_losses = []
val_accuracies = []

# Initialize TensorBoard writer
writer = SummaryWriter('logs/fashion_mnist_experiment_1')


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

        # Log training loss to TensorBoard
        writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)

    train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f}')


def val(epoch):
    model.eval()
    val_loss = 0
    gt_labels = []
    pred_labels = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            preds = torch.argmax(output, 1)
            gt_labels.append(label.cpu().numpy())
            pred_labels.append(preds.cpu().numpy())
            loss = criterion(output, label)
            val_loss += loss.item() * data.size(0)

            # Log validation loss to TensorBoard
            writer.add_scalar('Validation Loss', loss.item(), epoch * len(test_loader) + batch_idx)

    val_loss = val_loss / len(test_loader.dataset)
    gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
    acc = np.sum(gt_labels == pred_labels) / len(pred_labels)
    val_losses.append(val_loss)
    val_accuracies.append(acc)
    print(f'Epoch: {epoch} \tValidation Loss: {val_loss:.6f}, Accuracy: {acc:.6f}')

    # Log accuracy to TensorBoard
    writer.add_scalar('Accuracy', acc, epoch)

    # Visualization
    visualize_predictions(data.cpu(), preds.cpu(), label.cpu(), epoch)


def visualize_predictions(images, predictions, labels, epoch, num_images=6):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    if num_images == 1:
        axes = [axes]
    for i in range(num_images):
        axes[i].imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
        axes[i].set_title(f'Pred: {class_names[predictions[i]]}\nActual: {class_names[labels[i]]}')
        axes[i].axis('off')
    writer.add_figure('Predictions', fig, global_step=epoch)


def log_histograms(epoch):
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)
        if param.grad is not None:
            writer.add_histogram(name + '/grad', param.grad, epoch)


for epoch in range(1, epochs + 1):
    train(epoch)
    val(epoch)

save_path = "FashionModel.pkl"
torch.save(model, save_path)


dummy_input = torch.zeros((1, 1, 28, 28), device=device)
writer.add_graph(model, dummy_input)
# Close the TensorBoard writer
writer.close()

# Plotting loss and accuracy
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

plt.tight_layout()
plt.show()