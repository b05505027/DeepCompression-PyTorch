import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class ModelLoader:
    @staticmethod
    def load_model(model_name):
        if model_name.lower() == 'alexnet':
            model = models.alexnet(weights=None)
            model.classifier[6] = nn.Linear(4096, 10)
        elif model_name.lower() == 'vgg16':
            model = models.vgg16(weights=None)
        else:
            raise ValueError("Unsupported model. Please choose 'AlexNet' or 'VGG16'.")
        return model

class ModelTrainer:
    def __init__(self, model, trainloader, testloader, stat_collector, learning_rate=3e-3):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.stat_collector = stat_collector
        self.device = torch.device('mps:0') if torch.backends.mps.is_available() else 'cpu'
        self.model.to(self.device)

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                print('labels: ', labels)
                print('predicted: ', predicted)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print('correct: {:d}  total: {:d}'.format(correct, total))
        print('accuracy: {:.2f}%'.format(100 * accuracy))
        
        
        return accuracy


    def train_and_evaluate(self, epochs):
        accuracy = self.evaluate()
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            pbar = tqdm(enumerate(self.trainloader), total=len(self.trainloader),desc=f"Epoch {epoch + 1}/{epochs}")
            for i, data in pbar:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                if self.stat_collector:
                    self.stat_collector.log_loss(loss.item())
                    if i % 10 == 0:
                        self.stat_collector.plot_stats(interval=10)
                

            # Evaluate after each epoch
            accuracy = self.evaluate()
            if self.stat_collector:
                self.stat_collector.log_accuracy(accuracy)
            pbar.set_description(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(self.trainloader)}, Accuracy: {accuracy:.2f}%")
