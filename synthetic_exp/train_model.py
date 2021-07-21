import torch

from synthetic_exp.dataset import get_data_loaders
from synthetic_exp.models import MLPNet, DeepSymmetricNet


def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            yhat = model(inputs.double())
            # yhat2 = model2(inputs.float())

            _, predicted = torch.max(yhat.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        print('%s out of %s correct (%s)' % (correct, total, correct / total))

def train_model(model):
    batch_size = 32
    train_loader, val_loader, test_loader = get_data_loaders(batch_size, batch_size)
    # model2 = DeepSymmetricNet(True)
    model = model.double()
    # model2 = model2.float()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(100):
        print('Epoch %s' % (epoch,))
        for i, (inputs, targets) in enumerate(train_loader):
            evaluate_model(model, test_loader)
            print('batch %s/%s' % (i, int(3000/batch_size)+1))
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs.double())
            # yhat2 = model2(inputs.float())
            # calculate loss
            print(yhat)
            # print(yhat2)
            print(targets.long())
            # break
            loss = criterion(yhat, targets.long())
            print(loss)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
        # break




if __name__ == '__main__':
    args = ''
    # model = MLPNet(args) # [-0.1936, -1.9254, -3.4997]
    model = DeepSymmetricNet(False) # [     0.0000, -31826.6758, -18289.0509]
    train_model(model)
