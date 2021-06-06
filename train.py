import os
import math
import torch
import torch.optim as optim
from torchvision import transforms, datasets

from config import cfg
from nice import NICE

# Data
transform = transforms.ToTensor()
dataset = datasets.MNIST(root='./data/mnist', train=True, transform=transform, download=True)
testset = datasets.MNIST(root='./data/mnist', train=False, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg['TRAIN_BATCH_SIZE'],
                                         shuffle=True, pin_memory=True)
testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=cfg['TEST_BATCH_SIZE'],
                                         shuffle=True, pin_memory=True)

model = NICE(data_dim=784, num_coupling_layers=cfg['NUM_COUPLING_LAYERS'])
if cfg['USE_CUDA']:
  device = torch.device('cuda')
  model = model.to(device)

opt = optim.Adam(model.parameters())

best_likelihood = -math.inf

for epoch in range(cfg['TRAIN_EPOCHS']):
  mean_likelihood = 0.0
  num_minibatches = 0

  # Train the model
  model.train()

  for batch_id, (x, _) in enumerate(dataloader):
      x = x.view(-1, 784) + torch.rand(784) / 256.
      if cfg['USE_CUDA']:
        x = x.cuda()

      # TODO: Find a better rescaling that this shit
      x = torch.clamp(x, 0, 1)

      z, likelihood = model(x)
      loss = -torch.mean(likelihood)   # NLL

      loss.backward()
      opt.step()
      model.zero_grad()

      mean_likelihood -= loss
      num_minibatches += 1

  mean_likelihood /= num_minibatches
  print('Epoch {} completed. Log Likelihood: {}'.format(epoch, mean_likelihood))

  # Try only saving models that improve on a validation set
  # Save it as best model instead of this bullshit..
  if epoch % 5 == 0:
    model.eval()
    mean_likelihood = 0.0
    batches = 0
    with torch.no_grad():
      for (x, _) in testloader:
        x = x.view(-1, 784) + torch.rand(784) / 256.
        x = torch.clamp(x, 0, 1)
        z, likelihood = model(x)
        mean_likelihood += torch.mean(likelihood)
        batches += 1

    mean_likelihood /= batches

    if mean_likelihood > best_likelihood:
      best_likelihood = mean_likelihood
      print('Found new better model, saving this one')
      print('Epoch {} has test likelihood: {}'.format(epoch, mean_likelihood))

      save_path = os.path.join(cfg['MODEL_SAVE_PATH'], '{}.pt'.format(epoch))
      torch.save(model.state_dict(), save_path)

