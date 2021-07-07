import math

import torch
import torch.nn as nn
from tqdm import tqdm


def lr_finder(train_loader, model, optimizer, criterion, device, epochs=1,
            init_value:float = 1e-8, final_value:float = 10, beta:float = 0.98):
    num = epochs*len(train_loader) - 1
    mult = (final_value/init_value) ** (1/num)
    lr = init_value

    optimizer.param_groups[0]['lr'] = lr

    avg_loss = 0
    best_loss = 0
    batch_num = 0
    losses = []
    log_lrs = []
    for e in range(epochs):
        for images,labels in tqdm(train_loader):
            batch_num += 1

            # Running Model and getting loss for one batch.
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Computing the Smooted Loss
            avg_loss = (beta * loss) + ((1-beta) * loss.item())
            smoothed_loss = avg_loss / (1 - beta**batch_num)

            # Stop if Avg Loss is Exploding
            if batch_num > 1 and avg_loss > 4*best_loss:
                print(f'\nEarly Stopping, Current Loss of {avg_loss} is Diverged')
                return log_lrs, losses

            # Record the Best Loss
            if smoothed_loss < best_loss and batch_num == 1:
                best_loss = smoothed_loss

            # Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))

            # SGD
            loss.backward()
            optimizer.step()

            # Updating LR for next step.
            lr *= mult
            optimizer.param_groups[0]['lr'] = lr

    return log_lrs, losses