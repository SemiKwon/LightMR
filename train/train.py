from utils.utils import *

def train(model, dataloader, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        loss = outputs.loss
        model.backward(loss)
        model.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    average_loss = total_loss / len(dataloader)
    return average_loss