from utils.utils import *
from .dataset import CustomDataset


def collate_fn(batch, processor):
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    labels = [item['labels'] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    max_length = input_ids_padded.size(1)
    labels_padded = []
    for label in labels:
        padding_length = max_length - len(label)
        padded_label = torch.cat([label, torch.full((padding_length,), fill_value=-100, dtype=torch.long)])
        labels_padded.append(padded_label)
    labels_padded = torch.stack(labels_padded)

    pixel_values_stacked = torch.stack(pixel_values)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'pixel_values': pixel_values_stacked,
        'labels': labels_padded
    }


def get_dataloaders(video_dir, train_txt, test_txt, processor, batch_size=2, num_workers=10):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dataset = CustomDataset(video_dir, train_txt, processor)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    valid_size = total_size - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    test_dataset = CustomDataset(video_dir, test_txt, processor)

    def wrapped_collate_fn(batch):
        return collate_fn(batch, processor)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=wrapped_collate_fn,
        num_workers=num_workers
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=wrapped_collate_fn,
        num_workers=num_workers
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=wrapped_collate_fn,
        num_workers=num_workers
    )

    return train_dataloader, valid_dataloader, test_dataloader