from utils.utils import *

def parse_time(time):
    try:
        return float(time)
    except ValueError:
        return None

def interpret_time(time):
   starts_pattern = re.search(r'starts? at (\d+\.?\d*)', time.lower())
   ends_pattern = re.search(r'ends? at (\d+\.?\d*)', time.lower())

   if starts_pattern and ends_pattern:
       try:
           start = float(starts_pattern.group(1))
           end = float(ends_pattern.group(1))
           return start, end
       except ValueError:
           return None, None

   return None, None

def calculate_iou (pred_start, pred_end, gt_start, gt_end):
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return intersection / union if union > 0 else 0

def validate(model, dataloader, device, processor):
    model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_samples = 0
    iou_over_05 = 0
    iou_over_07 = 0

    progress_bar = tqdm(dataloader, desc='Validating', leave=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            labels = batch['labels']
            predicted_token_ids = torch.argmax(logits, dim=-1)

            decoded_preds = processor.batch_decode(predicted_token_ids, skip_special_tokens=True)
            labels_for_decoding = labels.detach().clone()
            labels_for_decoding[labels_for_decoding == -100] = processor.tokenizer.pad_token_id
            decoded_labels = processor.batch_decode(labels_for_decoding, skip_special_tokens=True)

            for label_text, pred_text in zip(decoded_labels, decoded_preds):
                gt_start, gt_end = interpret_time(label_text) 
                pred_start, pred_end = interpret_time(pred_text)

                if all(t is not None for t in [gt_start, gt_end, pred_start, pred_end]):
                    iou = calculate_iou(pred_start, pred_end, gt_start, gt_end) 
                else:
                    iou = 0.0

                total_iou += iou
                total_samples += 1

                if iou >= 0.5:
                    iou_over_05 += 1
                if iou >= 0.7:
                    iou_over_07 += 1

            current_avg_loss = total_loss / (batch_idx + 1)
            current_avg_iou = total_iou / total_samples if total_samples > 0 else 0.0
            current_iou_over_05_percent = (iou_over_05 / total_samples) * 100 if total_samples > 0 else 0.0
            current_iou_over_07_percent = (iou_over_07 / total_samples) * 100 if total_samples > 0 else 0.0

            progress_bar.set_postfix({
                'Loss': f'{current_avg_loss:.4f}',
                'Avg_IoU': f'{current_avg_iou * 100:.2f}',
                'IoU_05': f'{current_iou_over_05_percent:.2f}%',
                'IoU_07': f'{current_iou_over_07_percent:.2f}%'
            })

    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / total_samples if total_samples > 0 else 0.0
    iou_over_05_percent = (iou_over_05 / total_samples) * 100 if total_samples > 0 else 0.0
    iou_over_07_percent = (iou_over_07 / total_samples) * 100 if total_samples > 0 else 0.0

    return avg_loss, avg_iou, iou_over_05_percent, iou_over_07_percent