from utils.utils import *

from .train import *
from .validate import *


def train_validate(
    model, train_dataloader, valid_dataloader, processor, device, num_epochs, save_dir, deepspeed_config,
    patience
):

    logger = logging.getLogger('train_validate')

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.path.join(save_dir, 'training.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    logger.propagate = False

    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard'))

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=deepspeed_config,
        model_parameters=[p for p in model.parameters() if p.requires_grad]
    )

    trainable_params = 0
    all_param = 0
    for name, param in model_engine.module.named_parameters():
        num_params = param.numel()
        if param.requires_grad:
            trainable_params += num_params
        all_param += num_params
    logger.info(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss = train(
            model=model_engine,
            dataloader=train_dataloader,
            device=device,
        )
        train_losses.append(train_loss)
        logger.info(f"Train Loss: {train_loss:.4f}")

        val_loss, avg_iou, iou_over_05_percent, iou_over_07_percent = validate(
            model=model_engine,
            dataloader=valid_dataloader,
            device=device,
            processor=processor
        )
        logger.info(f"Validation Loss: {val_loss:.4f}")
        logger.info(f"Validation Avg IoU: {avg_iou * 100:.2f}%")
        logger.info(f"Validation IoU >= 0.5: {iou_over_05_percent:.2f}%")
        logger.info(f"Validation IoU >= 0.7: {iou_over_07_percent:.2f}%")

        val_losses.append(val_loss)
        val_losses.append(val_loss)

        writer.add_scalar('Loss/Train', train_loss, epoch + 1)
        writer.add_scalar('Loss/Validation', val_loss, epoch + 1)
        writer.add_scalar('Loss/Train', train_loss, epoch + 1)
        writer.add_scalar('Loss/Validation', val_loss, epoch + 1)
        writer.add_scalar('IoU/Avg', avg_iou, epoch + 1)
        writer.add_scalar('IoU/Over_0.5', iou_over_05_percent / 100.0, epoch + 1)
        writer.add_scalar('IoU/Over_0.7', iou_over_07_percent / 100.0, epoch + 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            ds_checkpoint_path = os.path.join(save_dir, 'deepspeed_checkpoint_best')
            model_engine.save_checkpoint(
                save_dir=ds_checkpoint_path,
                tag="best", 
                client_state={'epoch': epoch + 1} 
            )
            logger.info(f"DeepSpeed checkpoint saved at epoch {epoch + 1}")

            lora_best_dir = os.path.join(save_dir, 'lora_best')
            os.makedirs(lora_best_dir, exist_ok=True)
            model_engine.module.language_model.save_pretrained(lora_best_dir)
            logger.info(f"LoRA weights saved at epoch {epoch + 1}")

            custom_weights_path = os.path.join(save_dir, 'custom_layers_best.pt')
            torch.save({
                'qformer': model_engine.module.qformer.state_dict(),
                'mamba_tr': model_engine.module.mamba_tr.state_dict(),
                'language_projection': model_engine.module.language_projection.state_dict(),
            }, custom_weights_path)
            logger.info("Custom layers saved")

        else:
            epochs_no_improve += 1
            logger.info(f"Validation Loss has not improved for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping: No performance improvement for {patience} epochs.")
            break

    writer.close()
    logger.info("Training Complete")

    return train_losses, val_losses