from utils.utils import *
from train.train_validate import train_validate
from data.dataloader import get_dataloaders
from model.model import get_model

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--deepspeed_config', type=str, required=True)
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--train_txt', type=str, required=True)
    parser.add_argument('--test_txt', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.deepspeed_config, 'r') as f:
        ds_config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required.")
    
    model = get_model()
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl", use_fast=True)
    
    train_dataloader, valid_dataloader, _ = get_dataloaders(
        video_dir=args.video_dir,
        train_txt=args.train_txt,
        test_txt=args.test_txt,
        processor=processor,
        batch_size=args.batch_size
    )

    try:
        train_losses, val_losses = train_validate(
            model=model,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            processor=processor,
            device=device,
            num_epochs=args.num_epochs,
            save_dir=args.save_dir,
            deepspeed_config=ds_config,
            patience=args.patience
        )

    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise e
