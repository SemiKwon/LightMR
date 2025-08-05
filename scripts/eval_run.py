from utils.utils import *
from eval.eval import *
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
    parser.add_argument('--weight_dir', type=str, required=True)              # ✅ for load_model
    parser.add_argument('--ds_config_path', type=str, required=True)          # ✅ for load_model
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--train_txt', type=str, required=True)
    parser.add_argument('--test_txt', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    # Load deepspeed config
    with open(args.ds_config_path, 'r') as f:
        ds_config = json.load(f)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required.")

    # Load model and processor
    model = get_model()
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl", use_fast=True)

    # Load dataloader
    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(
        video_dir=args.video_dir,
        train_txt=args.train_txt,
        test_txt=args.test_txt,
        processor=processor,
        batch_size=args.batch_size
    )

    # Load model weights (DeepSpeed)
    model_engine = load_model(model, args.weight_dir, args.ds_config_path)

    try:
        results, avg_iou, iou_over_05, iou_over_07 = inference(
            model_engine=model_engine,
            processor=processor,
            test_dataloader=test_dataloader,
            device=device,
            beam_width=1,
            output_json_path=os.path.join(args.save_dir, "inference.json")
        )

        # Save result summary
        with open(os.path.join(args.save_dir, "inference_summary.json"), "w") as f:
            json.dump({
                "avg_iou": avg_iou,
                "iou_over_0.5": iou_over_05,
                "iou_over_0.7": iou_over_07
            }, f, indent=4)

    except Exception as e:
        logging.error(f"Inference failed with error: {str(e)}", exc_info=True)
        raise e