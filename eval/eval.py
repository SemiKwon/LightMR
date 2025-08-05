from utils.utils import *

def load_model(model, weights_dir, ds_config_path):
    with open(ds_config_path, 'r') as f:
        ds_config = json.load(f)
        
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters()
    )
    
    print("Loading weights...")
    model_states_path = os.path.join(
        weights_dir,
        'deepspeed_checkpoint_best/best/mp_rank_00_model_states.pt'
    )
    state_dict = torch.load(model_states_path, map_location='cpu')
    
    actual_model = model_engine.module
    try:
        missing_keys, unexpected_keys = actual_model.load_state_dict(state_dict, strict=False)
        print(f"Successfully loaded model states...")
    except Exception as e:
        print(f"Error loading model states: {e}")
        raise
    
    custom_layers_path = os.path.join(weights_dir, 'custom_layers_best.pt')
    if os.path.exists(custom_layers_path):
        custom_weights = torch.load(custom_layers_path, map_location='cpu')
        
        model_attr_dict = {
            'mamba_tr': hasattr(actual_model, 'mamba_tr'),
            'qformer': hasattr(actual_model, 'qformer'),
            'language_projection': hasattr(actual_model, 'language_projection')
        }
        
        print(f"Available model attributes: {model_attr_dict}")
        
        try:
            for key in custom_weights:
                if key in model_attr_dict and model_attr_dict[key]:
                    target_state_dict = getattr(actual_model, key).state_dict()
                    for param_name, param in custom_weights[key].items():
                        if param_name in target_state_dict:
                            if target_state_dict[param_name].shape != param.shape:
                                print(f"Size mismatch for {key}.{param_name}: " +
                                      f"saved {param.shape}, current {target_state_dict[param_name].shape}")
                    
                    print(f"Loading {key} weights...")
                    try:
                        getattr(actual_model, key).load_state_dict(custom_weights[key], strict=False)
                        print(f"Successfully loaded {key}")
                    except Exception as e:
                        print(f"Error loading {key}: {e}")
                else:
                    print(f"Model does not have attribute '{key}', skipping...")
            
            print("Custom layers loaded with best effort approach")
        except Exception as e:
            print(f"Error during weight loading: {e}")
    else:
        print(f"Custom weights file not found at {custom_layers_path}")
    
    lora_path = os.path.join(weights_dir, 'lora_best')
    if os.path.exists(lora_path):
        actual_model.language_model.load_adapter(lora_path, adapter_name="default")
        print("Successfully loaded LoRA weights.")
    else:
        print(f"LoRA weights not found at {lora_path}")
    return model_engine

def inference(model_engine, processor, test_dataloader, device, beam_width=5, max_tokens=20, output_json_path=None):
    model_engine.eval()
    results = []
    total_iou = total_samples = iou_over_05 = iou_over_07 = 0
    model = model_engine.module
    
    progress_bar = tqdm(test_dataloader, desc='Processing batches')
    
    with torch.no_grad():
        for batch in progress_bar:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            batch_size, num_frames, channels, height, width = batch['pixel_values'].shape
            pixel_values = batch['pixel_values'].reshape(batch_size * num_frames, channels, height, width)
            
            vision_outputs = model.vision_model(pixel_values)
            frame_embeds = vision_outputs[0]
            
            frame_attention_mask = torch.ones(
                frame_embeds.size()[:-1],
                dtype=torch.long,
                device=frame_embeds.device
            )
            
            query_tokens = model.query_tokens.expand(frame_embeds.size(0), -1, -1)
            
            qformer_outputs = model.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=frame_embeds,
                encoder_attention_mask=frame_attention_mask,
            )
            
            query_output = qformer_outputs.last_hidden_state
            num_queries = query_output.size(1)
            query_output = query_output.view(batch_size, num_frames, num_queries, -1)
            
            reduced_query_output = model.mamba_tr(query_output)
            
            B, F, K, D = reduced_query_output.shape
            reduced_query_output = reduced_query_output.reshape(B, F * K, D)
            
            visual_embeds = model.language_projection(reduced_query_output)
            text_embeds = model.language_model.get_input_embeddings()(batch['input_ids'])
            
            lm_inputs = torch.cat([visual_embeds, text_embeds], dim=1)
            lm_attention_mask = torch.cat([
                torch.ones((batch_size, visual_embeds.size(1)), dtype=torch.long, device=visual_embeds.device),
                batch['attention_mask']
            ], dim=1)
            
            outputs = model.language_model.generate(
                inputs_embeds=lm_inputs,
                attention_mask=lm_attention_mask,
                max_new_tokens=max_tokens,
                num_beams=beam_width,
                early_stopping=True
            )
            
            for idx, output in enumerate(outputs):
                decoded_pred = processor.decode(output, skip_special_tokens=True)
                #print(f"\nModel: {decoded_pred}")
                
                label = batch['labels'][idx].clone()
                label[label == -100] = processor.tokenizer.pad_token_id
                decoded_label = processor.decode(label, skip_special_tokens=True)
                #print(f"Label: {decoded_label}")
                
                gt_start, gt_end = interpret_time(decoded_label)
                pred_start, pred_end = interpret_time(decoded_pred)
                
                iou = None
                if all(t is not None for t in [gt_start, gt_end, pred_start, pred_end]):
                    iou = calculate_iou(pred_start, pred_end, gt_start, gt_end)
                    total_iou += iou
                    total_samples += 1
                    if iou >= 0.5: iou_over_05 += 1
                    if iou >= 0.7: iou_over_07 += 1
                
                results.append({
                    'prediction': decoded_pred,
                    'ground_truth': decoded_label,
                    'iou': iou
                })
            
            if total_samples > 0:
                progress_bar.set_postfix({
                    'avg_iou': f"{total_iou/total_samples:.4f}",
                    'iou_over_05': f"{(iou_over_05/total_samples)*100:.2f}%",
                    'iou_over_07': f"{(iou_over_07/total_samples)*100:.2f}%"
                })
    
    avg_iou = total_iou / total_samples if total_samples else 0.0
    iou_over_05_percent = (iou_over_05 / total_samples) * 100 if total_samples else 0.0
    iou_over_07_percent = (iou_over_07 / total_samples) * 100 if total_samples else 0.0
    
    if output_json_path:
        output_data = {
            "results": results,
            "summary": {
                "avg_iou": avg_iou,
                "iou_over_05_percent": iou_over_05_percent,
                "iou_over_07_percent": iou_over_07_percent
            }
        }
        with open(output_json_path, 'w') as json_file:
            json.dump(output_data, json_file, indent=4)
    
    return results, avg_iou, iou_over_05_percent, iou_over_07_percent