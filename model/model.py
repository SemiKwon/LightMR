from utils.utils import *

class MambaTR(nn.Module):
    def __init__(self, d_model=768, q=32, k=16, f=16, d_state=128):
        super().__init__()
        self.q = q
        self.k = k
        self.f = f
        self.d_model = d_model

        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2, bias=False)
        self.norm = nn.LayerNorm(d_model)

        self.score = nn.Sequential(
            nn.Linear(4 * d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, 1)
        )

        self.attn_q = nn.Parameter(torch.randn(4, d_model))
        self.attn_lin = nn.Linear(d_model, d_model, bias=False)

        self.time_pos_emb = nn.Embedding(f, d_model)

    def forward(self, x):
        B, F, Q, D = x.shape
        
        x_original = x.clone()

        x = x.permute(0, 2, 1, 3).contiguous().view(B * Q, F, D)
        
        time_ids = torch.arange(F, device=x.device)
        time_emb = self.time_pos_emb(time_ids) 
        x = x + time_emb.unsqueeze(0)
        
        x = self.norm(x)
        x = self.mamba(x)
        
        k_all = self.attn_lin(x) 
        attn_score = torch.einsum('hd,bfd->bhf', self.attn_q, k_all)  
        attn_w = torch.softmax(attn_score, dim=-1) 
        x_head = torch.einsum('bhf,bfd->bhd', attn_w, x)
        
        x_summary = x_head.flatten(1)
        
        patch_scores = self.score(x_summary).squeeze(-1) 
        patch_scores = patch_scores.view(B, Q) 
        
        topk_idx = patch_scores.topk(self.k, dim=1).indices  
        topk_idx = topk_idx.sort(dim=1).values  
        
        batch_idx = torch.arange(B, device=x.device).unsqueeze(-1) 
        selected_tokens = x_original[batch_idx, :, topk_idx]
        
        return selected_tokens

class Blip2(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision_model = model.vision_model
        self.qformer = model.qformer
        self.language_model = model.language_model
        self.query_tokens = model.query_tokens

        self.mamba_tr = MambaTR()

        language_hidden_size = self.language_model.config.hidden_size
        self.language_projection = nn.Linear(768, language_hidden_size)

    def forward(self, pixel_values, input_ids, attention_mask, labels=None, **kwargs):
        batch_size, num_frames, channels, height, width = pixel_values.shape
        
        pixel_values = pixel_values.reshape(batch_size * num_frames, channels, height, width)

        vision_outputs = self.vision_model(pixel_values)
        frame_embeds = vision_outputs[0]
        
        frame_attention_mask = torch.ones(
            frame_embeds.size()[:-1],
            dtype=torch.long,
            device=frame_embeds.device
        )

        query_tokens = self.query_tokens.expand(frame_embeds.size(0), -1, -1)

        qformer_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=frame_embeds,
            encoder_attention_mask=frame_attention_mask,
        )
        query_output = qformer_outputs.last_hidden_state
        num_queries = query_output.size(1)
        query_output = query_output.view(batch_size, num_frames, num_queries, -1)
        
        reduced_query_output = self.mamba_tr(query_output)
        
        B, F, K, D = reduced_query_output.shape
        
        reduced_query_output = reduced_query_output.reshape(B, F * K, D)
        
        visual_embeds = self.language_projection(reduced_query_output)
        text_embeds = self.language_model.get_input_embeddings()(input_ids)

        lm_inputs = torch.cat([visual_embeds, text_embeds], dim=1)
        lm_attention_mask = torch.cat([
            torch.ones((batch_size, visual_embeds.size(1)), dtype=torch.long, device=visual_embeds.device),
            attention_mask
        ], dim=1)

        if labels is not None:
            new_labels = -100 * torch.ones(
                (batch_size, lm_inputs.size(1)),
                dtype=labels.dtype,
                device=labels.device
            )

            num_visual_tokens = visual_embeds.size(1)
            seq_len = labels.size(1)

            max_text_len = lm_inputs.size(1) - num_visual_tokens
            seq_len = min(seq_len, max_text_len)

            new_labels[:, num_visual_tokens:num_visual_tokens + seq_len] = labels[:, :seq_len]
            labels = new_labels

            outputs = self.language_model(
                inputs_embeds=lm_inputs,
                attention_mask=lm_attention_mask,
                labels=labels
            )
            return outputs

        else:
            outputs = self.language_model(
                inputs_embeds=lm_inputs,
                attention_mask=lm_attention_mask
            )
            return outputs
        
def get_model():
    base_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xxl"
        )
    base_model = prepare_model_for_kbit_training(base_model)

    language_lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=[
            "q",
            "k",
            "v",
            "o",
            "wi_0",
            "wi_1",
            "wo"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )

    base_model.language_model = get_peft_model(
        base_model.language_model,
        language_lora_config
    )

    model = Blip2(base_model)
    model.requires_grad_(False)
    
    if hasattr(model, 'qformer'):
        model.qformer.requires_grad_(True)

    if hasattr(model, 'mamba_tr'):
        model.mamba_tr.requires_grad_(True)

    if hasattr(model, 'language_projection'):
        model.language_projection.requires_grad_(True)

    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True

    return model