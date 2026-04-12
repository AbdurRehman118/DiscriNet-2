import torch
import torch.nn as nn
from typing import Iterable
from peft import LoraConfig, get_peft_model


class FusionHead(nn.Module):
	"""
	Tiny bi-transformer encoder followed by gated fusion with the global image vector.
	"""
	def __init__(self, d_model: int = 256, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1) -> None:
		super().__init__()
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=nhead,
			dim_feedforward=4 * d_model,
			dropout=dropout,
			batch_first=True,
			activation="gelu",
			norm_first=True,
		)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.gate = nn.Sequential(
			nn.Linear(2 * d_model, d_model),
			nn.SiLU(),
			nn.Linear(d_model, 1),
			nn.Sigmoid(),
		)
		self.cls = nn.Sequential(
			nn.LayerNorm(d_model),
			nn.Dropout(dropout),
			nn.Linear(d_model, 1),
		)

	def forward(self, token_sequence: torch.Tensor, image_token: torch.Tensor, image_vector: torch.Tensor) -> torch.Tensor:
		# token_sequence: [B, T, H], image_token: [B, 1, H], image_vector: [B, H]
		x = torch.cat([image_token, token_sequence], dim=1)  # [B, T+1, H]
		x = self.encoder(x)
		pooled = x[:, 0]  # first token
		z = torch.cat([pooled, image_vector], dim=-1)
		g = self.gate(z)
		fused = g * pooled + (1.0 - g) * image_vector
		logit = self.cls(fused).squeeze(-1)
		return logit


class MMBTCLIP(nn.Module):
	def __init__(self, clip_model, proj_dim: int = 256, use_lora: bool = False) -> None:
		super().__init__()
		self.clip = clip_model
		self.use_lora = use_lora

		# Determine projection input dims from CLIP
		# For HF CLIP, image_embeds are of size projection_dim (e.g., 512 for ViT-B/32)
		image_vec_dim = getattr(self.clip.config, "projection_dim", None)
		if image_vec_dim is None and hasattr(self.clip, "visual"):
			# Fallback for alternative CLIP implementations
			image_vec_dim = getattr(self.clip.visual, "output_dim", None)
		if image_vec_dim is None:
			raise ValueError("Unable to infer CLIP image embedding dimension (projection_dim).")

		text_hidden = getattr(self.clip.config.text_config, "hidden_size", None)
		if text_hidden is None and hasattr(self.clip, "text_projection"):
			text_hidden = self.clip.text_projection.shape[0]
		if text_hidden is None:
			raise ValueError("Unable to infer CLIP text hidden dimension.")

		self.project_image_vector = nn.Linear(image_vec_dim, proj_dim)
		self.project_text_tokens = nn.Linear(text_hidden, proj_dim)
		self.make_image_token = nn.Linear(image_vec_dim, proj_dim)
		self.fusion = FusionHead(d_model=proj_dim, nhead=4, num_layers=2, dropout=0.1)

		if self.use_lora:
			cfg = LoraConfig(
				r=8,
				lora_alpha=16,
				target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
				lora_dropout=0.05,
				bias="none",
			)
			self.clip = get_peft_model(self.clip, cfg)

		# Start with everything frozen; training script will enable LoRA/base later
		self.configure_trainable(enable_lora=False, unfreeze_clip=False)

	def configure_trainable(self, enable_lora: bool, unfreeze_clip: bool) -> None:
		"""
		Control trainable parameters:
		  - enable_lora=True: train LoRA params inside CLIP (if present)
		  - unfreeze_clip=True: also unfreeze base CLIP weights
		Projections and fusion head are always trainable.
		"""
		# First, freeze all CLIP params
		for p in self.clip.parameters():
			p.requires_grad = False
		# Enable LoRA params if requested
		if enable_lora and self.use_lora:
			for name, p in self.clip.named_parameters():
				if "lora_" in name:
					p.requires_grad = True
		# Optionally unfreeze base CLIP
		if unfreeze_clip:
			for name, p in self.clip.named_parameters():
				if not name.endswith(".bias") and "lora_" not in name:
					p.requires_grad = True
		# Head is always trainable
		for m in (self.project_image_vector, self.project_text_tokens, self.make_image_token, self.fusion):
			for p in m.parameters():
				p.requires_grad = True

	def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, pixel_values: torch.Tensor) -> torch.Tensor:
		# Use gradient only when any CLIP params are trainable
		clip_requires_grad = any(p.requires_grad for p in self.clip.parameters())
		with torch.set_grad_enabled(self.training and clip_requires_grad):
			out = self.clip(
				pixel_values=pixel_values,
				input_ids=input_ids,
				attention_mask=attention_mask,
				output_hidden_states=True,
				return_dict=True,
			)

		# image_embeds: [B, Dv]; last_hidden_state: [B, T, Dt]
		if hasattr(out, "image_embeds"):
			image_vec = out.image_embeds
		else:
			# fallback: pool vision hidden states if needed
			image_vec = out.vision_model_output.pooler_output

		if hasattr(out, "text_model_output") and hasattr(out.text_model_output, "last_hidden_state"):
			text_tokens = out.text_model_output.last_hidden_state
		elif hasattr(out, "text_embeds"):
			text_tokens = out.text_embeds.unsqueeze(1)
		else:
			text_tokens = out.last_hidden_state

		# Project and fuse
		image_token = self.make_image_token(image_vec).unsqueeze(1)  # [B,1,H]
		text_proj = self.project_text_tokens(text_tokens)  # [B,T,H]
		image_proj = self.project_image_vector(image_vec)  # [B,H]
		logit = self.fusion(text_proj, image_token, image_proj)
		return logit


