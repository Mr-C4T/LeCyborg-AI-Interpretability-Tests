import numpy as np
import torch
import cv2
from typing import List, Dict, Tuple, Optional

class ACTPolicyWithAttention:
    """
    Wrapper for ACTPolicy that provides transformer attention visualizations and access to specific token attention.
    """

    def __init__(self, policy, image_shapes=None, specific_decoder_token_index: Optional[int] = None):
        self.policy = policy
        self.config = policy.config

        self.specific_decoder_token_index = specific_decoder_token_index
        if self.specific_decoder_token_index is not None:
            if not hasattr(self.config, 'chunk_size'):
                raise AttributeError("Policy's config object does not have 'chunk_size' attribute.")
            if not (0 <= self.specific_decoder_token_index < self.config.chunk_size):
                raise ValueError(
                    f"specific_decoder_token_index ({self.specific_decoder_token_index}) "
                    f"must be between 0 and chunk_size-1 ({self.config.chunk_size - 1})."
                )

        # Determine number of images from config
        if self.config.image_features:
            self.num_images = len(self.config.image_features)
        else:
            self.num_images = 0

        # Store image shapes if provided, otherwise will be detected at runtime
        self.image_shapes = image_shapes

        # For storing the last processed images and attention
        self.last_observation = None
        self.last_attention_maps = None

        if not hasattr(self.policy, 'model') or \
        not hasattr(self.policy.model, 'decoder') or \
        not hasattr(self.policy.model.decoder, 'layers') or \
        not self.policy.model.decoder.layers:
            raise AttributeError("Policy model structure does not match expected ACT architecture for target_layer.")
        self.target_layer = self.policy.model.decoder.layers[-1].multihead_attn

        # Build token mapping
        self.token_key_to_index = self._build_token_key_to_index()

    def _build_token_key_to_index(self):
        idx = 0
        mapping = {}
        mapping['latent'] = idx
        idx += 1
        if getattr(self.config, "robot_state_feature", None):
            mapping['observation.state'] = idx
            idx += 1
        # Always add observation.sensor (assuming your data always has it)
        mapping['observation.sensor'] = idx
        idx += 1
        if getattr(self.config, "env_state_feature", None):
            mapping['observation.env_state'] = idx
            idx += 1
        if hasattr(self.config, "image_features") and self.config.image_features:
            for image_key in self.config.image_features:
                mapping[image_key] = idx
                idx += 1
        print("Token mapping:", mapping)
        return mapping

    def select_action(self, observation: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        self.last_observation = observation.copy()
        images = self._extract_images(observation)
        image_spatial_shapes = self._get_image_spatial_shapes(images)
        attention_weights_capture = []

        def attention_hook(module, input_args, output_tuple):
            if isinstance(output_tuple, tuple) and len(output_tuple) > 1:
                attn_weights = output_tuple[1]
            else:
                attn_weights = getattr(module, 'attn_weights', None)
            if attn_weights is not None:
                attention_weights_capture.append(attn_weights.detach().cpu())

        handle = self.target_layer.register_forward_hook(attention_hook)
        with torch.inference_mode():
            action = self.policy.select_action(observation, force_model_run=True)
        handle.remove()

        if attention_weights_capture:
            attn = attention_weights_capture[0].to(action.device)
            attention_maps, proprio_attention = self._map_attention_to_images(attn, image_spatial_shapes)
            self.last_attention_maps = attention_maps
            self.last_proprio_attention = proprio_attention
            self.last_raw_attention = attn
        else:
            print("Warning: No attention weights were captured.")
            attention_maps = [None] * self.num_images
            self.last_attention_maps = attention_maps
            self.last_proprio_attention = 0.0
            self.last_raw_attention = None

        return action, attention_maps

    def _extract_images(self, observation: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        images = []
        if hasattr(self.config, "image_features"):
            for key in self.config.image_features:
                if key in observation:
                    images.append(observation[key])
        return images

    def _get_image_spatial_shapes(self, images: List[torch.Tensor]) -> List[Tuple[int, int]]:
        spatial_shapes = []
        for img_tensor in images:
            if img_tensor is None:
                spatial_shapes.append((0, 0))
                continue
            with torch.no_grad():
                if img_tensor.dim() == 3:
                    img_tensor_batched = img_tensor.unsqueeze(0)
                else:
                    img_tensor_batched = img_tensor
                img_tensor_batched = img_tensor_batched.to(next(self.policy.model.backbone.parameters()).device)
                feature_map_dict = self.policy.model.backbone(img_tensor_batched)
                feature_map = feature_map_dict["feature_map"]
                h, w = feature_map.shape[2], feature_map.shape[3]
                spatial_shapes.append((h, w))
        return spatial_shapes

    def _map_attention_to_images(self,
                                attention: torch.Tensor,
                                image_spatial_shapes: List[Tuple[int, int]]) -> Tuple[List[np.ndarray], float]:
        if attention.dim() == 4:
            attention = attention.mean(dim=1)
        elif attention.dim() != 3:
            raise ValueError(f"Unexpected attention dimension: {attention.shape}. Expected 3 or 4.")

        n_prefix_tokens = 1  # latent token
        proprio_token_idx = None
        sensor_token_idx = None
        if getattr(self.config, "robot_state_feature", None):
            proprio_token_idx = n_prefix_tokens
            n_prefix_tokens += 1
        # Use our mapping for sensor
        sensor_token_idx = self.token_key_to_index.get('observation.sensor', None)
        if getattr(self.config, "env_state_feature", None):
            n_prefix_tokens += 1

        proprio_attention = 0.0
        if proprio_token_idx is not None:
            if self.specific_decoder_token_index is not None:
                if 0 <= self.specific_decoder_token_index < attention.shape[1]:
                    proprio_attention_tensor = attention[:, self.specific_decoder_token_index, proprio_token_idx]
                else:
                    proprio_attention_tensor = attention[:, :, proprio_token_idx].mean(dim=1)
            else:
                proprio_attention_tensor = attention[:, :, proprio_token_idx].mean(dim=1)
            proprio_attention = proprio_attention_tensor[0].cpu().numpy().item()

        raw_numpy_attention_maps = []
        tokens_per_image = [h * w for h, w in image_spatial_shapes]
        current_src_token_idx = n_prefix_tokens
        for i, (h_feat, w_feat) in enumerate(image_spatial_shapes):
            if h_feat == 0 or w_feat == 0:
                raw_numpy_attention_maps.append(None)
                if tokens_per_image[i] > 0:
                    current_src_token_idx += tokens_per_image[i]
                continue
            num_img_tokens = tokens_per_image[i]
            start_idx = current_src_token_idx
            end_idx = start_idx + num_img_tokens
            current_src_token_idx = end_idx
            attention_to_img_features = attention[:, :, start_idx:end_idx]
            if self.specific_decoder_token_index is not None:
                if not (0 <= self.specific_decoder_token_index < attention_to_img_features.shape[1]):
                    print(f"Warning (map_attention): specific_decoder_token_index {self.specific_decoder_token_index} is out of bounds for actual tgt_len {attention_to_img_features.shape[1]}. Falling back to averaging.")
                    img_attn_tensor_for_map = attention_to_img_features.mean(dim=1)
                else:
                    img_attn_tensor_for_map = attention_to_img_features[:, self.specific_decoder_token_index, :]
            else:
                img_attn_tensor_for_map = attention_to_img_features.mean(dim=1)
            if img_attn_tensor_for_map.shape[0] > 1 and i == 0:
                print(f"Warning (map_attention): Batch size is {img_attn_tensor_for_map.shape[0]}. Processing first element for attention map.")
            if img_attn_tensor_for_map.shape[1] != num_img_tokens:
                print(f"Warning (map_attention): Mismatch in token count for image {i}. Expected {num_img_tokens}, got {img_attn_tensor_for_map.shape[1]}. Skipping map for this image.")
                raw_numpy_attention_maps.append(None)
                continue
            try:
                img_attn_map_1d_tensor = img_attn_tensor_for_map[0]
                img_attn_map_2d_tensor = img_attn_map_1d_tensor.reshape(h_feat, w_feat)
                raw_numpy_attention_maps.append(img_attn_map_2d_tensor.cpu().numpy())
            except RuntimeError as e:
                print(f"Error (map_attention): Reshaping attention for image {i}: {e}. Shape was {img_attn_tensor_for_map[0].shape}, target HxW: {h_feat}x{w_feat}. Num tokens: {num_img_tokens}. Skipping.")
                raw_numpy_attention_maps.append(None)
                continue

        global_min = float('inf')
        global_max = float('-inf')
        found_any_valid_map = False

        if proprio_attention is not None:
            if proprio_attention < global_min:
                global_min = proprio_attention
            if proprio_attention > global_max:
                global_max = proprio_attention
            found_any_valid_map = True

        for raw_map_np in raw_numpy_attention_maps:
            if raw_map_np is not None:
                current_min = raw_map_np.min()
                current_max = raw_map_np.max()
                if current_min < global_min:
                    global_min = current_min
                if current_max > global_max:
                    global_max = current_max
                found_any_valid_map = True

        if not found_any_valid_map:
            return raw_numpy_attention_maps, 0.0

        if global_min == float('inf') or global_max == float('-inf'):
            print("Warning (map_attention): Could not determine global min/max for attention. All maps might be invalid.")
            return [np.zeros_like(m, dtype=np.float32) if m is not None else None for m in raw_numpy_attention_maps], 0.0

        if global_max > global_min:
            normalized_proprio_attention = (proprio_attention - global_min) / (global_max - global_min)
        else:
            normalized_proprio_attention = 0.0

        final_normalized_attention_maps = []
        for raw_map_np in raw_numpy_attention_maps:
            if raw_map_np is None:
                final_normalized_attention_maps.append(None)
                continue
            if global_max > global_min:
                normalized_map = (raw_map_np - global_min) / (global_max - global_min)
            else:
                normalized_map = np.zeros_like(raw_map_np, dtype=np.float32)
            final_normalized_attention_maps.append(normalized_map)

        return final_normalized_attention_maps, normalized_proprio_attention

    def get_token_attention(self, attention_maps, observation, token_key: str):
        attn = getattr(self, "last_raw_attention", None)
        if attn is None:
            print(f"[WARN] No stored raw attention. Run select_action first.")
            return None
        while attn.dim() > 4:
            attn = attn[0]
        if attn.dim() == 4:
            attn_avg = attn.mean(dim=(0, 1))
        elif attn.dim() == 3:
            attn_avg = attn[0]
        else:
            print(f"[WARN] Unexpected attention dim: {attn.shape}")
            return None
        src_token_idx = self.token_key_to_index.get(token_key)
        if src_token_idx is None:
            print(f"[WARN] token_key={token_key} not mapped to any index in token_key_to_index.")
            return None
        attention_vec = attn_avg[:, src_token_idx]
        return attention_vec.detach().cpu().numpy()

    def visualize_attention(self,
                           images: Optional[List[torch.Tensor]] = None,
                           attention_maps: Optional[List[np.ndarray]] = None,
                           observation: Optional[Dict[str, torch.Tensor]] = None,
                           use_rgb: bool = False,
                           overlay_alpha: float = 0.5,
                           show_proprio_border: bool = True,
                           proprio_border_width: int = 15) -> List[np.ndarray]:
        if images is None:
            if observation is not None:
                images = self._extract_images(observation)
            elif self.last_observation is not None:
                images = self._extract_images(self.last_observation)
            else:
                raise ValueError("No images provided and no stored observation available")
        if attention_maps is None:
            if self.last_attention_maps is not None:
                attention_maps = self.last_attention_maps
            else:
                raise ValueError("No attention maps provided and no stored attention maps available")
        proprio_attention = getattr(self, 'last_proprio_attention', 0.0)
        visualizations = []
        for i, (img, attn_map) in enumerate(zip(images, attention_maps)):
            if img is None or attn_map is None:
                visualizations.append(None)
                continue
            if isinstance(img, torch.Tensor):
                if img.dim() == 4:
                    img = img.squeeze(0)
                img_np = img.permute(1, 2, 0).cpu().numpy()
                if img_np.max() > 1.0:
                    img_np = img_np / 255.0
            else:
                img_np = img
            h, w = img_np.shape[:2]
            attn_map_resized = cv2.resize(attn_map, (w, h))
            heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_resized), cv2.COLORMAP_JET)
            if use_rgb:
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            vis = cv2.addWeighted(
                np.uint8(255 * img_np), 1 - overlay_alpha,
                heatmap, overlay_alpha, 0
            )
            if show_proprio_border and proprio_attention > 0:
                border_intensity = int(255 * proprio_attention)
                if use_rgb:
                    border_color = (border_intensity, 0, border_intensity)
                else:
                    border_color = (border_intensity, 0, border_intensity)
                cv2.rectangle(vis, (0, 0), (w-1, h-1), border_color, proprio_border_width)
                text = f"Proprio: {proprio_attention:.3f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                cv2.rectangle(vis, (5, 5), (5 + text_width + 10, 5 + text_height + 10), (0, 0, 0), -1)
                cv2.putText(vis, text, (10, 5 + text_height), font, font_scale, (255, 255, 255), thickness)
            visualizations.append(vis)
        return visualizations

    def __getattr__(self, name):
        if name not in self.__dict__:
            return getattr(self.policy, name)
        return self.__dict__[name]
