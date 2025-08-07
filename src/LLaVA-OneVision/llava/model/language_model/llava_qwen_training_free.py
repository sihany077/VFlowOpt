from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from collections import defaultdict
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from sklearn.cluster import KMeans
# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig
import re
from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import math
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
import random
import torch.nn.functional as F
import einops
from . import clip
from torchvision.transforms import Resize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    import Image
    BICUBIC = Image.BICUBIC

VIT_ATTN_MAP = None
import os
import json


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor

def unpad_image_and_attn_map(image_feature, image_attn_map, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image and its corresponding attention map.

    Args:
    image_feature (torch.Tensor): The image feature tensor, assumed to be in CxHxW format.
    image_attn_map (torch.Tensor): The attention map tensor, assumed to be in HxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    tuple: A tuple of (unpadded_image_feature, unpadded_image_attn_map) tensors.
    """
    original_width, original_height = original_size
    current_height, current_width = image_feature.shape[1:]  # H, W of the image feature

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_image_feature = image_feature[:, padding : current_height - padding, :]
        unpadded_image_attn_map = image_attn_map[padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_image_feature = image_feature[:, :, padding : current_width - padding]
        unpadded_image_attn_map = image_attn_map[:, padding : current_width - padding]

    return unpadded_image_feature, unpadded_image_attn_map


def unpad_image_and_attn_map_get_global_imp_adjed_attn(
    image_feature, image_attn_map, original_size, base_image_attn_map, num_patch_width, num_patch_height, i, alpha):
    image_attn_map = image_attn_map.to(torch.float32)
    base_image_attn_map = base_image_attn_map.to(torch.float32)

    i = i
    alpha = alpha

    mean_attn = base_image_attn_map.mean()
    global_token_indice = torch.nonzero(base_image_attn_map > 5 * mean_attn, as_tuple=True)[0]
    reshaped_base_image_attn_map = base_image_attn_map.clone()
    reshaped_base_image_attn_map[global_token_indice] = 0

    mean_no_global_attn = reshaped_base_image_attn_map.mean()
    alpha *= mean_no_global_attn

    reshaped_base_image_attn_map = reshaped_base_image_attn_map.view(27, 27)

    original_width, original_height = original_size

    base_height, base_width = reshaped_base_image_attn_map.shape
    if original_height > original_width:
        new_height = base_height
        new_width = int(base_width * (original_width / original_height))
        width_diff = base_width - new_width
        left_pad = width_diff // 2
        right_pad = width_diff - left_pad
        reshaped_base_image_attn_map[:, :left_pad] = 0
        reshaped_base_image_attn_map[:, -right_pad:] = 0
    else:
        new_width = base_width
        new_height = int(base_height * (original_height / original_width))
        height_diff = base_height - new_height
        top_pad = height_diff // 2
        bottom_pad = height_diff - top_pad
        reshaped_base_image_attn_map[:top_pad, :] = 0
        reshaped_base_image_attn_map[-bottom_pad:, :] = 0

    tile_height = base_height // num_patch_height
    tile_width = base_width // num_patch_width

    mean_values = []
    for i in range(num_patch_height):
        for j in range(num_patch_width):
            start_y = i * tile_height
            end_y = (i + 1) * tile_height
            start_x = j * tile_width
            end_x = (j + 1) * tile_width

            tile_attn = reshaped_base_image_attn_map[start_y:end_y, start_x:end_x]
            non_zero_values = tile_attn[tile_attn != 0]
            non_zero_values_mean = non_zero_values.mean()
            mean_values.append(non_zero_values_mean)

    mean_values.append(mean_no_global_attn)

    mean_values = torch.stack(mean_values) 
    softmax_values = torch.softmax(mean_values, dim=0) 

    softmax_patch_values = softmax_values[:-1] 
    softmax_mean_attn_value = softmax_values[-1] 

    for i in range(num_patch_height):
        for j in range(num_patch_width):
            region_start_y = i * 27
            region_end_y = (i + 1) * 27
            region_start_x = j * 27
            region_end_x = (j + 1) * 27

            index = i * num_patch_width + j
            softmax_value = softmax_patch_values[index]

            image_attn_map[region_start_y:region_end_y, region_start_x:region_end_x] += alpha*softmax_value

    base_image_attn_map += alpha * softmax_mean_attn_value



    current_height, current_width = image_feature.shape[1:]  # H, W of the image feature

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_image_feature = image_feature[:, padding : current_height - padding, :]
        unpadded_image_attn_map = image_attn_map[padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_image_feature = image_feature[:, :, padding : current_width - padding]
        unpadded_image_attn_map = image_attn_map[:, padding : current_width - padding]
    
    return unpadded_image_feature, unpadded_image_attn_map, base_image_attn_map



class LlavaQwenTrainingFreeConfig(Qwen2Config):
    model_type = "llava_qwen_training_free"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenTrainingFreeConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)
    
class LlavaQwenTrainingFreeForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenTrainingFreeConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen_training_free"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()
        self.resize = Resize(224, interpolation=BICUBIC)
    
    def get_model(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        illava_config=None,
        source_indice_list=None,
        raw_frames=None,
        image_token_length=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, _) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)
        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            if illava_config["enable_illava_llm"]:
                return super().forward_illava(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                illava_config=illava_config,
                source_indice_list=source_indice_list,
                raw_frames=raw_frames,
                image_token_length=image_token_length,
            )
            else:
                return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        illava_config= None,
        questions_only= None, 
        raw_frames = None, 
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _, source_indice_list, image_token_length) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes, questions_only=questions_only, raw_frames=raw_frames, illava_config=illava_config)
        else:
            # we don't perform illava
            inputs_embeds = self.get_model().embed_tokens(inputs)
            source_indice_list = None
            image_token_length = 1

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, illava_config=illava_config, source_indice_list=source_indice_list, raw_frames=raw_frames if illava_config != None and illava_config['illava_track_llm_source'] else None, image_token_length=image_token_length, **kwargs)
        
    def encode_images(self, images, selected_indice=None, Layer_index=2, reduce_tokens=25, t=7.0, alpha_v=0.0, track_source=False, raw_frames=None):

        image_features, attn_maps = self.get_model().get_vision_tower()(images, selected_indice, Layer_index, reduce_tokens, t, alpha_v, track_source, raw_frames)

        # image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features, attn_maps[0]
    
    def get_Spatial_Pool(self, image_feature, pooling_stride=2, new_h=None, new_w=None):
        if new_h != None and new_w != None:
            height = new_h
            width = new_w
        else:
            height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, pooling_stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, pooling_stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, weight = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / pooling_stride), math.ceil(weight / pooling_stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')
        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature
    
    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None, questions_only=None, raw_frames=None, illava_config=None):
        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None
            
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            new_h = new_w = None
            #  Perform self selection     attn_maps->list; attn_maps[0].shape->torch.Size([729])
            if illava_config['enable_illava_vit']:
                concat_images = torch.cat([image for image in images_list], dim=0)  # lists of [32, 3, 384, 384]
                split_sizes = [image.shape[0] for image in images_list]
                encoded_image_features, attn_maps = self.encode_images(concat_images, None, Layer_index=illava_config['illava_vit_k'], reduce_tokens=0, t=illava_config['illava_vit_t'], alpha_v=illava_config['illava_vit_alpha_v'], track_source=illava_config['illava_track_vit_source'], raw_frames=raw_frames[0] if illava_config['illava_track_vit_source'] else None) #TNC 
                image_attn_map = torch.stack(attn_maps)
            else:
                image_tokens = 729
                concat_images = torch.cat([image for image in images_list], dim=0)
                split_sizes = [image.shape[0] for image in images_list]
                encoded_image_features, _ = self.encode_images(concat_images, None, None, 0, False, None) #TNC  
                source_indice_list = [[ torch.arange(image_tokens) for _ in range(len(image_list_i))] for image_list_i in images_list]
                h_cur = w_cur = int(math.sqrt(image_tokens))
                for i in range(len(source_indice_list)):
                    source_indice_list[i].append((h_cur, w_cur))

            # This is a list, each element is [num_images, patch * patch, dim]
            # rank_print(f"Concat images : {concat_images.shape}")
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            image_features = []
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    image_features.append(self.get_Spatial_Pool(image_feat, pooling_stride=self.config.mm_spatial_pool_stride, new_h=new_h, new_w=new_w))
                    if illava_config['enable_illava_vit']:
                        image_attn_map = self.get_Spatial_Pool(image_attn_map.unsqueeze(-1), pooling_stride=self.config.mm_spatial_pool_stride, new_h=new_h, new_w=new_w).squeeze(-1)
                else:
                    image_features.append(image_feat)
            # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
            # image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            # print(f"image_features:{image_features[0].shape}") # [7, 361, 3584]

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if "unpad" in mm_patch_merge_type:
                            # image_feature = image_feature.permute(2, 0, 1).contiguous()
                            # image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            # image_feature = image_feature.permute(1, 2, 0).contiguous()
                            image_feature = image_feature.flatten(0, 1)
                            image_feature = torch.cat((image_feature, self.model.image_newline[None].to(image_feature.device)), dim=0)
                            image_token_length = image_feature.shape[0]
                            if illava_config['enable_illava_vit']:
                                image_attn_map = image_attn_map.flatten(0, 1)
                                image_attn_map = torch.cat((image_attn_map, torch.ones(1).to(image_attn_map.device)), dim=0)

                                reduce_tokens = round(image_token_length * (1-illava_config['illava_vit_r']))
                                image_token_length = image_token_length - reduce_tokens

                                mask = torch.zeros_like(image_attn_map, dtype=torch.bool)  # 初始化为全False


                                topk_indices = torch.topk(image_attn_map, k=image_token_length).indices
                                
                                mask[topk_indices] = True

                                image_feature = image_feature[mask]  
                                image_attn_map = image_attn_map[mask]

                                os.environ["VIT_ATTN_MAP"] = json.dumps(image_attn_map.tolist())
                            image_token_length = image_feature.shape[0]
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]

                        if illava_config['enable_illava_vit']:
                            attn_maps = torch.stack(attn_maps, dim=0)
                            base_image_attn_map = attn_maps[0]
                            image_attn_map =attn_maps[1:]

                        if new_h is None and new_w is None:
                            height = width = self.get_vision_tower().num_patches_per_side
                        else:
                            height = new_h
                            width = new_w
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))
                            # print(f"max_num_patches:{max_num_patches}") # 9

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            if illava_config['enable_illava_vit']:
                                image_attn_map = image_attn_map.view(num_patch_height, num_patch_width, height, width)

                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2] # undergo this for single-images
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)

                            if illava_config['enable_illava_vit']:
                                image_attn_map = image_attn_map.permute(0, 2, 1, 3).contiguous()
                                image_attn_map = image_attn_map.flatten(0, 1).flatten(1, 2)

                                # NOTE to change
                                # print("vision_zip")
                                image_feature, image_attn_map = unpad_image_and_attn_map(image_feature, image_attn_map, image_sizes[image_idx])

                            else:  
                                image_feature = unpad_image(image_feature, image_sizes[image_idx])


                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                target_h, target_w = int(h // times), int(w // times)
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [target_h, target_w], mode="bilinear")[0]

                                if illava_config['enable_illava_vit']:
                                    image_attn_map = image_attn_map[None, None]  # 添加 batch 和 channel 维度
                                    image_attn_map = nn.functional.interpolate(image_attn_map, [target_h, target_w], mode="bilinear", align_corners=False)[0, 0]

                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)

                            if illava_config['enable_illava_vit']:
                                image_attn_map = torch.cat((image_attn_map, torch.ones(image_attn_map.shape[0], 1).to(image_attn_map.device)), dim=-1)
                                # local_image_attn_map = image_attn_map.clone()
                                local_w = image_attn_map.shape[1]
                                image_attn_map = image_attn_map.flatten(0, 1)

                        elif "unpad" in mm_patch_merge_type: # undergo this for multi-images
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                            if illava_config['enable_illava_vit']: 
                                image_attn_map = torch.cat((base_image_attn_map, image_attn_map), dim=0)
                                image_token_length = image_feature.shape[0]
                                # print(image_token_length)
                                reduce_tokens = round(image_token_length * (1-illava_config['illava_vit_r']))

                        # --------------token merging----------------
                            if illava_config['enable_illava_vit']:

                                a = illava_config['illava_vit_m'] 
                                num_merged_tokens = int(image_token_length/(a*a))
                                
                                sorted_indices = torch.sort(image_attn_map, dim=-1).indices  
                                low_k = sorted_indices[:reduce_tokens+num_merged_tokens] 
                                to_keep_mask = torch.ones(image_attn_map.shape[0], dtype=torch.bool, device=image_attn_map.device)
                                to_keep_mask[low_k] = False 
                                # ============================================


                                base_mask = low_k < 729
                                base_indices = low_k[base_mask].cpu()             
                                image_indices = (low_k[~base_mask] - 729).cpu()     

                                base_points = torch.stack([base_indices // 27, base_indices % 27], dim=1)  
                                image_points = torch.stack([image_indices // local_w, image_indices % local_w], dim=1)  

                                base_grid_coords = (base_points.float() / a).floor().long()
                                image_grid_coords = (image_points.float() / a).floor().long()

                                def group_by_grid_torch(indices, grid_coords):
                                    _, inverse = torch.unique(grid_coords, dim=0, return_inverse=True)
                                    groups = []
                                    # for i in range(inverse.max().item() + 1):
                                    max_val = inverse.max().item() if inverse.numel() > 0 else -1
                                    for i in range(max_val + 1):
                                        group = indices[inverse == i]
                                        groups.append(group)
                                    return groups

                                base_groups = group_by_grid_torch(base_indices, base_grid_coords)
                                image_groups = group_by_grid_torch(image_indices, image_grid_coords)

                                low_k_groups = []
                                low_k_groups.extend(base_groups)
                                low_k_groups.extend([grp + 729 for grp in image_groups])

                            # ============================================

                                for i, group_indices in enumerate(low_k_groups):
                                    values_to_merge = image_feature[group_indices] 
                                    weights = image_attn_map[group_indices].float() 
                                    weights /= weights.sum()

                                    merged_feature = (weights[:, None] * values_to_merge).sum(dim=0) 
                                    max_weight_index = group_indices[image_attn_map[group_indices].argmax()]
                                    image_feature[max_weight_index] = merged_feature
                                    to_keep_mask[max_weight_index] = True



                                image_feature = image_feature[to_keep_mask]

                                image_attn_map = image_attn_map[to_keep_mask]
                                os.environ["VIT_ATTN_MAP"] = json.dumps(image_attn_map.tolist())
                        # ============================================================
                        

                        image_token_length = image_feature.shape[0]

                    else:  # single image operations
                        image_feature = image_feature[0]
                        image_token_length = image_feature.shape[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            if illava_config['enable_illava_vit']:
                image_features, attn_maps = self.encode_images(images, None, Layer_index=illava_config['illava_vit_k'], reduce_tokens=0, t=illava_config['illava_vit_t'], alpha_v=illava_config['illava_vit_alpha_v'], track_source=illava_config['illava_track_vit_source'], raw_frames=raw_frames[0] if illava_config['illava_track_vit_source'] else None)

                reduce_tokens_per_img = round(729 * (1-illava_config['illava_vit_r']))
                image_tokens_per_img = 729 - reduce_tokens_per_img
                image_token_length = image_tokens_per_img * len(image_features)
                attn_maps = torch.stack(attn_maps)

                batch_size, num_tokens = attn_maps.shape
                mask = torch.zeros_like(attn_maps, dtype=torch.bool)  # 初始化为全False

                for i in range(batch_size):
                    attn_map = attn_maps[i]
                    
                    topk_indices = torch.topk(attn_map, k=image_tokens_per_img).indices
                    
                    mask[i, topk_indices] = True

                selected_features_per_img = []
                for i in range(batch_size):
                    img_mask = mask[i]
                    img_features = image_features[i, img_mask] 
                    selected_features_per_img.append(img_features)

                image_features = torch.stack(selected_features_per_img, dim=0)
                image_attn_map = attn_maps[mask]
                os.environ["VIT_ATTN_MAP"] = json.dumps(image_attn_map.tolist())

            else:
                image_tokens = 729
                image_features, _ = self.encode_images(images, None, None, 0, False, None) #TNC  
                # source_indice_list = [[ torch.arange(image_tokens) for _ in range(len(images))] ]
                # h_cur = w_cur = int(math.sqrt(image_tokens))
                # for i in range(len(source_indice_list)):
                #     source_indice_list[i].append((h_cur, w_cur))
                image_token_length = image_tokens * len(image_features)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")
        # rank_print(f"Image_features shape : {image_features[0].shape}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, None, image_token_length

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen_training_free", LlavaQwenTrainingFreeConfig)
AutoModelForCausalLM.register(LlavaQwenTrainingFreeConfig, LlavaQwenTrainingFreeForCausalLM)





