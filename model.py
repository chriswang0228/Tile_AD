from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import segmentation_models_pytorch_v2 as smp
from transformers import SegformerForSemanticSegmentation, Mask2FormerForUniversalSegmentation
from transformers.file_utils import ModelOutput

class RPL_CoroCL(nn.Module):
    def __init__(
        self,
        backbone='efficientnet-b6',
        ckpt=None
    ):
        super().__init__()
        model = smp.create_model(arch='DeepLabV3Plus',encoder_name=backbone, encoder_weights='imagenet',
                                              classes=13, activation=None, encoder_depth=5, decoder_channels=256)
        if ckpt is not None:
            model.load_state_dict(torch.load(ckpt))
        self.enc = model.encoder
        self.aspp = model.decoder.aspp
        self.up_aspp = model.decoder.up
        self.block1 = model.decoder.block1
        self.rpl = model.decoder.aspp
        self.up_rpl = model.decoder.up
        self.seg_head = nn.Sequential(
            model.decoder.block2,
            model.segmentation_head
        )
        self.corocl_head = nn.Sequential(
            nn.Conv2d(256, 304, kernel_size=1, bias=False),
            nn.BatchNorm2d(304),
            nn.ReLU(),
        )
        self.rpl_head = nn.Sequential(
            nn.Conv2d(256, 304, kernel_size=1, bias=False),
            nn.BatchNorm2d(304),
            nn.ReLU(),
        )

    def forward(self, x):
        features = self.enc(x)
        aspp_features = self.aspp(features[-1])
        aspp_features = self.up_aspp(aspp_features)
        high_res_features = self.block1(features[-4])
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        pseudo_label = self.seg_head(concat_features)
        
        rpl_features = self.rpl(features[-1])
        rpl_features = self.up_rpl(rpl_features)
        prediction = self.rpl_head(rpl_features)
        prediction = self.seg_head(concat_features+prediction)
        corocl_output = self.corocl_head(rpl_features)
        
        
        return pseudo_label, prediction, corocl_output
    
class DLV3_CoroCL(nn.Module):
    def __init__(
        self,
        backbone='efficientnet-b6',
        ckpt=None,
        classes=13,
    ):
        super().__init__()
        if 'mit' in backbone:
            model = smp.create_model(arch='DeepLabV3Plus',encoder_name=backbone, encoder_weights='imagenet',
                                              classes=classes, activation=None, encoder_output_stride=32, encoder_depth=5, decoder_channels=256)
        else:
            model = smp.create_model(arch='DeepLabV3Plus',encoder_name=backbone, encoder_weights='imagenet',
                                              classes=classes, activation=None, encoder_depth=5, decoder_channels=256)
        if ckpt is not None:
            model.load_state_dict(torch.load(ckpt))
        self.enc = model.encoder
        self.aspp = model.decoder.aspp
        self.up_aspp = model.decoder.up
        self.block1 = model.decoder.block1
        self.seg_head = nn.Sequential(
            model.decoder.block2,
            model.segmentation_head
        )
        self.corocl_head = nn.Sequential(
            nn.Conv2d(304, 304, kernel_size=1, bias=False),
            nn.BatchNorm2d(304),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=4.0),
        )

    def forward(self, x):
        features = self.enc(x)
        aspp_features = self.aspp(features[-1])
        aspp_features = self.up_aspp(aspp_features)
        high_res_features = self.block1(features[-4])
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        prediction = self.seg_head(concat_features)
        corocl_output = self.corocl_head(concat_features)
        
        
        
        return prediction, corocl_output

  
class MySegformer(nn.Module):
    def __init__(
        self,
        ckpt='nvidia/segformer-b3-finetuned-cityscapes-1024-1024'
    ):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(ckpt)

    def forward(self, x):
        out = self.model.segformer.encoder(x, output_hidden_states = True).hidden_states
        c1, c2, c3, c4 = out
        n, _, h, w = c4.shape
        _c4 = self.model.decode_head.linear_c[3](c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = nn.functional.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.model.decode_head.linear_c[2](c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = nn.functional.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.model.decode_head.linear_c[1](c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = nn.functional.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.model.decode_head.linear_c[0](c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        c = self.model.decode_head.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.model.decode_head.dropout(c)
        x = self.model.decode_head.classifier(x)
        
        return {'layer1':c1,
                'layer2':c2,
                'layer3':c3,
                'layer4':c4,
                'representation':c,
                'pred':x}
        
class SegformerMLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states
@dataclass 
class Mask2FormerModelOutput(ModelOutput):

    encoder_last_hidden_state: torch.FloatTensor = None
    pixel_decoder_last_hidden_state: torch.FloatTensor = None
    transformer_decoder_last_hidden_state: torch.FloatTensor = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_intermediate_states: Tuple[torch.FloatTensor] = None
    masks_queries_logits: Tuple[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
@dataclass
class Mask2FormerForUniversalSegmentationOutput(ModelOutput):
    
    loss: Optional[torch.FloatTensor] = None
    class_queries_logits: torch.FloatTensor = None
    masks_queries_logits: torch.FloatTensor = None
    auxiliary_logits: Optional[List[Dict[str, torch.FloatTensor]]] = None
    encoder_last_hidden_state: torch.FloatTensor = None
    pixel_decoder_last_hidden_state: torch.FloatTensor = None
    transformer_decoder_last_hidden_state: torch.FloatTensor = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_hidden_states: Optional[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
     
class Mask2former_CoroCL(nn.Module):
    def __init__(
        self,
        pretrain="facebook/mask2former-swin-large-cityscapes-semantic",
        classes=13,
        cl_pos='pixel_decoder'
    ):
        super().__init__()
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(pretrain, num_labels=classes, ignore_mismatched_sizes=True)
        self.cl_pos = cl_pos
        if self.cl_pos=='pixel_decoder':
            self.linear_c = nn.ModuleList([SegformerMLP(256, 768, bias=True) for i in range(3)])
            self.linear_fuse = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=8.0),
                nn.Conv2d(2304, 768, kernel_size=1, bias=False),
                nn.BatchNorm2d(768),
                nn.ReLU(),
                
            )
        elif self.cl_pos=='backbone':
            self.linear_c = nn.ModuleList([SegformerMLP(192, 768, bias=True),
                             SegformerMLP(384, 768, bias=True),
                             SegformerMLP(768, 768, bias=True),
                             SegformerMLP(1536, 768, bias=True),
                             ])
            self.linear_fuse = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=4.0),
                nn.Conv2d(3072, 768, kernel_size=1, bias=False),
                nn.BatchNorm2d(768),
                nn.ReLU(),
                
            )

    def forward(self, x):
        x1 = self.model(x)
        class_x = x1.class_queries_logits
        masks_x = x1.masks_queries_logits
        _b, _, h, w = x.shape
        masks_x = torch.nn.functional.interpolate(
            masks_x, size=(384, 384), mode="bilinear", align_corners=False
        )
        class_x = class_x.softmax(dim=-1)[..., :-1]
        masks_x = masks_x.sigmoid()  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        x1 = torch.einsum("bqc, bqhw -> bchw", class_x, masks_x)
        x1 = torch.nn.functional.interpolate(
            x1, size=(h, w), mode="bilinear", align_corners=False
        )
        if self.cl_pos=='pixel_decoder':
            c1, c2, c3 = self.model.base_model.pixel_level_module(x)['decoder_hidden_states']
            n, _, h, w = c3.shape

            _c1 = self.linear_c[2](c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
            _c1 = nn.functional.interpolate(_c1, size=c3.size()[2:],mode='bilinear',align_corners=False)

            _c2 = self.linear_c[1](c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
            _c2 = nn.functional.interpolate(_c2, size=c3.size()[2:],mode='bilinear',align_corners=False)

            _c3 = self.linear_c[0](c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])

            embed = self.linear_fuse(torch.cat([_c3, _c2, _c1], dim=1))
        elif self.cl_pos=='backbone':
            c1, c2, c3, c4 = self.model.base_model.pixel_level_module.encoder(x)['feature_maps']
            n, _, h, w = c4.shape

            _c4 = self.linear_c[3](c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
            _c4 = nn.functional.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

            _c3 = self.linear_c[2](c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
            _c3 = nn.functional.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

            _c2 = self.linear_c[1](c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
            _c2 = nn.functional.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

            _c1 = self.linear_c[0](c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

            embed = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        
        
        
        return x1, embed
    
    
class My_Mask2former_CoroCL(nn.Module):
    def __init__(
        self,
        pretrain="facebook/mask2former-swin-large-cityscapes-semantic",
        classes=13,
        cl_pos='pixel_decoder',
        cl_layer=3,
        tf_decoder_layers=10
    ):
        super().__init__()
        cfg = Mask2FormerForUniversalSegmentation.from_pretrained(pretrain, num_labels=classes, ignore_mismatched_sizes=True).config
        cfg.decoder_layers=tf_decoder_layers
        self.model = Mask2FormerForUniversalSegmentation(cfg)
        self.cl_pos = cl_pos
        self.cl_layer=cl_layer
        if self.cl_pos=='pixel_decoder':
            self.linear_c = nn.ModuleList([SegformerMLP(256, 256, bias=True) for i in range(self.cl_layer)])
            self.linear_fuse = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=8.0),
                nn.Conv2d(256*self.cl_layer, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                
            )
        elif self.cl_pos=='backbone':
            self.linear_c = nn.ModuleList([SegformerMLP(192, 768, bias=True),
                             SegformerMLP(384, 768, bias=True),
                             SegformerMLP(768, 768, bias=True),
                             SegformerMLP(1536, 768, bias=True),
                             ])
            self.linear_fuse = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=4.0),
                nn.Conv2d(3072, 768, kernel_size=1, bias=False),
                nn.BatchNorm2d(768),
                nn.ReLU(),
                
            )

    def forward(self, x):
        pixel_values = x
        pixel_mask = None,
        output_hidden_states = None,
        output_attentions = None,
        return_dict = None
        output_attentions = output_attentions if output_attentions is not None else self.model.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.model.model.config.use_return_dict

        batch_size, _, height, width = pixel_values.shape

        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=pixel_values.device)

        pixel_level_module_output = self.model.model.pixel_level_module(
            pixel_values=pixel_values, output_hidden_states=output_hidden_states
        )
        pixel_level_module_output.decoder_hidden_states = pixel_level_module_output.decoder_hidden_states[::-1]
        transformer_module_output = self.model.model.transformer_module(
                    multi_scale_features=pixel_level_module_output.decoder_hidden_states,
                    mask_features=pixel_level_module_output.decoder_last_hidden_state,
                    output_hidden_states=True,
                    output_attentions=output_attentions,
                )

        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None
        transformer_decoder_intermediate_states = None

        if output_hidden_states:
            encoder_hidden_states = pixel_level_module_output.encoder_hidden_states
            pixel_decoder_hidden_states = pixel_level_module_output.decoder_hidden_states
            transformer_decoder_hidden_states = transformer_module_output.hidden_states
            transformer_decoder_intermediate_states = transformer_module_output.intermediate_hidden_states
            
        output = Mask2FormerModelOutput(
            encoder_last_hidden_state=pixel_level_module_output.encoder_last_hidden_state,
            pixel_decoder_last_hidden_state=pixel_level_module_output.decoder_last_hidden_state,
            transformer_decoder_last_hidden_state=transformer_module_output.last_hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            pixel_decoder_hidden_states=pixel_decoder_hidden_states,
            transformer_decoder_hidden_states=transformer_decoder_hidden_states,
            transformer_decoder_intermediate_states=transformer_decoder_intermediate_states,
            attentions=transformer_module_output.attentions,
            masks_queries_logits=transformer_module_output.masks_queries_logits,
        )
        if not return_dict:
            output = tuple(v for v in output.values() if v is not None)
        class_queries_logits = ()
        for decoder_output in output.transformer_decoder_intermediate_states:
                    class_prediction = self.model.class_predictor(decoder_output.transpose(0, 1))
                    class_queries_logits += (class_prediction,)
        masks_queries_logits = output.masks_queries_logits
        output = Mask2FormerForUniversalSegmentationOutput(
            class_queries_logits=class_queries_logits[-1],
            masks_queries_logits=masks_queries_logits[-1],
            encoder_hidden_states=encoder_hidden_states,
            pixel_decoder_hidden_states=pixel_decoder_hidden_states,
            transformer_decoder_hidden_states=transformer_decoder_hidden_states,
        )
            
        class_x = output.class_queries_logits
        masks_x = output.masks_queries_logits
        _b, _, h, w = x.shape
        masks_x = torch.nn.functional.interpolate(
            masks_x, size=(384, 384), mode="bilinear", align_corners=False
        )
        class_x = class_x.softmax(dim=-1)[..., :-1]
        masks_x = masks_x.sigmoid()  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        x = torch.einsum("bqc, bqhw -> bchw", class_x, masks_x)
        x = torch.nn.functional.interpolate(
            x, size=(h, w), mode="bilinear", align_corners=False
        )


        if self.cl_pos=='pixel_decoder':
            hidden_states = pixel_level_module_output.decoder_hidden_states
            n, _, h, w = hidden_states[0].shape

            c_list = []
            for idx in range(self.cl_layer):
                _c = self.linear_c[idx](hidden_states[idx]).permute(0,2,1).reshape(n, -1, hidden_states[idx].shape[2], hidden_states[idx].shape[3])
                _c = nn.functional.interpolate(_c, size=hidden_states[0].size()[2:],mode='bilinear',align_corners=False)
                c_list.append(_c)

            embed = self.linear_fuse(torch.cat(c_list, dim=1))
        elif self.cl_pos=='backbone':
            c1, c2, c3, c4 = self.model.base_model.pixel_level_module.encoder(x)['feature_maps']
            n, _, h, w = c4.shape

            _c4 = self.linear_c[3](c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
            _c4 = nn.functional.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

            _c3 = self.linear_c[2](c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
            _c3 = nn.functional.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

            _c2 = self.linear_c[1](c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
            _c2 = nn.functional.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

            _c1 = self.linear_c[0](c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

            embed = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        
        
        
        return x, embed