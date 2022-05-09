from functools import partial

from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel
from transformers import DistilBertModel

from torch import nn
import torch
import torch.nn.functional as F


class ALBEF(nn.Module):
    def __init__(self, config, text_encoder, ckpt=None):
        super().__init__()

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        bert_config = BertConfig.from_json_file(config['bert_config'])

        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)

        D = self.text_encoder.config.hidden_size
        self.cls_head = nn.Sequential(
                nn.Linear(D, D),
                nn.ReLU(),
                nn.Linear(D, D))

        self.ans_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        D = self.ans_encoder.config.dim
        self.ans_head = nn.Sequential(
                nn.Linear(D, D),
                nn.ReLU(),
                nn.Linear(D, D))

        self.freeze_weights()

    def freeze_weights(self):
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.embeddings.parameters():
            param.requires_grad = False
        for param in self.ans_encoder.embeddings.parameters():
            param.requires_grad = False

    def forward(self, frames, text, choices, targets, nframe, alpha=0, train=True):
        frame_embeds = self.visual_encoder(frames)
        frame_atts = torch.ones(frame_embeds.size()[:-1],dtype=torch.long).to(frames.device)

        output = self.text_encoder(text.input_ids,
                                   attention_mask=text.attention_mask,
                                   encoder_hidden_states=frame_embeds,
                                   encoder_attention_mask=frame_atts)
        feat = self.cls_head(output[0][:, 0])
        feat = feat.view(feat.size(0)//nframe[0], nframe[0], -1)

        ans_output = self.ans_encoder(choices.input_ids,
                                      attention_mask=choices.attention_mask)
        ans_feat = self.ans_head(ans_output[0][:, 0])
        ans_feat = ans_feat.view(ans_feat.size(0)//5, 5, -1)

        logits = torch.bmm(feat, ans_feat.transpose(1, 2)).mean(dim=1)

        if train:
            loss = F.cross_entropy(logits, targets)
            return loss
        else:
            return logits

