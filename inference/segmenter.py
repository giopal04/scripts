from fastai.callback.hook import model_sizes
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.vision.learner import _update_first_layer
import timm

def retrieve_timm_model(model, pretrained=True, show_components=False):
    model = timm.create_model(model, pretrained)
    model_components = list(model.children())
    
    if show_components:
        for i, component in enumerate(model_components):
            print(f'{i}: {component}')
            
    return model, model_components

class TransformerEncoder(nn.Module):
    def __init__(self, model, cut=-3):
        super().__init__()
        self.encoder = self.create_encoder(model, cut)
    
    def forward(self, x):
        return self.encoder(x)
    
    def create_encoder(self, backbone, cut):
        _, backbone_components = retrieve_timm_model(backbone)
        return nn.Sequential(*backbone_components[:cut])
    
class MaskTransformer(nn.Module):
    def __init__(self, model, cls, img_dim, cut_in=4, cut_out=-3):
        super().__init__()
        self.backbone = model
        self.img_dim = img_dim
        self.embedding_dim = None
        self.cls = cls
        self.trasformer = self.create_decoder(model, cut_in, cut_out)
        self.cls_embedding = nn.Parameter(torch.rand((1, self.cls, self.embedding_dim)))
        
    def forward(self, x):
        B, _, _ = x.shape
        x = self.trasformer(x)
        x = F.normalize(x, dim=2)
        cls_emb = torch.transpose(self.trasformer(self.cls_embedding), 1, 2)
        x = torch.matmul(x, cls_emb)
        return x
    
    def create_decoder(self, backbone, cut_in, cut_out):
        _, backbone_components = retrieve_timm_model(backbone)
        sz = self.img_dim
        self.embedding_dim = model_sizes(nn.Sequential(*backbone_components), (sz,sz))[1][-1]
        return nn.Sequential(*backbone_components[cut_in:cut_out])
    
class UpsampleLayer(nn.Module):
    def __init__(self, img_dim, patch_size, cls):
        super().__init__()
        self.img_dim = img_dim
        self.patch_size = patch_size
        self.scales = self.scale_factors()
        self.cls = cls
        self.upsample1 = nn.Upsample(scale_factor=(self.scales[0], self.scales[0]), mode='bilinear')
        self.conv = nn.Conv2d(self.cls, self.cls, 1)
        self.upsample2 = nn.Upsample(scale_factor=(self.scales[1], self.scales[1]), mode='bilinear') 
    
    def forward(self, x):
        B, _, _ = x.shape
        x = x.permute(0,2,1)
        x = x.view(B, self.cls, int(self.img_dim/self.patch_size), int(self.img_dim/self.patch_size))
        x = self.upsample1(x)
        x = self.conv(x)
        x = self.upsample2(x)
        return x
    
    def scale_factors(self):
        sf = []
        sf.append(int(self.patch_size/4))
        sf.append(int(4))
        return sf

class Segmenter(nn.Module):#configuration (-3, 4, -3)
    def __init__(self, model, cls, img_dim, patch_size, cut_enc=-3, cut_in_dec=4, cut_out_dec=-3):
        super().__init__()
        self.backbone = model
        self.cls = cls
        self.img_dim = img_dim
        self.patch_size = patch_size
        
        self.encoder  = TransformerEncoder(self.backbone, cut_enc)
        self.decoder  = MaskTransformer(self.backbone, self.cls, self.img_dim, cut_in_dec, cut_out_dec)
        self.upsample = UpsampleLayer(self.img_dim, self.patch_size, self.cls)
        #self.segmenter = nn.Sequential(self.encoder, self.decoder, self.upsample)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.upsample(x)
        return x
