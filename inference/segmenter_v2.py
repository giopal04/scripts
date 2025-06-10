import numpy as np
from fastai.vision.all import PILImage, PILMask, Resize, Image, model_sizes
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import matplotlib.pyplot as plt

def PIL2tensor(x, is_img=True):
    if is_img:
        img = PILImage.create(x)
        return torch.tensor(np.array(img), dtype=torch.int32).permute(2,0,1)
    else:
        msk = PILMask.create(x)
        return torch.tensor(np.array(msk), dtype=torch.int32)

def prepare_image(image_list, img_dim):#requires a get_mask function which returns the Mask path
    k = np.random.randint(len(image_list))
    
    img = PILImage.create(image_list[k])
    tfm = Resize(img_dim)
    img = tfm(img)
    img_t = PIL2tensor(img)
    
    msk = PILMask.create(get_mask(image_list[k]))
    msk = tfm(msk)
    msk_t = PIL2tensor(msk, False)
    
    return img, img_t, msk, msk_t

def retrieve_timm_model(model, pretrained=True, show_components=False):
    model = timm.create_model(model, pretrained)
    model_components = list(model.children())
    
    if show_components:
        for i, component in enumerate(model_components):
            print(f'{i}: {component}')
            
    return model, model_components

def pass_through(x, arc):#only for timm model
    print(f'0: {x.shape}')
    for i, layer in enumerate(list(arc.children())):
        x = layer(x)
        print(f'{i+1}: {x.shape}')
        
def decode(x, codes):# codes is a dictionary that encodes the classes
    for i, code in enumerate(codes):
        if i == x: print(code)

def preds_probs_from_image(imgt, model, ):
    _, h, w = imgt.shape
    imgt = imgt.view(1,3,h,w)/255
    imgt = imgt.to(torch.device('cuda:0'))
    with torch.no_grad():
        preds = model(imgt)
    
    s_max = nn.Softmax2d()
    gray_masks = s_max(preds)*255
    
    return preds, gray_masks

def show_last_layer(img, output, cutoff=True):#img.shape = (c, h, w)
    plt.imshow(img.permute(1,2,0))
    plt.show()
    
    gray_msk = output[0].cpu()
    for i in range(32):
        plt.imshow(gray_msk[i], cmap='gray_r')
        plt.show()
        if cutoff and i == 5: break

def prediction_for_dataset(image_list, img_dim, learner, save_path):
    tfm = Resize(size=img_dim)
    
    msk_pred = {}
    for img in tqdm(image_list):
        img_1 = PILImage.create(img)
        img_1 = tfm(img_1)
        msk_t, _, _ = learner.predict(img_1)
        msk = Image.fromarray(np.array(msk_t).astype(np.uint8), mode='L')
        msk.save(save_path)

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
    def __init__(self, model, cls, img_dim, encoder_embedding, cut_in=4, cut_out=-3):
        super().__init__()
        self.decoder = model
        self.img_dim = img_dim
        self.encoder_embedding = encoder_embedding
        self.decoder_embedding = None # defined in the create_decoder method
        self.cls = cls

        self.trasformer = self.create_decoder(model, cut_in, cut_out)
        self.project = nn.Linear(self.encoder_embedding, self.decoder_embedding)
        
        self.cls_embedding = nn.Parameter(torch.rand((1, self.cls, self.decoder_embedding)))
        
    def forward(self, x):
        B, _, _ = x.shape
        x = self.project(x)
        x = self.trasformer(x)
        x = F.normalize(x, dim=2)
        cls_emb = torch.transpose(self.trasformer(self.cls_embedding), 1, 2)
        x = torch.matmul(x, cls_emb)
        return x
    
    def create_decoder(self, model, cut_in, cut_out):
        _, model_components = retrieve_timm_model(model, pretrained=False)
        sz = self.img_dim
        self.decoder_embedding = model_sizes(nn.Sequential(*model_components), (sz,sz))[1][-1]
        return nn.Sequential(*model_components[cut_in:cut_out])
    
        
    
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
        #sf.append(int(self.patch_size/4))
        #sf.append(int(4))
        if self.patch_size != 14:
            sf.append(int(self.patch_size/4))
            sf.append(int(4))
        else:
            sf.append(int(self.patch_size/2))
            sf.append(int(2))
        return sf

class Segmenter(nn.Module):#configuration (-3, 4, -3)
    def __init__(self, backbone, cls, img_dim, patch_size, decoder=None, cut_enc=-3, cut_in_dec=4, cut_out_dec=-3):
        super().__init__()
        self.backbone = backbone

        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = backbone

        self.cls = cls
        self.img_dim = img_dim
        self.patch_size = patch_size
        
        self.encoder = TransformerEncoder(self.backbone, cut_enc)
        encoder_embedding = model_sizes(nn.Sequential(*list(self.encoder.children())), (self.img_dim, self.img_dim))[-1][-1]

        self.decoder = MaskTransformer(self.decoder, cls=self.cls, img_dim=self.img_dim, encoder_embedding=encoder_embedding, cut_in=cut_in_dec, cut_out=cut_out_dec)
        self.upsample = UpsampleLayer(self.img_dim, self.patch_size, self.cls)
        #self.segmenter = nn.Sequential(self.encoder, self.decoder, self.upsample)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.upsample(x)
        return x
            
