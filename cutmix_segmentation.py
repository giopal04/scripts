#%%
from fastai.vision.all import * #type: ignore

device = torch.device('cuda:0')
#%%

path = Path('/home/giorgio/.fastai/data/camvid')

valid_images = np.loadtxt(path/'valid.txt', dtype=str)
codes = list(np.loadtxt(path/'codes.txt', dtype=str))

def get_mask(x): return path/'labels'/f'{x.stem}_P.png'

def func_splitter(x): return True if f'{x.stem}.png' in valid_images else False

images = get_image_files(path/'images')
msk = PILMask.create(get_mask(images[0]))
sz = msk.shape #type: ignore
eighth = tuple(int(x/8) for x in sz)
#%%
def image_loader(path, classes, size, bs, test=False):
    
    camvid = DataBlock(
        blocks = (ImageBlock, MaskBlock(classes)),
        get_items = get_image_files,
        splitter = FuncSplitter(func_splitter),
        get_y = get_mask,
        batch_tfms = [Normalize.from_stats(*imagenet_stats)]
    )
    
    if test:
        camvid.summary(path)
    
    loaded_data = camvid.dataloaders(path, bs=bs)
    loaded_data.vocab = classes 
    
    return loaded_data

#
#%%
class TestCallback(Callback):    
    def after_epoch(self):
        print('Finished')
#%%
dls = image_loader(path/'images', codes, eighth, 4)
#learn = unet_learner(dls, resnet18, cbs=TestCallback())
#learn.fit(2, 1e-3)
# %%
import random as rnd

def PIL2tensor(x, is_img=True):
    if is_img:
        img = PILImage.create(x)
        return tensor(np.array(img), dtype=torch.int32).permute(2,0,1)
    else:
        msk = PILMask.create(x)
        return tensor(np.array(msk), dtype=torch.int32)

def my_rand_bbox(img_shape, lam=None, count=None):
    if lam == None:
        lam = round(rnd.uniform(0.5, 0.9), 2)
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    yl = np.random.randint(0, img_h - cut_h, size=count)
    xl = np.random.randint(0, img_w - cut_w, size=count)
    yh = yl + cut_h
    xh = xl + cut_w
    
    return yl, yh, xl, xh

def sattalo_shuffle(x):
    order = list(range(x))
    deranged = order.copy()
    n = len(deranged)
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i - 1)
        deranged[i], deranged[j] = deranged[j], deranged[i]
    return deranged

def cutmix_function(imgs, bb, patch_idxs):
    
    bs = imgs.shape[0]
    yl, yh, xl, xh = bb
    
    patch_list = []
    for i in range(bs):
        img = imgs[i].clone()
        if img.shape[0] == 3:
            patch = img[:,yl[i]:yh[i],xl[i]:xh[i]]
            patch_list.append(patch)
        else:
            patch = img[yl[i]:yh[i],xl[i]:xh[i]]
            patch_list.append(patch)
    
    #patched_imgs = []
    for i, idx in enumerate(patch_idxs):
        #print(f'{i = }\t{idx = }')
        img = imgs[i]
        
        if img.shape[0] == 3:
            img[:,yl[i]:yh[i],xl[i]:xh[i]] = patch_list[idx]        
            #patched_imgs.append(img.clone())
        else:
            img[yl[i]:yh[i],xl[i]:xh[i]] = patch_list[idx]        
            #patched_imgs.append(img.clone())
        

def cutmix_segmentation(batch, lam=None):
    
    imgs, msks = batch
    img_shape = imgs.shape[1:]
    bs = imgs.shape[0]
    
    bb = my_rand_bbox(img_shape, lam, count=bs)
    patch_idxs = sattalo_shuffle(bs)
    
    cutmix_function(imgs, bb, patch_idxs)
    cutmix_function(msks, bb, patch_idxs)
    
    return batch

def show_cutmix(dls):
    batch = dls.train.one_batch()
    dls.show_batch(batch)
    
    new_batch = cutmix_segmentation(batch)
    dls.show_batch(new_batch)

# %%
class CutMixSegmentation(Callback):
    run_after,run_valid = [Normalize],False
    
    def __init__(self, lam=None): self.lam = lam
        
    def before_batch(self):
        cutmix_batch = (self.xb[0], self.yb[0])
        
        cutmix_segmentation(cutmix_batch, lam=self.lam)
        
        return
# %%
learn = unet_learner(dls, resnet18, cbs=CutMixSegmentation())
learn.fit(1, 1e-3)
# %%
