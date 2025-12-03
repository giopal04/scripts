"""
File containg some fastai callbacks implemented

1. CutMixSegmentation
    Implements cut mix for segmentation
2. ShowLossesMetricsCallback
    Shows metrics and losses chart at the end of training

"""

from fastai.vision.all import *
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
    
    for i, idx in enumerate(patch_idxs):
        img = imgs[i]
        
        if img.shape[0] == 3:
            img[:,yl[i]:yh[i],xl[i]:xh[i]] = patch_list[idx]        
        else:
            img[yl[i]:yh[i],xl[i]:xh[i]] = patch_list[idx]        
        
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

class CutMixSegmentation(Callback):
    run_after,run_valid = [Normalize],False
    
    def __init__(self, lam=None): self.lam = lam
        
    def before_batch(self):
        cutmix_batch = (self.xb[0], self.yb[0])
        
        cutmix_segmentation(cutmix_batch, lam=self.lam)
        
        return

class ShowLossesMetricsCallback(Callback):
    order,run_valid=65,False

    def before_fit(self):
        self.run = not hasattr(self.learn, 'lr_finder') and not hasattr(self, "gather_preds")
        if not(self.run): return
        self.nb_batches = []
        assert hasattr(self.learn, 'progress')

        self.metrics_names = [m.name for m in self.learn.metrics]
        self.fig, self.axs = plt.subplots(1,2, figsize=(12,4))
        
        self.val_losses, self.metrics = [], []

    def after_train(self): self.nb_batches.append(self.train_iter)
    
    def update_graphs(self):

        self.axs[0].clear()
        self.axs[0].set_title("Losses")
        self.axs[0].plot(range_of(self.learn.recorder.losses), self.learn.recorder.losses, label='train')
        self.axs[0].plot(self.nb_batches, self.val_losses, label='valid')
        self.axs[0].legend()

        self.axs[1].clear()
        self.axs[1].set_title("Metrics")
        for name, metric in zip(self.metrics_names, self.metrics):
            self.axs[1].plot(self.nb_batches, metric, label=name)
        self.axs[1].legend()

        self.fig.canvas.draw()
        #display(self.fig)

    def after_fit(self):
        rec = self.learn.recorder
        self.val_losses = [v[1] for v in rec.values]
        self.metrics = np.array([v[2:] for v in rec.values]).T
        self.update_graphs()
