import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models
 
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=256, rates=[6,12,18]):
        super().__init__()
        self.convs=nn.ModuleList([nn.Conv2d(in_ch,out_ch,1,bias=False)]+
            [nn.Conv2d(in_ch,out_ch,3,padding=r,dilation=r,bias=False) for r in rates])
        self.gap=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(in_ch,out_ch,1,bias=False))
        self.proj=nn.Conv2d(out_ch*(len(rates)+2),out_ch,1)
    def forward(self,x):
        sz=x.shape[2:]
        feats=[c(x) for c in self.convs]+[F.interpolate(self.gap(x),sz,mode='bilinear',align_corners=False)]
        return self.proj(torch.cat(feats,1))
 
class DeepLabV3(nn.Module):
    def __init__(self, n_cls=21):
        super().__init__()
        bb=models.resnet50(pretrained=False)
        self.enc=nn.Sequential(*list(bb.children())[:-2])
        self.aspp=ASPP(2048)
        self.head=nn.Sequential(nn.Conv2d(256,256,3,padding=1,bias=False),
                                  nn.BatchNorm2d(256),nn.ReLU(),nn.Conv2d(256,n_cls,1))
    def forward(self,x):
        sz=x.shape[2:]; f=self.aspp(self.enc(x))
        return F.interpolate(self.head(f),sz,mode='bilinear',align_corners=False)
 
model=DeepLabV3(21); x=torch.randn(2,3,256,256); out=model(x)
print(f"Input: {x.shape} → Segmentation: {out.shape}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
