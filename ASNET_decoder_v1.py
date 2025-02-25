
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple

def make_decoder(in_channels, mid_channels_1, mid_channels_2, out_channels, drop_rate, final_flag=False):
	decoder_layer = nn.Sequential(
			nn.Conv2d(in_channels,mid_channels_1,3,padding=1),
			nn.BatchNorm2d(mid_channels_1),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_channels_1,mid_channels_2,3,padding=1),
			nn.BatchNorm2d(mid_channels_2),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_channels_2,out_channels,3,padding=1),
			nn.Dropout2d(drop_rate),
			)
	if final_flag:
		decoder_layer.append(nn.Upsample(scale_factor=2, mode='bilinear'))
	else:
		decoder_layer.append(nn.BatchNorm2d(out_channels))
		decoder_layer.append(nn.ReLU(inplace=True))
		decoder_layer.append(nn.Upsample(scale_factor=2, mode='bilinear'))
	return decoder_layer


'''class PatchEmbedding(nn.Module):
	def __init__(self, img_size=256, patch_size=7, stride=4, in_chans=3, embed_dim=768):#224--768
		super().__init__()
		img_size = to_2tuple(img_size)
		patch_size = to_2tuple(patch_size)

		self.img_size = img_size
		self.patch_size = patch_size
		self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
		self.num_patches = self.H * self.W
		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
							  padding=(patch_size[0] // 2, patch_size[1] // 2))
		self.norm = nn.LayerNorm(embed_dim)


	def forward(self, x):
		x = self.proj(x)
		_, _, H, W = x.shape
		x = x.flatten(2).transpose(1, 2)
		x = self.norm(x)

		return x, H, W'''

class s2b_attention(nn.Module):
	def __init__(self,dim,num_heads=8,attn_drop_rate=0.,proj_drop_rate=0.):
		super().__init__()
		assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

		self.dim = dim
		self.num_heads = num_heads
		self.q = nn.Linear(dim, dim, bias=True)
		self.k = nn.Linear(dim, dim, bias=True)
		self.v = nn.Linear(dim, dim, bias=True)
		
		self.norm = nn.LayerNorm(dim)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop_rate)
		self.attn_drop = nn.Dropout(attn_drop_rate)
		
	def forward(self,x,mask,):
		B, N, C = x.shape
		x = self.norm(x)
		q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
		k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
		v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

		attn = (q @ k.transpose(-2, -1))
		attn = attn * self.Mask(mask)[:,None,:,:]
		attn = F.softmax(attn, dim=-1)
		attn = self.attn_drop(attn)

		x = (attn @ v).transpose(1, 2).reshape(B, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)

		return x

	def Mask(self,binary_mask):
		binary_mask[binary_mask<1] = 0
		binary_mask[binary_mask>0] = 1
		B, C, W, H = binary_mask.shape
		binary_mask_flat = binary_mask.reshape(B,C,H*W)[:,0,:]
		
		'''mask = torch.ones(B,H*W,H*W)
		for B_idx in range(B):
			for idx1 in range(H*W):
				for idx2 in range(H*W):
					#remove the influence from background to smoke
					if binary_mask[B_idx][idx1//W][idx1%W][0] == 0 & binary_mask[B_idx][idx2//W][idx2%W][0] == 1:
						mask[B_idx][idx1][idx2] = 0'''
		#binary_mask_flat = torch.tensor(binary_mask_flat)
		mask = ((binary_mask_flat.unsqueeze(1)==0) & (binary_mask_flat.unsqueeze(2)==1)) | \
			   ((binary_mask_flat.unsqueeze(1)==0) & (binary_mask_flat.unsqueeze(2)==0))
		mask = (~mask).float()

		return mask

class b2s_attention(nn.Module):
	def __init__(self,dim,num_heads=8,attn_drop_rate=0.,proj_drop_rate=0.):
		super().__init__()
		assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

		self.dim = dim
		self.num_heads = num_heads
		self.q = nn.Linear(dim, dim, bias=True)
		self.k = nn.Linear(dim, dim, bias=True)
		self.v = nn.Linear(dim, dim, bias=True)
		
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop_rate)
		self.attn_drop = nn.Dropout(attn_drop_rate)
		
	def forward(self,x,mask,):
		B, N, C = x.shape
		q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
		k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
		v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

		attn = (q @ k.transpose(-2, -1))
		attn = attn * self.Mask(mask)[:,None,:,:]
		attn = F.softmax(attn, dim=-1)
		attn = self.attn_drop(attn)

		x = (attn @ v).transpose(1, 2).reshape(B, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)

		return x

	def Mask(self,binary_mask):
		binary_mask[binary_mask<1] = 0
		binary_mask[binary_mask>0] = 1
		B, C, H, W = binary_mask.shape
		binary_mask_flat = binary_mask.reshape(B,C,H*W)[:,0,:]
		
		#remove the influence from  smoke to background
		#binary_mask_flat = torch.tensor(binary_mask_flat)
		mask = (binary_mask_flat.unsqueeze(1)==1) &( binary_mask_flat.unsqueeze(2)==0)
		mask = (~mask).float()

		return mask

class s2b_decoder(nn.Module):
	def __init__(self, embed_dims=[64, 128, 320, 512], depths=[2, 2, 3, 2],drop_rate=0.):#可炼丹处
		super().__init__()
		self.decoder0 = nn.ModuleList([
			s2b_attention(embed_dims[0],attn_drop_rate=drop_rate,proj_drop_rate=drop_rate)
			for i in range(depths[0])]
		)
		self.decoder1 = nn.ModuleList([
			s2b_attention(embed_dims[1],attn_drop_rate=drop_rate,proj_drop_rate=drop_rate)
			for i in range(depths[1])]
		)
		self.decoder2 = nn.ModuleList([
			s2b_attention(embed_dims[2],attn_drop_rate=drop_rate,proj_drop_rate=drop_rate)
			for i in range(depths[2])]
		)
		self.decoder3 = nn.ModuleList([
			s2b_attention(embed_dims[3],attn_drop_rate=drop_rate,proj_drop_rate=drop_rate)
			for i in range(depths[3])]
		)

	def flat(self,e):#B,N,C
		return e.permute(0,2,3,1).reshape(e.shape[0],e.shape[2]*e.shape[3],-1)
	def reflat(self,d,H,W):
		return d.permute(0,2,1).reshape(d.shape[0],-1,H,W)
	
	def forward(self,feats,mask):
		e0,e1,e2,e3 = feats#B,C,H,W
		d0 = self.flat(e0)
		d1 = self.flat(e1)
		d2 = self.flat(e2)
		d3 = self.flat(e3)

		s0 = F.interpolate(mask,to_2tuple(e0.shape[2]))
		s1 = F.interpolate(mask,to_2tuple(e1.shape[2]))
		s2 = F.interpolate(mask,to_2tuple(e2.shape[2]))
		s3 = F.interpolate(mask,to_2tuple(e3.shape[2]))
		for deco in self.decoder0:
			d0 = deco(d0,s0)
		d0 = self.reflat(d0,e0.shape[2],e0.shape[3])
		for deco in self.decoder1:
			d1 = deco(d1,s1)
		d1 = self.reflat(d1,e1.shape[2],e1.shape[3])
		for deco in self.decoder2:
			d2 = deco(d2,s2)
		d2 = self.reflat(d2,e2.shape[2],e2.shape[3])
		for deco in self.decoder3:
			d3 = deco(d3,s3)
		d3 = self.reflat(d3,e3.shape[2],e3.shape[3])
		 
		return d0,d1,d2,d3
	
class b2s_decoder(nn.Module):
	def __init__(self, embed_dims=[64, 128, 320, 512], depths=[2, 2, 3, 2],drop_rate=0.):#可炼丹处
		super().__init__()
		self.decoder0 = nn.ModuleList([
			b2s_attention(embed_dims[0],attn_drop_rate=drop_rate,proj_drop_rate=drop_rate)
			for i in range(depths[0])]
		)
		self.decoder1 = nn.ModuleList([
			b2s_attention(embed_dims[1],attn_drop_rate=drop_rate,proj_drop_rate=drop_rate)
			for i in range(depths[1])]
		)
		self.decoder2 = nn.ModuleList([
			b2s_attention(embed_dims[2],attn_drop_rate=drop_rate,proj_drop_rate=drop_rate)
			for i in range(depths[2])]
		)
		self.decoder3 = nn.ModuleList([
			b2s_attention(embed_dims[3],attn_drop_rate=drop_rate,proj_drop_rate=drop_rate)
			for i in range(depths[3])]
		)

	def flat(self,e):
		return e.permute(0,2,3,1).reshape(e.shape[0],e.shape[2]*e.shape[3],-1)
	
	def reflat(self,d,H,W):
		return d.permute(0,2,1).reshape(d.shape[0],-1,H,W)
	
	def forward(self,feats,mask):
		e0,e1,e2,e3 = feats#B,C,H,W
		d0 = self.flat(e0)
		d1 = self.flat(e1)
		d2 = self.flat(e2)
		d3 = self.flat(e3)

		s0 = F.interpolate(mask,to_2tuple(e0.shape[2]))
		s1 = F.interpolate(mask,to_2tuple(e1.shape[2]))
		s2 = F.interpolate(mask,to_2tuple(e2.shape[2]))
		s3 = F.interpolate(mask,to_2tuple(e3.shape[2]))
		for deco in self.decoder0:
			d0 = deco(d0,s0)
		d0 = self.reflat(d0,e0.shape[2],e0.shape[3])
		for deco in self.decoder1:
			d1 = deco(d1,s1)
		d1 = self.reflat(d1,e1.shape[2],e1.shape[3])
		for deco in self.decoder2:
			d2 = deco(d2,s2)
		d2 = self.reflat(d2,e2.shape[2],e2.shape[3])
		for deco in self.decoder3:
			d3 = deco(d3,s3)
		d3 = self.reflat(d3,e3.shape[2],e3.shape[3])
		 
		return d0,d1,d2,d3

class asnet_decoder(nn.Module):
	def __init__(self,feats_channels=[64,128,320,512],depths = [2,2,3,2],sb_drop_rate=0.,cnn_drop_rate=0.,seg_drop_rate=0.,mode='train',branch='seg'):
							  #shape: 64, 32, 16, 8
		super().__init__()
		self.mode = mode
		self.feats_channels = feats_channels
		self.branch = branch
		if branch == 'seg':
			self.d_seg_channels = [320,128,64,32,2]
			self.d_fusion_channels = [512,320,128,64,2]
		elif branch == 'alpha':
			self.d_seg_channels = [320,128,64,32,1]
			self.d_fusion_channels = [512,320,128,64,1]
		
							#shape:16, 32,64,128,256
							
		self.s2b_decoder = s2b_decoder(feats_channels,depths,drop_rate=sb_drop_rate)
		self.b2s_decoder = b2s_decoder(feats_channels,depths,drop_rate=sb_drop_rate)

		self.d0_seg = make_decoder(feats_channels[3],320,320,self.d_seg_channels[0],drop_rate=cnn_drop_rate)
		self.d1_seg = make_decoder(self.d_seg_channels[0]+feats_channels[2],256,256,self.d_seg_channels[1],drop_rate=cnn_drop_rate)
		self.d2_seg = make_decoder(self.d_seg_channels[1]+feats_channels[1],128,128,self.d_seg_channels[2],drop_rate=cnn_drop_rate)
		self.d3_seg = make_decoder(self.d_seg_channels[2]+feats_channels[0],64,64,self.d_seg_channels[3],drop_rate=cnn_drop_rate)
		self.d4_seg = make_decoder(self.d_seg_channels[3]+0,16,16,self.d_seg_channels[4],drop_rate=cnn_drop_rate,final_flag =True)

		self.d0_fusion = make_decoder(feats_channels[3]*2,640,640,self.d_fusion_channels[0],drop_rate=cnn_drop_rate)
		self.d1_fusion = make_decoder(self.d_fusion_channels[0]+feats_channels[2]*2,512,512,self.d_fusion_channels[1],drop_rate=cnn_drop_rate)
		self.d2_fusion = make_decoder(self.d_fusion_channels[1]+feats_channels[1]*2,320,320,self.d_fusion_channels[2],drop_rate=cnn_drop_rate)
		self.d3_fusion = make_decoder(self.d_fusion_channels[2]+feats_channels[0]*2,128,128,self.d_fusion_channels[3],drop_rate=cnn_drop_rate)
		self.d4_fusion = make_decoder(self.d_fusion_channels[3],32,32,self.d_fusion_channels[4],drop_rate=cnn_drop_rate,final_flag=True)

	def forward(self,feats):
		
		e1,e2,e3,e4 = feats

		d0_g = self.d0_seg(e4)
		d1_g = self.d1_seg(torch.cat((d0_g, e3),1))
		d2_g = self.d2_seg(torch.cat((d1_g, e2),1))
		d3_g = self.d3_seg(torch.cat((d2_g, e1),1))
		d4_g = self.d4_seg(d3_g)

		if self.branch == 'seg':
			seg_sig = F.sigmoid(d4_g)	#B,2,H,W
			_, seg = torch.max(seg_sig,1)
			seg = seg[:,None,:,:].float()

		elif self.branch == 'alpha':
			seg_sig = F.sigmoid(d4_g)	#B,1,H,W
			seg = torch.zeros(seg_sig.shape)
			seg[seg_sig > 0.4] = 1

		d0_s2b,d1_s2b,d2_s2b,d3_s2b, = self.s2b_decoder([e1,e2,e3,e4],seg)
		d0_b2s,d1_b2s,d2_b2s,d3_b2s, = self.b2s_decoder([e1,e2,e3,e4],seg)

		d0_f = self.d0_fusion(torch.cat((d3_s2b,d3_b2s),1))
		d1_f = self.d1_fusion(torch.cat((d0_f,d2_s2b,d2_b2s),1))
		d2_f = self.d2_fusion(torch.cat((d1_f,d1_s2b,d1_b2s),1))
		d3_f = self.d3_fusion(torch.cat((d2_f,d0_s2b,d0_b2s),1))
		d4_f = self.d4_fusion(d3_f)

		alpha = F.sigmoid(d4_f)

		if self.mode == 'train':
			return seg_sig,alpha
		
		elif self.mode == 'pics':
			outs = {}
			outs['e1']=e1
			outs['e2']=e2
			outs['e3']=e3
			outs['e4']=e4

			outs['d0_g']=d0_g
			outs['d1_g']=d1_g
			outs['d2_g']=d2_g
			outs['d3_g']=d3_g
			outs['d4_g']=d4_g

			outs['d0_s2b']=d0_s2b
			outs['d1_s2b']=d1_s2b
			outs['d2_s2b']=d2_s2b
			outs['d3_s2b']=d3_s2b

			outs['d0_b2s']=d0_b2s
			outs['d1_b2s']=d1_b2s
			outs['d2_b2s']=d2_b2s
			outs['d3_b2s']=d3_b2s

			return seg_sig,alpha,outs

if __name__ == '__main__':
	x0 = torch.ones(1, 64, 64, 64)
	x1 = torch.ones(1, 128, 32, 32)
	x2 = torch.ones(1, 320, 16, 16)
	x3 = torch.ones(1, 512, 8, 8)

	model = asnet_decoder()
	seg_sig,alpha = model([x0,x1,x2,x3])
	print(seg_sig.shape)
	print(alpha.shape)
	


