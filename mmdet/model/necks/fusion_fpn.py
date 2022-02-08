import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule, xavier_init

from mmcv.runner import auto_fp16
from ..builder import NECKS
#from .antialias import Downsample as downsamp


class Spatial_attention(nn.Module):
    def __init__(self,in_channels):
        super(Spatial_attention,self).__init__()
#        self.key = nn.Conv2d(in_channels,in_channels//8, kernel_size=1, stride=1,bias=True)
#        self.query = nn.Conv2d(in_channels,in_channels//8, kernel_size=1, stride=1,bias=True)
#        self.value = nn.Conv2d(in_channels,in_channels , kernel_size= 1,stride=1,bias=True)
        
        self.conv = nn.Conv2d(in_channels*2,1, kernel_size=5, stride=1,padding=2,bias=True)
#        self.conv1 = nn.Conv2d(in_channels,1, kernel_size=5, stride=1,padding=2,bias=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        _,C,H,W = x.shape
#        m_batchsize,C,width ,height = x.size()
#        print('x:',x.size())
#        proj_query  = self.query(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
#        proj_key =  self.key(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
#        print('query: ',proj_query.shape)
#        print('key: ',proj_key.shape)
#        energy =  torch.bmm(proj_query,proj_key) # transpose check
#        attention = self.softmax(energy) # BX (N) X (N) 
#        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
#
#        out = torch.bmm(proj_value,attention.permute(0,2,1) )
#        out = out.view(m_batchsize,C,width,height)
        
#        proj_query  = self.query(x).view(_,-1,W*H).permute(0,2,1) # B X CX(N)
#        proj_key =  self.key(x).view(_,-1,W*H) # B X C x (*W*H)
#        energy =  torch.bmm(proj_query,proj_key)
#        attention = nn.Softmax(dim=1)(energy)
#        proj_value = self.value(x).view(_,-1,W*H)
#        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        
        ################
#        out = out.view(_,C,H,W)
#        gap = nn.AvgPool2d(kernel_size = 5,stride=1,padding=2)(x)
        gap = nn.AdaptiveAvgPool2d((H,W))(x)
        gmp = nn.AdaptiveMaxPool2d((H,W))(x)
#        gmp = nn.MaxPool2d(kernel_size = 5,stride=1,padding=2)(x)
        sum_feat = torch.cat((gap,gmp),dim=1)
#        sum_feat = gap+gmp
        out = self.conv(sum_feat)
#        out = nn.ReLU()(out)
#        out = self.conv1(out)
        out = nn.Softmax(dim=1)(out) # self.sigmoid(out) # 
        return out

class channel_attention(nn.Module): 
    def __init__(self,in_channels):
        super(channel_attention,self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels//4, kernel_size=1, stride=1,bias=True)
        self.conv1 = nn.Conv2d(in_channels//4, in_channels, kernel_size=1, stride=1,bias=True)
        
    def forward(self,x):
#        x1 = self.conv(x)
#        gap = F.adaptive_avg_pool2d(x,output_size=1)
        gmp = F.adaptive_max_pool2d(x,output_size=1)
#        sum_feat = torch.cat((gap,gmp),dim=1)
        out = self.conv(gmp)
        out = nn.ReLU(inplace=True)(out)
        out = self.conv1(out)
#        out = nn.Softmax(dim=1)(out)
        out = nn.Sigmoid()(out)
        return out

class Attention(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Attention,self).__init__()
        self.spatial_att = Spatial_attention(in_channels)
        self.channels_att = channel_attention(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,bias=True)
        
    def forward(self,x):
        
#        spatial_mask= self.spatial_att(x)
#        out_1 = (spatial_mask*x)
##        out = out_1+x
#        channel_mask = self.channels_att(out_1)
#        out_2 = (out_1*channel_mask)
##        out = torch.cat((out_1,out_2),dim=1)
#        out = self.conv(out_2)
        
        channel_mask = self.channels_att(x)
        out_1 = (channel_mask*x)
#        out = out_1+x
        spatial_mask = self.spatial_att(out_1)
        out_2 = (out_1*spatial_mask)
#        out = torch.cat((out_1,out_2),dim=1)
        out = self.conv(out_2)
        return out

    
class Downsampling(nn.Module):
    def __init__(self,in_channels,out_channels,scale_factor):
        super(Downsampling,self).__init__()
        
        self.scale_factor = int(np.log2(scale_factor))
        modules_body = []
        for i in range(self.scale_factor):
#            modules_body.append(downsampling_unit(out_channels))
#            self.downsampling = downsamp(channels=out_channels,filt_size=2,stride=2)
            self.downsampling =  nn.Conv2d(out_channels, out_channels, 3, stride=2,padding=1, bias=True)
            #downsamp(channels=out_channels,filt_size=2,stride=2),
#                                nn.Conv2d(out_channels, out_channels, 1, stride=1, padding=0, bias=False)) #in_channels*2
            # nn.AvgPool2d(kernel_size=2,stride=2)
            modules_body.append(self.downsampling)
#            in_channels = int(in_channels * 2)
        
        self.body = nn.Sequential(*modules_body)
#        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,padding=1,bias=True)
        
    def forward(self,x):
#        for i in range(self.scale_factor):
#            out = F.avg_pool2d(x,kernel_size=2, stride=2)
#        out = self.conv(x)
#        out = nn.ReLU()(out)
        out = self.body(x)   
        return out
    

class Upsampling(nn.Module):
    def __init__(self,in_channels,out_channels,scale_factor):
        super(Upsampling,self).__init__()
        
        self.scale_factor = int(np.log2(scale_factor))
        modules_body = []
        for i in range(self.scale_factor):
#            modules_body.append(upsampling_unit(out_channels))
            self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
             #nn.ConvTranspose2d(out_channels,out_channels, 3, stride=2, padding=1, output_padding=1,bias=False)
#            #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            #nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), #bilinear
#                               # nn.Conv2d(out_channels,out_channels , 1, stride=1, padding=0, bias=False)) #in_channels//2
            modules_body.append(self.upsampling)
##            in_channels = int(in_channels // 2)
        
        self.body = nn.Sequential(*modules_body)
#        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,padding=1,bias=True)
#        self.in_channels = in_channels
        
    def forward(self,x):
#        out = self.conv(x) 
#        out = nn.ReLU()(out)
        out = self.body(x)
#        print('out_up:',out.shape)                   
        return out
    
#class Flatten(nn.Module):
#    def forward(self, x):
#        return x.view(x.size(0), -1)
    
class Fusionmodule(nn.Module):
    def __init__(self,out_channels):
        super(Fusionmodule,self).__init__()
        
        self.selected = nn.Sequential(
#               nn.AvgPool2d(kernel_size=3, stride=1,padding=1),
               nn.AdaptiveAvgPool2d(output_size = 1),
#               Flatten(),
#               nn.Linear(out_channels,4),
               nn.Conv2d(out_channels, 4, kernel_size=1, stride=1,bias=True),
#               nn.BatchNorm2d(4),
               nn.ReLU(inplace=True)
#               nn.Conv2d(out_channels, 4, kernel_size=1, stride=1,bias=True),
#               nn.ReLU(inplace=True)
                )
        self.bot = nn.Sequential(
               nn.AdaptiveMaxPool2d(output_size = 1),
               nn.Conv2d(out_channels, 4, kernel_size=1, stride=1,bias=True),
##               nn.BatchNorm2d(4),
               nn.ReLU(inplace=True)
##               nn.Conv2d(out_channels//4, 4, kernel_size=1, stride=1,bias=True),
##               nn.ReLU(inplace=True)
                )
#        self.conv1 = nn.Conv2d(8, 4, kernel_size=1, stride=1,bias=True)
        self.fcs = nn.ModuleList([])
        for i in range(4):
            self.fcs.append(nn.Conv2d(4, out_channels, kernel_size=1, stride=1,bias=True))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,inp_feats):
        
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]
        #print('len:',len(inp_feats))
#        print('shape:',inp_feats.shape)
        inp_feats = torch.cat(inp_feats, dim=1)
        #print('inp_feature:',inp_feats.shape)
        inp_feats = inp_feats.view(batch_size, 4, n_feats,inp_feats.shape[2], inp_feats.shape[3])
#        feature_sum = inp_feats[:,0,:,:]
#        for i in range(1,3):
#            feature_sum = feature_sum*inp_feats[:,i,:,:]
        feature_sum = torch.sum(inp_feats,dim=1)
#        print('feature_:',feature_sum.shape)
        selected_feat = self.selected(feature_sum)
#        gmp = nn.AdaptiveMaxPool2d((inp_feats[0].shape[2], inp_feats[0].shape[3]))(feature_sum)
        selected_bot = self.bot(feature_sum)
#        selected_final = torch.cat((selected_feat,selected_bot),dim=1)
#        selected_final = self.conv1(selected_final)
        selected_final = selected_feat+selected_bot
#        print(selected_final.shape)
#        selected_final = self.softmax(selected_final)
        atten_vectors = [fc(selected_final) for fc in self.fcs]
        atten_vectors = torch.cat(atten_vectors,dim=1)
        atten_vectors = atten_vectors.view(batch_size, 4, n_feats,1,1)#inp_feats[0].shape[2], inp_feats[0].shape[3])
        atten_vectors = self.softmax(atten_vectors)
        
        out = torch.sum(inp_feats*atten_vectors,dim=1)
        return out


        
@NECKS.register_module()
class Fusion_FPN(nn.Module):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbon/data/object_detection/Code/mmdetection/work_dirs/Lupinv2/e feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(Fusion_FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
      
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.attention = nn.ModuleList()
#        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.fusion = nn.ModuleList()
#        self.fusion_d = nn.ModuleList()
#        self.attention_2 = nn.ModuleList()
#        self.conv1 = nn.ModuleList()
#        self.downsamp = nn.ModuleList()
#        self.upfpn = nn.ModuleList()
        
        for i in range(self.start_level, self.backbone_end_level):
#            l_conv = ConvModule(
#                in_channels[i],
#                out_channels,
#                1,
#                conv_cfg=conv_cfg,
#                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
#                act_cfg=act_cfg,
#                inplace=False)
            att_l = Attention(
#                    out_channels,
                    in_channels[i],
                    out_channels
                    )
#            att_2 = Attention(
#                    out_channels,
#                    out_channels
#                    )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            fus_l = Fusionmodule(
                    out_channels
                    )
#            fus_dl = Fusionmodule(
#                    out_channels
#                    )
#            conv_feat = ConvModule(
#                in_channels[i],
#                out_channels,
#                1,
#                padding=0,
#                conv_cfg=conv_cfg,
#                norm_cfg=norm_cfg,
#                act_cfg=act_cfg,
            
#                inplace=False)
#            if i > 0:
#                d_conv = ConvModule(
#                        out_channels,
#                        out_channels,
#                        3,
#                        stride=2,
#                        padding=1,
#                        conv_cfg=conv_cfg,
#                        norm_cfg=norm_cfg,
#                        act_cfg=act_cfg,
#                        inplace=False)
#                dfpn_conv = ConvModule(
#                        out_channels,
#                        out_channels,
#                        3,
#                        padding=1,
#                        conv_cfg=conv_cfg,
#                        norm_cfg=norm_cfg,
#                        act_cfg=act_cfg,
#                        inplace=False)
#                self.downsamp.append(d_conv)
#                self.upfpn.append(dfpn_conv)
            self.fusion.append(fus_l)
#            self.lateral_convs.append(l_conv)
            self.attention.append(att_l)
            self.fpn_convs.append(fpn_conv)
#            self.attention_2.append(att_2)
#            self.fusion_d.append(fus_dl)
#            self.conv1.append(conv_feat)
            
        
#        self.Attention = Attention(out_channels)
#        self.Fusion = Fusionmodule(
#                    out_channels
#                    )
        
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)
        

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
#        laterals = []
#        for i, lateral_att in enumerate(self.attention):
#            print('i:',i,'input',inputs[i + self.start_level].shape)
#            print('out_att',lateral_att(inputs[i + self.start_level]).shape)
#            laterals.append(lateral_att(inputs[i + self.start_level]))
                 
#         build laterals
#        laterals_f = [
#            lateral_conv(inputs[i + self.start_level])
#            for i, lateral_conv in enumerate(self.lateral_convs)
#        ]
        laterals = [
            lateral_att(inputs[i + self.start_level])
            for i, lateral_att in enumerate(self.attention)
        ]
#        
        # build top-down path
        used_backbone_levels = len(laterals)
      
        fus_feat = []
        for i, lateral_fus in enumerate(self.fusion):
            in_feat = []
            for j in range(len(laterals)):   
                if j == i:
#                    print('j==i and j =',j)
#                    print(laterals[j].shape)
                    feat_ch = laterals[j]#self.conv1[j]()
                    in_feat.append(feat_ch)                    
                elif j > i:     
                    scale_factor = (laterals[i].shape[3])//(laterals[j].shape[3])
#                    print('j>i and j =',j,'input_shape:',inputs[j].shape,'scale_fac = ',scale_factor)
                    feat_up = Upsampling(self.in_channels[j],self.out_channels,scale_factor).cuda()(laterals[j])
                    in_feat.append(feat_up)
                elif j < i:
                    scale_factor = (laterals[j].shape[3])//(laterals[i].shape[3])
#                    feat_down = self.downsampling(laterals[j])
#                    print('feat_down_conv',feat_down.shape)
                    feat_down = Downsampling(self.in_channels[j],self.out_channels,scale_factor).cuda()(laterals[j])
#                    print('feat_down_down',feat_down.shape)
                    in_feat.append(feat_down)
##            print(len(in_feat))
            fus_feat.append(lateral_fus(in_feat))
        

        for i in range(len(fus_feat)):
            fus_feat[i] += laterals[i]
            
#        laterals_out = []
#        laterals_out.append(fus_feat[0])
#        for i in range(used_backbone_levels-1):
##            laterals_tem = F.avg_pool2d(fus_feat[i],kernel_size=2, stride=2) + fus_feat[i+1]
#            laterals_tem = self.downsamp[i](fus_feat[i]) + fus_feat[i+1]   
##            laterals_tem = Downsampling(self.out_channels,self.out_channels,2).cuda()(fus_feat[i]) + fus_feat[i+1]   
#            laterals_out.append(laterals_tem)
            
#        laterals_out = [
#            lateral_att(laterals_out[i + self.start_level])
#            for i, lateral_att in enumerate(self.attention_2)
#        ]
        laterals[used_backbone_levels - 1] = fus_feat[used_backbone_levels - 1] 
        for i in range(used_backbone_levels - 1, 0, -1):
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] =  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)(fus_feat[i]) + fus_feat[i-1]
                #F.interpolate(fus_feat[i],**self.upsample_cfg) + fus_feat[i-1]
#                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)(laterals[i]) + fus_feat[i-1]
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = F.interpolate(fus_feat[i], size=prev_shape, **self.upsample_cfg) + fus_feat[i-1]
#            
#        for i in range(used_backbone_levels - 1, 0, -1):
#            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
#            #  it cannot co-exist with `size` in `F.interpolate`.
#            if 'scale_factor' in self.upsample_cfg:
#                laterals[i - 1] += F.interpolate(laterals[i],
#                                                 **self.upsample_cfg)
#            else:
#                prev_shape = laterals[i - 1].shape[2:]
#                laterals[i - 1] += F.interpolate(
#                    laterals[i], size=prev_shape, **self.upsample_cfg)
#        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
