import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from quant import *
from sparsity import *


#### DenseDepthNet ####

class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()        
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB( self.convB( self.convA( torch.cat([up_x, concat_with], dim=1)  ) )  )

class Decoder(nn.Module):
    def __init__(self, num_features=1664, decoder_width = 1.0):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)
        
        # features = 1664 * 1.0 = 1664

        # conv2d 1664, 1664
        # Upsample ( 1664 + 256, 832)
        # Upsample ( 832 + 128 , 416)
        # Upsample ( 416 + 64 , 208 )
        # Upsample ( 208 + 64, 104 )
        # conv2d (104, 1)


        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)

        self.up1 = UpSample(skip_input=features//1 + 256, output_features=features//2)
        self.up2 = UpSample(skip_input=features//2 + 128,  output_features=features//4)
        self.up3 = UpSample(skip_input=features//4 + 64,  output_features=features//8)
        self.up4 = UpSample(skip_input=features//8 + 64,  output_features=features//16)

        self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[4], features[6], features[8], features[12]
        x_d0 = self.conv2(F.relu(x_block4))

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        return self.conv3(x_d4)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()       
        self.original_model = models.densenet169( pretrained=False )

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items(): features.append( v(features[-1]) )
        return features

class PTModel(nn.Module):
    def __init__(self):
        super(PTModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder( self.encoder(x) )
    



#### DispNet ####



from torch.nn.init import xavier_uniform_, zeros_


# def downsample_conv(in_planes, out_planes, kernel_size=3):
#     return nn.Sequential(
#         nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
#         nn.ReLU(inplace=True)
#     )


def predict_disp(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
        nn.Sigmoid()
    )


def conv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True)
    )



def crop_like(input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]






#def downsample_conv_leaky(in_planes, out_planes, kernel_size=3):
#    return nn.Sequential(
#        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
#        nn.LeakyReLU(inplace=True),
#        nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
#        nn.LeakyReLU(inplace=True)
#    )


def predict_disp_leaky(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
        nn.Sigmoid()
    )


def conv_leaky(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.LeakyReLU(inplace=True)
    )


def upconv_leaky(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.LeakyReLU(inplace=True)
    )


#### ReLU downsample conv (nn.sequential) ####

class downsample_conv(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super(downsample_conv, self).__init__()        
        self.convA = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2)
        self.reluA = nn.ReLU(inplace=True)
        self.convB = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.reluB = nn.ReLU(inplace=True)
        

    def forward(self, x):
        outconv1 = self.reluA(self.convA(x))
        outconv2 = self.reluB(self.convB(outconv1))
        
        return outconv2


class downsample_conv_quant(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super(downsample_conv_quant, self).__init__()        
        self.convA = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2)
        self.reluA = nn.ReLU(inplace=True)
        self.convB = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.reluB = nn.ReLU(inplace=True)
        

    def forward(self, x, type= "fxp", n_exp = 5, n_man = 10, mode = "round", device = "cuda"):
        outconv1 = Quant.apply( self.reluA(self.convA(x)), type, n_exp, n_man, mode, device )
        outconv2 = self.reluB(self.convB(outconv1))
        
        return outconv2
    

class downsample_conv_quant_encoding(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super(downsample_conv_quant_encoding, self).__init__()        
        self.convA = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2)
        self.reluA = nn.ReLU(inplace=True)
        self.convB = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.reluB = nn.ReLU(inplace=True)
        

    def forward(self, x, type= "fxp", n_exp = 5, n_man = 10, slice_width= 4, mode = "round", device = "cuda"):
        total_count_activation = 0
        zero_count_activation = 0
        outconv1 = Quant.apply( self.reluA(self.convA(x)), type, n_exp, n_man, mode, device )
        bin_out_conv1 = bin_fxp(outconv1, n_exp, n_man, mode, device)
        total_count_activation += encoding(bin_out_conv1, n_exp + n_man, slice_width)[0]
        zero_count_activation += encoding(bin_out_conv1,  n_exp + n_man, slice_width)[1]
        outconv2 = self.reluB(self.convB(outconv1))
        
        return outconv2, total_count_activation, zero_count_activation
    
class downsample_conv_quant_signed_encoding(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super(downsample_conv_quant_signed_encoding, self).__init__()        
        self.convA = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2)
        self.reluA = nn.ReLU(inplace=True)
        self.convB = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.reluB = nn.ReLU(inplace=True)
        

    def forward(self, x, type= "fxp", n_exp = 5, n_man = 10, slice_width= 4, mode = "round", device = "cuda"):
        total_count_activation = 0
        zero_count_activation = 0
        outconv1 = Quant.apply( self.reluA(self.convA(x)), type, n_exp, n_man, mode, device )
        bin_out_conv1 = bin_fxp(outconv1, n_exp, n_man, mode, device)
        total_count_activation += signed_encoding(bin_out_conv1, n_exp + n_man, slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_conv1,  n_exp + n_man, slice_width)[1]
        outconv2 = self.reluB(self.convB(outconv1))
        
        return outconv2, total_count_activation, zero_count_activation
    
class downsample_conv_quant_no_encoding(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super(downsample_conv_quant_no_encoding, self).__init__()        
        self.convA = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2)
        self.reluA = nn.ReLU(inplace=True)
        self.convB = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.reluB = nn.ReLU(inplace=True)
        

    def forward(self, x, type= "fxp", n_exp = 5, n_man = 10, slice_width= 4, mode = "round", device = "cuda"):
        total_count_activation = 0
        zero_count_activation = 0
        outconv1 = Quant.apply( self.reluA(self.convA(x)), type, n_exp, n_man, mode, device )
        bin_out_conv1 = bin_fxp(outconv1, n_exp, n_man, mode, device)
        total_count_activation += no_encoding(bin_out_conv1, n_exp + n_man, slice_width)[0]
        zero_count_activation += no_encoding(bin_out_conv1,  n_exp + n_man, slice_width)[1]
        outconv2 = self.reluB(self.convB(outconv1))
        
        return outconv2, total_count_activation, zero_count_activation
    
#### Leaky ReLU downsample conv (nn.sequential) ####
    

class downsample_conv_leaky(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super(downsample_conv_leaky, self).__init__()        
        self.convA = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2)
        self.reluA = nn.LeakyReLU(inplace=True)
        self.convB = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.reluB = nn.LeakyReLU(inplace=True)
        

    def forward(self, x):
        outconv1 = self.reluA(self.convA(x))
        outconv2 = self.reluB(self.convB(outconv1))
        
        return outconv2


class downsample_conv_quant_leaky(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super(downsample_conv_quant_leaky, self).__init__()        
        self.convA = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2)
        self.reluA = nn.LeakyReLU(inplace=True)
        self.convB = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.reluB = nn.LeakyReLU(inplace=True)
        

    def forward(self, x, type= "fxp", n_exp = 5, n_man = 10, mode = "round", device = "cuda"):
        outconv1 = Quant.apply( self.reluA(self.convA(x)), type, n_exp, n_man, mode, device )
        outconv2 = self.reluB(self.convB(outconv1))
        
        return outconv2
    

class downsample_conv_quant_encoding_leaky(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super(downsample_conv_quant_encoding_leaky, self).__init__()        
        self.convA = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2)
        self.reluA = nn.LeakyReLU(inplace=True)
        self.convB = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.reluB = nn.LeakyReLU(inplace=True)
        

    def forward(self, x, type= "fxp", n_exp = 5, n_man = 10, slice_width= 4, mode = "round", device = "cuda"):
        total_count_activation = 0
        zero_count_activation = 0
        outconv1 = Quant.apply( self.reluA(self.convA(x)), type, n_exp, n_man, mode, device )
        bin_out_conv1 = bin_fxp(outconv1, n_exp, n_man, mode, device)
        total_count_activation += encoding(bin_out_conv1, n_exp + n_man, slice_width)[0]
        zero_count_activation += encoding(bin_out_conv1,  n_exp + n_man, slice_width)[1]
        outconv2 = self.reluB(self.convB(outconv1))
        
        return outconv2, total_count_activation, zero_count_activation
    
class downsample_conv_quant_signed_encoding_leaky(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super(downsample_conv_quant_signed_encoding_leaky, self).__init__()        
        self.convA = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2)
        self.reluA = nn.LeakyReLU(inplace=True)
        self.convB = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.reluB = nn.LeakyReLU(inplace=True)
        

    def forward(self, x, type= "fxp", n_exp = 5, n_man = 10, slice_width= 4, mode = "round", device = "cuda"):
        total_count_activation = 0
        zero_count_activation = 0
        outconv1 = Quant.apply( self.reluA(self.convA(x)), type, n_exp, n_man, mode, device )
        bin_out_conv1 = bin_fxp(outconv1, n_exp, n_man, mode, device)
        total_count_activation += signed_encoding(bin_out_conv1, n_exp + n_man, slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_conv1,  n_exp + n_man, slice_width)[1]
        outconv2 = self.reluB(self.convB(outconv1))
        
        return outconv2, total_count_activation, zero_count_activation
    
class downsample_conv_quant_no_encoding_leaky(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super(downsample_conv_quant_no_encoding_leaky, self).__init__()        
        self.convA = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2)
        self.reluA = nn.LeakyReLU(inplace=True)
        self.convB = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.reluB = nn.LeakyReLU(inplace=True)
        

    def forward(self, x, type= "fxp", n_exp = 5, n_man = 10, slice_width= 4, mode = "round", device = "cuda"):
        total_count_activation = 0
        zero_count_activation = 0
        outconv1 = Quant.apply( self.reluA(self.convA(x)), type, n_exp, n_man, mode, device )
        bin_out_conv1 = bin_fxp(outconv1, n_exp, n_man, mode, device)
        total_count_activation += no_encoding(bin_out_conv1, n_exp + n_man, slice_width)[0]
        zero_count_activation += no_encoding(bin_out_conv1,  n_exp + n_man, slice_width)[1]
        outconv2 = self.reluB(self.convB(outconv1))
        
        return outconv2, total_count_activation, zero_count_activation


class DispNetS(nn.Module):

    def __init__(self, alpha=10, beta=0.01):
        super(DispNetS, self).__init__()

        self.alpha = alpha
        self.beta = beta

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv1 = downsample_conv(3,              conv_planes[0], kernel_size=7)
        self.conv2 = downsample_conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = downsample_conv(conv_planes[1], conv_planes[2])
        self.conv4 = downsample_conv(conv_planes[2], conv_planes[3])
        self.conv5 = downsample_conv(conv_planes[3], conv_planes[4])
        self.conv6 = downsample_conv(conv_planes[4], conv_planes[5])
        self.conv7 = downsample_conv(conv_planes[5], conv_planes[6])

        upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv7 = upconv(conv_planes[6],   upconv_planes[0])
        self.upconv6 = upconv(upconv_planes[0], upconv_planes[1])
        self.upconv5 = upconv(upconv_planes[1], upconv_planes[2])
        self.upconv4 = upconv(upconv_planes[2], upconv_planes[3])
        self.upconv3 = upconv(upconv_planes[3], upconv_planes[4])
        self.upconv2 = upconv(upconv_planes[4], upconv_planes[5])
        self.upconv1 = upconv(upconv_planes[5], upconv_planes[6])

        self.iconv7 = conv(upconv_planes[0] + conv_planes[5], upconv_planes[0])
        self.iconv6 = conv(upconv_planes[1] + conv_planes[4], upconv_planes[1])
        self.iconv5 = conv(upconv_planes[2] + conv_planes[3], upconv_planes[2])
        self.iconv4 = conv(upconv_planes[3] + conv_planes[2], upconv_planes[3])
        self.iconv3 = conv(1 + upconv_planes[4] + conv_planes[1], upconv_planes[4])
        self.iconv2 = conv(1 + upconv_planes[5] + conv_planes[0], upconv_planes[5])
        self.iconv1 = conv(1 + upconv_planes[6], upconv_planes[6])

        self.predict_disp4 = predict_disp(upconv_planes[3])
        self.predict_disp3 = predict_disp(upconv_planes[4])
        self.predict_disp2 = predict_disp(upconv_planes[5])
        self.predict_disp1 = predict_disp(upconv_planes[6])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        out_upconv7 = crop_like(self.upconv7(out_conv7), out_conv6)
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)

        out_upconv6 = crop_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_iconv6 = self.iconv6(concat6)

        out_upconv5 = crop_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)

        out_upconv4 = crop_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_iconv4 = self.iconv4(concat4)
        disp4 = self.alpha * self.predict_disp4(out_iconv4) + self.beta

        out_upconv3 = crop_like(self.upconv3(out_iconv4), out_conv2)
        disp4_up = crop_like(F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        concat3 = torch.cat((out_upconv3, out_conv2, disp4_up), 1)
        out_iconv3 = self.iconv3(concat3)
        disp3 = self.alpha * self.predict_disp3(out_iconv3) + self.beta

        out_upconv2 = crop_like(self.upconv2(out_iconv3), out_conv1)
        disp3_up = crop_like(F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        concat2 = torch.cat((out_upconv2, out_conv1, disp3_up), 1)
        out_iconv2 = self.iconv2(concat2)
        disp2 = self.alpha * self.predict_disp2(out_iconv2) + self.beta

        out_upconv1 = crop_like(self.upconv1(out_iconv2), x)
        disp2_up = crop_like(F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=False), x)
        concat1 = torch.cat((out_upconv1, disp2_up), 1)
        out_iconv1 = self.iconv1(concat1)
        disp1 = self.alpha * self.predict_disp1(out_iconv1) + self.beta

        if self.training:
            return disp1, disp2, disp3, disp4
        else:
            return disp1, disp2, disp3, disp4
        


class DispNetS_leaky(nn.Module):

    def __init__(self, alpha=10, beta=0.01):
        super(DispNetS_leaky, self).__init__()

        self.alpha = alpha
        self.beta = beta

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv1 = downsample_conv_leaky(3,              conv_planes[0], kernel_size=7)
        self.conv2 = downsample_conv_leaky(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = downsample_conv_leaky(conv_planes[1], conv_planes[2])
        self.conv4 = downsample_conv_leaky(conv_planes[2], conv_planes[3])
        self.conv5 = downsample_conv_leaky(conv_planes[3], conv_planes[4])
        self.conv6 = downsample_conv_leaky(conv_planes[4], conv_planes[5])
        self.conv7 = downsample_conv_leaky(conv_planes[5], conv_planes[6])

        upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv7 = upconv_leaky(conv_planes[6],   upconv_planes[0])
        self.upconv6 = upconv_leaky(upconv_planes[0], upconv_planes[1])
        self.upconv5 = upconv_leaky(upconv_planes[1], upconv_planes[2])
        self.upconv4 = upconv_leaky(upconv_planes[2], upconv_planes[3])
        self.upconv3 = upconv_leaky(upconv_planes[3], upconv_planes[4])
        self.upconv2 = upconv_leaky(upconv_planes[4], upconv_planes[5])
        self.upconv1 = upconv_leaky(upconv_planes[5], upconv_planes[6])

        self.iconv7 = conv_leaky(upconv_planes[0] + conv_planes[5], upconv_planes[0])
        self.iconv6 = conv_leaky(upconv_planes[1] + conv_planes[4], upconv_planes[1])
        self.iconv5 = conv_leaky(upconv_planes[2] + conv_planes[3], upconv_planes[2])
        self.iconv4 = conv_leaky(upconv_planes[3] + conv_planes[2], upconv_planes[3])
        self.iconv3 = conv_leaky(1 + upconv_planes[4] + conv_planes[1], upconv_planes[4])
        self.iconv2 = conv_leaky(1 + upconv_planes[5] + conv_planes[0], upconv_planes[5])
        self.iconv1 = conv_leaky(1 + upconv_planes[6], upconv_planes[6])

        self.predict_disp4 = predict_disp_leaky(upconv_planes[3])
        self.predict_disp3 = predict_disp_leaky(upconv_planes[4])
        self.predict_disp2 = predict_disp_leaky(upconv_planes[5])
        self.predict_disp1 = predict_disp_leaky(upconv_planes[6])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        out_upconv7 = crop_like(self.upconv7(out_conv7), out_conv6)
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)

        out_upconv6 = crop_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_iconv6 = self.iconv6(concat6)

        out_upconv5 = crop_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)

        out_upconv4 = crop_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_iconv4 = self.iconv4(concat4)
        disp4 = self.alpha * self.predict_disp4(out_iconv4) + self.beta

        out_upconv3 = crop_like(self.upconv3(out_iconv4), out_conv2)
        disp4_up = crop_like(F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        concat3 = torch.cat((out_upconv3, out_conv2, disp4_up), 1)
        out_iconv3 = self.iconv3(concat3)
        disp3 = self.alpha * self.predict_disp3(out_iconv3) + self.beta

        out_upconv2 = crop_like(self.upconv2(out_iconv3), out_conv1)
        disp3_up = crop_like(F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        concat2 = torch.cat((out_upconv2, out_conv1, disp3_up), 1)
        out_iconv2 = self.iconv2(concat2)
        disp2 = self.alpha * self.predict_disp2(out_iconv2) + self.beta

        out_upconv1 = crop_like(self.upconv1(out_iconv2), x)
        disp2_up = crop_like(F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=False), x)
        concat1 = torch.cat((out_upconv1, disp2_up), 1)
        out_iconv1 = self.iconv1(concat1)
        disp1 = self.alpha * self.predict_disp1(out_iconv1) + self.beta

        if self.training:
            return disp1, disp2, disp3, disp4
        else:
            return disp1, disp2, disp3, disp4
        

class DispNetS_Q(nn.Module):

    def __init__(self, alpha=10, beta=0.01, type = "fxp", n_exp = 5, n_man = 10, mode = "trunc", device = "cuda" ):
        super(DispNetS_Q, self).__init__()

        self.type = type
        self.n_exp = n_exp
        self.n_man = n_man
        self.mode = mode
        self.device = device
        self.alpha = alpha
        self.beta = beta
        

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv1 = downsample_conv_quant(3,              conv_planes[0], kernel_size=7)
        self.conv2 = downsample_conv_quant(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = downsample_conv_quant(conv_planes[1], conv_planes[2])
        self.conv4 = downsample_conv_quant(conv_planes[2], conv_planes[3])
        self.conv5 = downsample_conv_quant(conv_planes[3], conv_planes[4])
        self.conv6 = downsample_conv_quant(conv_planes[4], conv_planes[5])
        self.conv7 = downsample_conv_quant(conv_planes[5], conv_planes[6])

        upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv7 = upconv(conv_planes[6],   upconv_planes[0])
        self.upconv6 = upconv(upconv_planes[0], upconv_planes[1])
        self.upconv5 = upconv(upconv_planes[1], upconv_planes[2])
        self.upconv4 = upconv(upconv_planes[2], upconv_planes[3])
        self.upconv3 = upconv(upconv_planes[3], upconv_planes[4])
        self.upconv2 = upconv(upconv_planes[4], upconv_planes[5])
        self.upconv1 = upconv(upconv_planes[5], upconv_planes[6])

        self.iconv7 = conv(upconv_planes[0] + conv_planes[5], upconv_planes[0])
        self.iconv6 = conv(upconv_planes[1] + conv_planes[4], upconv_planes[1])
        self.iconv5 = conv(upconv_planes[2] + conv_planes[3], upconv_planes[2])
        self.iconv4 = conv(upconv_planes[3] + conv_planes[2], upconv_planes[3])
        self.iconv3 = conv(1 + upconv_planes[4] + conv_planes[1], upconv_planes[4])
        self.iconv2 = conv(1 + upconv_planes[5] + conv_planes[0], upconv_planes[5])
        self.iconv1 = conv(1 + upconv_planes[6], upconv_planes[6])

        self.predict_disp4 = predict_disp(upconv_planes[3])
        self.predict_disp3 = predict_disp(upconv_planes[4])
        self.predict_disp2 = predict_disp(upconv_planes[5])
        self.predict_disp1 = predict_disp(upconv_planes[6])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        x = Quant.apply(x, self.type, self.n_exp, self.n_man, self.mode, self.device)
        
        out_conv1 = self.conv1(x, self.type, self.n_exp, self.n_man, self.mode, self.device)
        out_conv1 = Quant.apply(out_conv1, self.type, self.n_exp, self.n_man, self.mode, self.device)
        
        out_conv2 = self.conv2(out_conv1, self.type, self.n_exp, self.n_man, self.mode, self.device)
        out_conv2 = Quant.apply(out_conv2, self.type, self.n_exp, self.n_man, self.mode, self.device)
        
        out_conv3 = self.conv3(out_conv2, self.type, self.n_exp, self.n_man, self.mode, self.device)
        out_conv3 = Quant.apply(out_conv3, self.type, self.n_exp, self.n_man, self.mode, self.device)
        
        out_conv4 = self.conv4(out_conv3, self.type, self.n_exp, self.n_man, self.mode, self.device)
        out_conv4 = Quant.apply(out_conv4, self.type, self.n_exp, self.n_man, self.mode, self.device)
        
        out_conv5 = self.conv5(out_conv4, self.type, self.n_exp, self.n_man, self.mode, self.device)
        out_conv5 = Quant.apply(out_conv5, self.type, self.n_exp, self.n_man, self.mode, self.device)
        
        out_conv6 = self.conv6(out_conv5, self.type, self.n_exp, self.n_man, self.mode, self.device)
        out_conv6 = Quant.apply(out_conv6, self.type, self.n_exp, self.n_man, self.mode, self.device)
        
        out_conv7 = self.conv7(out_conv6, self.type, self.n_exp, self.n_man, self.mode, self.device)
        out_conv7 = Quant.apply(out_conv7, self.type, self.n_exp, self.n_man, self.mode, self.device)
        
        out_upconv7 = crop_like(self.upconv7(out_conv7), out_conv6)
        out_upconv7 = Quant.apply(out_upconv7, self.type, self.n_exp, self.n_man, self.mode, self.device)
        
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)
        out_iconv7 = Quant.apply(out_iconv7, self.type, self.n_exp, self.n_man, self.mode, self.device)
        

        out_upconv6 = crop_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_iconv6 = self.iconv6(concat6)
        out_iconv6 = Quant.apply(out_iconv6, self.type, self.n_exp, self.n_man, self.mode, self.device)
        

        out_upconv5 = crop_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)
        out_iconv5 = Quant.apply(out_iconv5, self.type, self.n_exp, self.n_man, self.mode, self.device)
        


        out_upconv4 = crop_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_iconv4 = self.iconv4(concat4)
        out_iconv4 = Quant.apply(out_iconv4, self.type, self.n_exp, self.n_man, self.mode, self.device)
        
        disp4 = self.alpha * self.predict_disp4(out_iconv4) + self.beta
        disp4 = Quant.apply(disp4, self.type, self.n_exp, self.n_man, self.mode, self.device)
        

        out_upconv3 = crop_like(self.upconv3(out_iconv4), out_conv2)
        disp4_up = crop_like(F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        disp4_up = Quant.apply(disp4_up, self.type, self.n_exp, self.n_man, self.mode, self.device)
        
        concat3 = torch.cat((out_upconv3, out_conv2, disp4_up), 1)
        out_iconv3 = self.iconv3(concat3)
        out_iconv3 = Quant.apply(out_iconv3, self.type, self.n_exp, self.n_man, self.mode, self.device)
        
        disp3 = self.alpha * self.predict_disp3(out_iconv3) + self.beta
        disp3 = Quant.apply(disp3, self.type, self.n_exp, self.n_man, self.mode, self.device)
        

        out_upconv2 = crop_like(self.upconv2(out_iconv3), out_conv1)
        disp3_up = crop_like(F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        disp3_up = Quant.apply(disp3_up, self.type, self.n_exp, self.n_man, self.mode, self.device)
        
        
        concat2 = torch.cat((out_upconv2, out_conv1, disp3_up), 1)
        out_iconv2 = self.iconv2(concat2)
        out_iconv2 = Quant.apply(out_iconv2, self.type, self.n_exp, self.n_man, self.mode, self.device)
        
        disp2 = self.alpha * self.predict_disp2(out_iconv2) + self.beta
        disp2 = Quant.apply(disp2, self.type, self.n_exp, self.n_man, self.mode, self.device)
        

        out_upconv1 = crop_like(self.upconv1(out_iconv2), x)
        disp2_up = crop_like(F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=False), x)
        disp2_up = Quant.apply(disp2_up, self.type, self.n_exp, self.n_man, self.mode, self.device)
        
        concat1 = torch.cat((out_upconv1, disp2_up), 1)
        out_iconv1 = self.iconv1(concat1)
        out_iconv1 = Quant.apply(out_iconv1, self.type, self.n_exp, self.n_man, self.mode, self.device)
        
        disp1 = self.alpha * self.predict_disp1(out_iconv1) + self.beta
        disp1 = Quant.apply(disp1, self.type, self.n_exp, self.n_man, self.mode, self.device)
        

        if self.training:
            return disp1, disp2, disp3, disp4
        else:
            return disp1, disp2, disp3, disp4



class DispNetS_Q_full(nn.Module):

    def __init__(self, alpha=10, beta=0.01, type = "fxp", n_exp_en = 5, n_man_en = 10,  n_exp_de = 5, n_man_de = 10, mode = "trunc", device = "cuda" ):
        super(DispNetS_Q_full, self).__init__()

        self.type = type
        self.n_exp_en = n_exp_en
        self.n_man_en = n_man_en
        self.n_exp_de = n_exp_de
        self.n_man_de = n_man_de

        self.mode = mode
        self.device = device
        self.alpha = alpha
        self.beta = beta
        

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv1 = downsample_conv_quant(3,              conv_planes[0], kernel_size=7)
        self.conv2 = downsample_conv_quant(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = downsample_conv_quant(conv_planes[1], conv_planes[2])
        self.conv4 = downsample_conv_quant(conv_planes[2], conv_planes[3])
        self.conv5 = downsample_conv_quant(conv_planes[3], conv_planes[4])
        self.conv6 = downsample_conv_quant(conv_planes[4], conv_planes[5])
        self.conv7 = downsample_conv_quant(conv_planes[5], conv_planes[6])

        upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv7 = upconv(conv_planes[6],   upconv_planes[0])
        self.upconv6 = upconv(upconv_planes[0], upconv_planes[1])
        self.upconv5 = upconv(upconv_planes[1], upconv_planes[2])
        self.upconv4 = upconv(upconv_planes[2], upconv_planes[3])
        self.upconv3 = upconv(upconv_planes[3], upconv_planes[4])
        self.upconv2 = upconv(upconv_planes[4], upconv_planes[5])
        self.upconv1 = upconv(upconv_planes[5], upconv_planes[6])

        self.iconv7 = conv(upconv_planes[0] + conv_planes[5], upconv_planes[0])
        self.iconv6 = conv(upconv_planes[1] + conv_planes[4], upconv_planes[1])
        self.iconv5 = conv(upconv_planes[2] + conv_planes[3], upconv_planes[2])
        self.iconv4 = conv(upconv_planes[3] + conv_planes[2], upconv_planes[3])
        self.iconv3 = conv(1 + upconv_planes[4] + conv_planes[1], upconv_planes[4])
        self.iconv2 = conv(1 + upconv_planes[5] + conv_planes[0], upconv_planes[5])
        self.iconv1 = conv(1 + upconv_planes[6], upconv_planes[6])

        self.predict_disp4 = predict_disp(upconv_planes[3])
        self.predict_disp3 = predict_disp(upconv_planes[4])
        self.predict_disp2 = predict_disp(upconv_planes[5])
        self.predict_disp1 = predict_disp(upconv_planes[6])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        x = Quant.apply(x, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        
        out_conv1 = self.conv1(x, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        out_conv1 = Quant.apply(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        
        out_conv2 = self.conv2(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        out_conv2 = Quant.apply(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        
        out_conv3 = self.conv3(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        out_conv3 = Quant.apply(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        
        out_conv4 = self.conv4(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        out_conv4 = Quant.apply(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        
        out_conv5 = self.conv5(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        out_conv5 = Quant.apply(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        
        out_conv6 = self.conv6(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        out_conv6 = Quant.apply(out_conv6, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        
        out_conv7 = self.conv7(out_conv6, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        out_conv7 = Quant.apply(out_conv7, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        
        out_upconv7 = crop_like(self.upconv7(out_conv7), out_conv6)
        out_upconv7 = Quant.apply(out_upconv7, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)
        out_iconv7 = Quant.apply(out_iconv7, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        

        out_upconv6 = crop_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_iconv6 = self.iconv6(concat6)
        out_iconv6 = Quant.apply(out_iconv6, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        

        out_upconv5 = crop_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)
        out_iconv5 = Quant.apply(out_iconv5, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        


        out_upconv4 = crop_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_iconv4 = self.iconv4(concat4)
        out_iconv4 = Quant.apply(out_iconv4, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        
        disp4 = self.alpha * self.predict_disp4(out_iconv4) + self.beta
        disp4 = Quant.apply(disp4, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        

        out_upconv3 = crop_like(self.upconv3(out_iconv4), out_conv2)
        disp4_up = crop_like(F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        disp4_up = Quant.apply(disp4_up, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        
        concat3 = torch.cat((out_upconv3, out_conv2, disp4_up), 1)
        out_iconv3 = self.iconv3(concat3)
        out_iconv3 = Quant.apply(out_iconv3, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        
        disp3 = self.alpha * self.predict_disp3(out_iconv3) + self.beta
        disp3 = Quant.apply(disp3, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        

        out_upconv2 = crop_like(self.upconv2(out_iconv3), out_conv1)
        disp3_up = crop_like(F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        disp3_up = Quant.apply(disp3_up, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        
        
        concat2 = torch.cat((out_upconv2, out_conv1, disp3_up), 1)
        out_iconv2 = self.iconv2(concat2)
        out_iconv2 = Quant.apply(out_iconv2, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        
        disp2 = self.alpha * self.predict_disp2(out_iconv2) + self.beta
        disp2 = Quant.apply(disp2, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        

        out_upconv1 = crop_like(self.upconv1(out_iconv2), x)
        disp2_up = crop_like(F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=False), x)
        disp2_up = Quant.apply(disp2_up, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        
        concat1 = torch.cat((out_upconv1, disp2_up), 1)
        out_iconv1 = self.iconv1(concat1)
        out_iconv1 = Quant.apply(out_iconv1, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        
        disp1 = self.alpha * self.predict_disp1(out_iconv1) + self.beta
        disp1 = Quant.apply(disp1, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        

        if self.training:
            return disp1, disp2, disp3, disp4
        else:
            return disp1, disp2, disp3, disp4



class DispNetS_Q_full_encoding(nn.Module):

    def __init__(self, alpha=10, beta=0.01, type = "fxp", n_exp_en = 5, n_man_en = 10,  n_exp_de = 5, n_man_de = 10, slice_width = 4, mode = "trunc", device = "cuda" ):
        super(DispNetS_Q_full_encoding, self).__init__()

        self.type = type
        self.n_exp_en = n_exp_en
        self.n_man_en = n_man_en
        self.n_exp_de = n_exp_de
        self.n_man_de = n_man_de
        self.slice_width = slice_width

        self.mode = mode
        self.device = device
        self.alpha = alpha
        self.beta = beta
        

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv1 = downsample_conv_quant_encoding(3,              conv_planes[0], kernel_size=7)
        self.conv2 = downsample_conv_quant_encoding(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = downsample_conv_quant_encoding(conv_planes[1], conv_planes[2])
        self.conv4 = downsample_conv_quant_encoding(conv_planes[2], conv_planes[3])
        self.conv5 = downsample_conv_quant_encoding(conv_planes[3], conv_planes[4])
        self.conv6 = downsample_conv_quant_encoding(conv_planes[4], conv_planes[5])
        self.conv7 = downsample_conv_quant_encoding(conv_planes[5], conv_planes[6])

        upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv7 = upconv(conv_planes[6],   upconv_planes[0])
        self.upconv6 = upconv(upconv_planes[0], upconv_planes[1])
        self.upconv5 = upconv(upconv_planes[1], upconv_planes[2])
        self.upconv4 = upconv(upconv_planes[2], upconv_planes[3])
        self.upconv3 = upconv(upconv_planes[3], upconv_planes[4])
        self.upconv2 = upconv(upconv_planes[4], upconv_planes[5])
        self.upconv1 = upconv(upconv_planes[5], upconv_planes[6])

        self.iconv7 = conv(upconv_planes[0] + conv_planes[5], upconv_planes[0])
        self.iconv6 = conv(upconv_planes[1] + conv_planes[4], upconv_planes[1])
        self.iconv5 = conv(upconv_planes[2] + conv_planes[3], upconv_planes[2])
        self.iconv4 = conv(upconv_planes[3] + conv_planes[2], upconv_planes[3])
        self.iconv3 = conv(1 + upconv_planes[4] + conv_planes[1], upconv_planes[4])
        self.iconv2 = conv(1 + upconv_planes[5] + conv_planes[0], upconv_planes[5])
        self.iconv1 = conv(1 + upconv_planes[6], upconv_planes[6])

        self.predict_disp4 = predict_disp(upconv_planes[3])
        self.predict_disp3 = predict_disp(upconv_planes[4])
        self.predict_disp2 = predict_disp(upconv_planes[5])
        self.predict_disp1 = predict_disp(upconv_planes[6])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        # For sparsity check
        total_count_activation = 0
        zero_count_activation = 0

        x = Quant.apply(x, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        
        # Encoder
        out_conv1 = self.conv1(x, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv1(x, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv1(x, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv1 = Quant.apply(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv1 = bin_fxp(out_conv1, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += encoding(bin_out_conv1, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_conv1, self.n_exp_en + self.n_man_en, self.slice_width)[1]


        out_conv2 = self.conv2(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv2(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv2(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv2 = Quant.apply(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv2 = bin_fxp(out_conv2, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += encoding(bin_out_conv2, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_conv2, self.n_exp_en + self.n_man_en, self.slice_width)[1]


        out_conv3 = self.conv3(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv3(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv3(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv3 = Quant.apply(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv3 = bin_fxp(out_conv3, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += encoding(bin_out_conv3, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_conv3, self.n_exp_en + self.n_man_en, self.slice_width)[1]


        out_conv4 = self.conv4(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv4(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv4(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv4 = Quant.apply(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv4 = bin_fxp(out_conv4, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += encoding(bin_out_conv4, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_conv4, self.n_exp_en + self.n_man_en, self.slice_width)[1]

        out_conv5 = self.conv5(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv5(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv5(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv5 = Quant.apply(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv5 = bin_fxp(out_conv5, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += encoding(bin_out_conv5, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_conv5, self.n_exp_en + self.n_man_en, self.slice_width)[1]

        out_conv6 = self.conv6(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv6(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv6(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv6 = Quant.apply(out_conv6, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv6 = bin_fxp(out_conv6, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += encoding(bin_out_conv6, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_conv6, self.n_exp_en + self.n_man_en, self.slice_width)[1]
        
        out_conv7 = self.conv7(out_conv6, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv7(out_conv7, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv7(out_conv7, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv7 = Quant.apply(out_conv7, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv7 = bin_fxp(out_conv7, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += encoding(bin_out_conv7, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_conv7, self.n_exp_en + self.n_man_en, self.slice_width)[1]
        

        # Decoder

        out_upconv7 = crop_like(self.upconv7(out_conv7), out_conv6)
        out_upconv7 = Quant.apply(out_upconv7, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_upconv7 = bin_fxp(out_upconv7, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_out_upconv7, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_upconv7, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)
        out_iconv7 = Quant.apply(out_iconv7, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv7 = bin_fxp(out_iconv7, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_out_iconv7, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_iconv7, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv6 = crop_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_iconv6 = self.iconv6(concat6)
        out_iconv6 = Quant.apply(out_iconv6, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv6 = bin_fxp(out_iconv6, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_out_iconv6, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_iconv6, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv5 = crop_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)
        out_iconv5 = Quant.apply(out_iconv5, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv5 = bin_fxp(out_iconv5, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_out_iconv5, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_iconv5, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        


        out_upconv4 = crop_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_iconv4 = self.iconv4(concat4)
        out_iconv4 = Quant.apply(out_iconv4, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv4 = bin_fxp(out_iconv4, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_out_iconv4, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_iconv4, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        disp4 = self.alpha * self.predict_disp4(out_iconv4) + self.beta
        disp4 = Quant.apply(disp4, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp4 = bin_fxp(disp4, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_disp4, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_disp4, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv3 = crop_like(self.upconv3(out_iconv4), out_conv2)
        disp4_up = crop_like(F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        disp4_up = Quant.apply(disp4_up, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp4_up = bin_fxp(disp4_up, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_disp4_up, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_disp4_up, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        concat3 = torch.cat((out_upconv3, out_conv2, disp4_up), 1)
        out_iconv3 = self.iconv3(concat3)
        out_iconv3 = Quant.apply(out_iconv3, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv3 = bin_fxp(out_iconv3, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_out_iconv3, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_iconv3, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        disp3 = self.alpha * self.predict_disp3(out_iconv3) + self.beta
        disp3 = Quant.apply(disp3, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp3 = bin_fxp(disp3, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_disp3, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_disp3, self.n_exp_de + self.n_man_de, self.slice_width)[1]
    

        out_upconv2 = crop_like(self.upconv2(out_iconv3), out_conv1)
        disp3_up = crop_like(F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        disp3_up = Quant.apply(disp3_up, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp3_up = bin_fxp(disp3_up, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_disp3_up, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_disp3_up, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        
        concat2 = torch.cat((out_upconv2, out_conv1, disp3_up), 1)
        out_iconv2 = self.iconv2(concat2)
        out_iconv2 = Quant.apply(out_iconv2, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv2 = bin_fxp(out_iconv2, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_out_iconv2, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_iconv2, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        disp2 = self.alpha * self.predict_disp2(out_iconv2) + self.beta
        disp2 = Quant.apply(disp2, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp2 = bin_fxp(disp2, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_disp2, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_disp2, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv1 = crop_like(self.upconv1(out_iconv2), x)
        disp2_up = crop_like(F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=False), x)
        disp2_up = Quant.apply(disp2_up, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp2_up = bin_fxp(disp2_up, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_disp2_up, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_disp2_up, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        concat1 = torch.cat((out_upconv1, disp2_up), 1)
        out_iconv1 = self.iconv1(concat1)
        out_iconv1 = Quant.apply(out_iconv1, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_iconv1 = bin_fxp(out_iconv1, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_iconv1, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_iconv1, self.n_exp_de + self.n_man_de, self.slice_width)[1]
    
        disp1 = self.alpha * self.predict_disp1(out_iconv1) + self.beta
        disp1 = Quant.apply(disp1, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp1 = bin_fxp(disp1, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_disp1, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_disp1, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        if self.training:
            return disp1, disp2, disp3, disp4
        else:
            return disp1, disp2, disp3, disp4, total_count_activation, zero_count_activation
        


class DispNetS_Q_full_signed_encoding(nn.Module):

    def __init__(self, alpha=10, beta=0.01, type = "fxp", n_exp_en = 5, n_man_en = 10,  n_exp_de = 5, n_man_de = 10, slice_width = 4, mode = "trunc", device = "cuda" ):
        super(DispNetS_Q_full_signed_encoding, self).__init__()

        self.type = type
        self.n_exp_en = n_exp_en
        self.n_man_en = n_man_en
        self.n_exp_de = n_exp_de
        self.n_man_de = n_man_de
        self.slice_width = slice_width

        self.mode = mode
        self.device = device
        self.alpha = alpha
        self.beta = beta
        

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv1 = downsample_conv_quant_signed_encoding(3,              conv_planes[0], kernel_size=7)
        self.conv2 = downsample_conv_quant_signed_encoding(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = downsample_conv_quant_signed_encoding(conv_planes[1], conv_planes[2])
        self.conv4 = downsample_conv_quant_signed_encoding(conv_planes[2], conv_planes[3])
        self.conv5 = downsample_conv_quant_signed_encoding(conv_planes[3], conv_planes[4])
        self.conv6 = downsample_conv_quant_signed_encoding(conv_planes[4], conv_planes[5])
        self.conv7 = downsample_conv_quant_signed_encoding(conv_planes[5], conv_planes[6])

        upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv7 = upconv(conv_planes[6],   upconv_planes[0])
        self.upconv6 = upconv(upconv_planes[0], upconv_planes[1])
        self.upconv5 = upconv(upconv_planes[1], upconv_planes[2])
        self.upconv4 = upconv(upconv_planes[2], upconv_planes[3])
        self.upconv3 = upconv(upconv_planes[3], upconv_planes[4])
        self.upconv2 = upconv(upconv_planes[4], upconv_planes[5])
        self.upconv1 = upconv(upconv_planes[5], upconv_planes[6])

        self.iconv7 = conv(upconv_planes[0] + conv_planes[5], upconv_planes[0])
        self.iconv6 = conv(upconv_planes[1] + conv_planes[4], upconv_planes[1])
        self.iconv5 = conv(upconv_planes[2] + conv_planes[3], upconv_planes[2])
        self.iconv4 = conv(upconv_planes[3] + conv_planes[2], upconv_planes[3])
        self.iconv3 = conv(1 + upconv_planes[4] + conv_planes[1], upconv_planes[4])
        self.iconv2 = conv(1 + upconv_planes[5] + conv_planes[0], upconv_planes[5])
        self.iconv1 = conv(1 + upconv_planes[6], upconv_planes[6])

        self.predict_disp4 = predict_disp(upconv_planes[3])
        self.predict_disp3 = predict_disp(upconv_planes[4])
        self.predict_disp2 = predict_disp(upconv_planes[5])
        self.predict_disp1 = predict_disp(upconv_planes[6])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        # For sparsity check
        total_count_activation = 0
        zero_count_activation = 0

        x = Quant.apply(x, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        
        
        # Encoder
        out_conv1 = self.conv1(x, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv1(x, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv1(x, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv1 = Quant.apply(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv1 = bin_fxp(out_conv1, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_conv1, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_conv1, self.n_exp_en + self.n_man_en, self.slice_width)[1]


        out_conv2 = self.conv2(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv2(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv2(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv2 = Quant.apply(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv2 = bin_fxp(out_conv2, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_conv2, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_conv2, self.n_exp_en + self.n_man_en, self.slice_width)[1]


        out_conv3 = self.conv3(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv3(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv3(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv3 = Quant.apply(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv3 = bin_fxp(out_conv3, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_conv3, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_conv3, self.n_exp_en + self.n_man_en, self.slice_width)[1]


        out_conv4 = self.conv4(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv4(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv4(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv4 = Quant.apply(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv4 = bin_fxp(out_conv4, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_conv4, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_conv4, self.n_exp_en + self.n_man_en, self.slice_width)[1]

        out_conv5 = self.conv5(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv5(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv5(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv5 = Quant.apply(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv5 = bin_fxp(out_conv5, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_conv5, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_conv5, self.n_exp_en + self.n_man_en, self.slice_width)[1]

        out_conv6 = self.conv6(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv6(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv6(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv6 = Quant.apply(out_conv6, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv6 = bin_fxp(out_conv6, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_conv6, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_conv6, self.n_exp_en + self.n_man_en, self.slice_width)[1]
        
        out_conv7 = self.conv7(out_conv6, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv7(out_conv7, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv7(out_conv7, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv7 = Quant.apply(out_conv7, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv7 = bin_fxp(out_conv7, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_conv7, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_conv7, self.n_exp_en + self.n_man_en, self.slice_width)[1]


        # Decoder

        out_upconv7 = crop_like(self.upconv7(out_conv7), out_conv6)
        out_upconv7 = Quant.apply(out_upconv7, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_upconv7 = bin_fxp(out_upconv7, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_upconv7, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_upconv7, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)
        out_iconv7 = Quant.apply(out_iconv7, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv7 = bin_fxp(out_iconv7, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_iconv7, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_iconv7, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv6 = crop_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_iconv6 = self.iconv6(concat6)
        out_iconv6 = Quant.apply(out_iconv6, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv6 = bin_fxp(out_iconv6, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_iconv6, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_iconv6, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv5 = crop_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)
        out_iconv5 = Quant.apply(out_iconv5, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv5 = bin_fxp(out_iconv5, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_iconv5, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_iconv5, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        


        out_upconv4 = crop_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_iconv4 = self.iconv4(concat4)
        out_iconv4 = Quant.apply(out_iconv4, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv4 = bin_fxp(out_iconv4, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_iconv4, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_iconv4, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        disp4 = self.alpha * self.predict_disp4(out_iconv4) + self.beta
        disp4 = Quant.apply(disp4, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp4 = bin_fxp(disp4, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_disp4, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_disp4, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv3 = crop_like(self.upconv3(out_iconv4), out_conv2)
        disp4_up = crop_like(F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        disp4_up = Quant.apply(disp4_up, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp4_up = bin_fxp(disp4_up, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_disp4_up, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_disp4_up, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        concat3 = torch.cat((out_upconv3, out_conv2, disp4_up), 1)
        out_iconv3 = self.iconv3(concat3)
        out_iconv3 = Quant.apply(out_iconv3, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv3 = bin_fxp(out_iconv3, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_iconv3, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_iconv3, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        disp3 = self.alpha * self.predict_disp3(out_iconv3) + self.beta
        disp3 = Quant.apply(disp3, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp3 = bin_fxp(disp3, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_disp3, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_disp3, self.n_exp_de + self.n_man_de, self.slice_width)[1]
    

        out_upconv2 = crop_like(self.upconv2(out_iconv3), out_conv1)
        disp3_up = crop_like(F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        disp3_up = Quant.apply(disp3_up, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp3_up = bin_fxp(disp3_up, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_disp3_up, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_disp3_up, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        
        concat2 = torch.cat((out_upconv2, out_conv1, disp3_up), 1)
        out_iconv2 = self.iconv2(concat2)
        out_iconv2 = Quant.apply(out_iconv2, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv2 = bin_fxp(out_iconv2, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_iconv2, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_iconv2, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        disp2 = self.alpha * self.predict_disp2(out_iconv2) + self.beta
        disp2 = Quant.apply(disp2, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp2 = bin_fxp(disp2, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_disp2, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_disp2, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv1 = crop_like(self.upconv1(out_iconv2), x)
        disp2_up = crop_like(F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=False), x)
        disp2_up = Quant.apply(disp2_up, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp2_up = bin_fxp(disp2_up, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_disp2_up, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_disp2_up, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        concat1 = torch.cat((out_upconv1, disp2_up), 1)
        out_iconv1 = self.iconv1(concat1)
        out_iconv1 = Quant.apply(out_iconv1, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_iconv1 = bin_fxp(out_iconv1, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_iconv1, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_iconv1, self.n_exp_de + self.n_man_de, self.slice_width)[1]
    
        disp1 = self.alpha * self.predict_disp1(out_iconv1) + self.beta
        disp1 = Quant.apply(disp1, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp1 = bin_fxp(disp1, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_disp1, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_disp1, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        if self.training:
            return disp1, disp2, disp3, disp4
        else:
            return disp1, disp2, disp3, disp4, total_count_activation, zero_count_activation
        


class DispNetS_Q_full_no_encoding(nn.Module):

    def __init__(self, alpha=10, beta=0.01, type = "fxp", n_exp_en = 5, n_man_en = 10,  n_exp_de = 5, n_man_de = 10, slice_width = 4, mode = "trunc", device = "cuda" ):
        super(DispNetS_Q_full_no_encoding, self).__init__()

        self.type = type
        self.n_exp_en = n_exp_en
        self.n_man_en = n_man_en
        self.n_exp_de = n_exp_de
        self.n_man_de = n_man_de
        self.slice_width = slice_width

        self.mode = mode
        self.device = device
        self.alpha = alpha
        self.beta = beta
        

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv1 = downsample_conv_quant_no_encoding(3,              conv_planes[0], kernel_size=7)
        self.conv2 = downsample_conv_quant_no_encoding(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = downsample_conv_quant_no_encoding(conv_planes[1], conv_planes[2])
        self.conv4 = downsample_conv_quant_no_encoding(conv_planes[2], conv_planes[3])
        self.conv5 = downsample_conv_quant_no_encoding(conv_planes[3], conv_planes[4])
        self.conv6 = downsample_conv_quant_no_encoding(conv_planes[4], conv_planes[5])
        self.conv7 = downsample_conv_quant_no_encoding(conv_planes[5], conv_planes[6])

        upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv7 = upconv(conv_planes[6],   upconv_planes[0])
        self.upconv6 = upconv(upconv_planes[0], upconv_planes[1])
        self.upconv5 = upconv(upconv_planes[1], upconv_planes[2])
        self.upconv4 = upconv(upconv_planes[2], upconv_planes[3])
        self.upconv3 = upconv(upconv_planes[3], upconv_planes[4])
        self.upconv2 = upconv(upconv_planes[4], upconv_planes[5])
        self.upconv1 = upconv(upconv_planes[5], upconv_planes[6])

        self.iconv7 = conv(upconv_planes[0] + conv_planes[5], upconv_planes[0])
        self.iconv6 = conv(upconv_planes[1] + conv_planes[4], upconv_planes[1])
        self.iconv5 = conv(upconv_planes[2] + conv_planes[3], upconv_planes[2])
        self.iconv4 = conv(upconv_planes[3] + conv_planes[2], upconv_planes[3])
        self.iconv3 = conv(1 + upconv_planes[4] + conv_planes[1], upconv_planes[4])
        self.iconv2 = conv(1 + upconv_planes[5] + conv_planes[0], upconv_planes[5])
        self.iconv1 = conv(1 + upconv_planes[6], upconv_planes[6])

        self.predict_disp4 = predict_disp(upconv_planes[3])
        self.predict_disp3 = predict_disp(upconv_planes[4])
        self.predict_disp2 = predict_disp(upconv_planes[5])
        self.predict_disp1 = predict_disp(upconv_planes[6])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        # For sparsity check
        total_count_activation = 0
        zero_count_activation = 0

        x = Quant.apply(x, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)


        
        # Encoder
        out_conv1 = self.conv1(x, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv1(x, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv1(x, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv1 = Quant.apply(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv1 = bin_fxp(out_conv1, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_conv1, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_conv1, self.n_exp_en + self.n_man_en, self.slice_width)[1]


        out_conv2 = self.conv2(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv2(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv2(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv2 = Quant.apply(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv2 = bin_fxp(out_conv2, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_conv2, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_conv2, self.n_exp_en + self.n_man_en, self.slice_width)[1]


        out_conv3 = self.conv3(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv3(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv3(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv3 = Quant.apply(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv3 = bin_fxp(out_conv3, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_conv3, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_conv3, self.n_exp_en + self.n_man_en, self.slice_width)[1]


        out_conv4 = self.conv4(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv4(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv4(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv4 = Quant.apply(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv4 = bin_fxp(out_conv4, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_conv4, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_conv4, self.n_exp_en + self.n_man_en, self.slice_width)[1]

        out_conv5 = self.conv5(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv5(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv5(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv5 = Quant.apply(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv5 = bin_fxp(out_conv5, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_conv5, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_conv5, self.n_exp_en + self.n_man_en, self.slice_width)[1]

        out_conv6 = self.conv6(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv6(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv6(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv6 = Quant.apply(out_conv6, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv6 = bin_fxp(out_conv6, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_conv6, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_conv6, self.n_exp_en + self.n_man_en, self.slice_width)[1]
        
        out_conv7 = self.conv7(out_conv6, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv7(out_conv7, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv7(out_conv7, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv7 = Quant.apply(out_conv7, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv7 = bin_fxp(out_conv7, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_conv7, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_conv7, self.n_exp_en + self.n_man_en, self.slice_width)[1]
        

        # Decoder

        out_upconv7 = crop_like(self.upconv7(out_conv7), out_conv6)
        out_upconv7 = Quant.apply(out_upconv7, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_upconv7 = bin_fxp(out_upconv7, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_upconv7, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_upconv7, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)
        out_iconv7 = Quant.apply(out_iconv7, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv7 = bin_fxp(out_iconv7, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_iconv7, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_iconv7, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv6 = crop_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_iconv6 = self.iconv6(concat6)
        out_iconv6 = Quant.apply(out_iconv6, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv6 = bin_fxp(out_iconv6, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_iconv6, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_iconv6, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv5 = crop_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)
        out_iconv5 = Quant.apply(out_iconv5, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv5 = bin_fxp(out_iconv5, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_iconv5, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_iconv5, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        


        out_upconv4 = crop_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_iconv4 = self.iconv4(concat4)
        out_iconv4 = Quant.apply(out_iconv4, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv4 = bin_fxp(out_iconv4, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_iconv4, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_iconv4, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        disp4 = self.alpha * self.predict_disp4(out_iconv4) + self.beta
        disp4 = Quant.apply(disp4, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp4 = bin_fxp(disp4, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_disp4, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_disp4, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv3 = crop_like(self.upconv3(out_iconv4), out_conv2)
        disp4_up = crop_like(F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        disp4_up = Quant.apply(disp4_up, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp4_up = bin_fxp(disp4_up, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_disp4_up, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_disp4_up, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        concat3 = torch.cat((out_upconv3, out_conv2, disp4_up), 1)
        out_iconv3 = self.iconv3(concat3)
        out_iconv3 = Quant.apply(out_iconv3, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv3 = bin_fxp(out_iconv3, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_iconv3, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_iconv3, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        disp3 = self.alpha * self.predict_disp3(out_iconv3) + self.beta
        disp3 = Quant.apply(disp3, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp3 = bin_fxp(disp3, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_disp3, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_disp3, self.n_exp_de + self.n_man_de, self.slice_width)[1]
    

        out_upconv2 = crop_like(self.upconv2(out_iconv3), out_conv1)
        disp3_up = crop_like(F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        disp3_up = Quant.apply(disp3_up, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp3_up = bin_fxp(disp3_up, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_disp3_up, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_disp3_up, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        
        concat2 = torch.cat((out_upconv2, out_conv1, disp3_up), 1)
        out_iconv2 = self.iconv2(concat2)
        out_iconv2 = Quant.apply(out_iconv2, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv2 = bin_fxp(out_iconv2, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_iconv2, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_iconv2, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        disp2 = self.alpha * self.predict_disp2(out_iconv2) + self.beta
        disp2 = Quant.apply(disp2, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp2 = bin_fxp(disp2, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_disp2, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_disp2, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv1 = crop_like(self.upconv1(out_iconv2), x)
        disp2_up = crop_like(F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=False), x)
        disp2_up = Quant.apply(disp2_up, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp2_up = bin_fxp(disp2_up, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_disp2_up, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_disp2_up, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        concat1 = torch.cat((out_upconv1, disp2_up), 1)
        out_iconv1 = self.iconv1(concat1)
        out_iconv1 = Quant.apply(out_iconv1, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_iconv1 = bin_fxp(out_iconv1, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_iconv1, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_iconv1, self.n_exp_de + self.n_man_de, self.slice_width)[1]
    
        disp1 = self.alpha * self.predict_disp1(out_iconv1) + self.beta
        disp1 = Quant.apply(disp1, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp1 = bin_fxp(disp1, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_disp1, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_disp1, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        if self.training:
            return disp1, disp2, disp3, disp4
        else:
            return disp1, disp2, disp3, disp4, total_count_activation, zero_count_activation
        



class DispNetS_Q_full_encoding_leaky(nn.Module):

    def __init__(self, alpha=10, beta=0.01, type = "fxp", n_exp_en = 5, n_man_en = 10,  n_exp_de = 5, n_man_de = 10, slice_width = 4, mode = "trunc", device = "cuda" ):
        super(DispNetS_Q_full_encoding_leaky, self).__init__()

        self.type = type
        self.n_exp_en = n_exp_en
        self.n_man_en = n_man_en
        self.n_exp_de = n_exp_de
        self.n_man_de = n_man_de
        self.slice_width = slice_width

        self.mode = mode
        self.device = device
        self.alpha = alpha
        self.beta = beta
        

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv1 = downsample_conv_quant_encoding_leaky(3,              conv_planes[0], kernel_size=7)
        self.conv2 = downsample_conv_quant_encoding_leaky(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = downsample_conv_quant_encoding_leaky(conv_planes[1], conv_planes[2])
        self.conv4 = downsample_conv_quant_encoding_leaky(conv_planes[2], conv_planes[3])
        self.conv5 = downsample_conv_quant_encoding_leaky(conv_planes[3], conv_planes[4])
        self.conv6 = downsample_conv_quant_encoding_leaky(conv_planes[4], conv_planes[5])
        self.conv7 = downsample_conv_quant_encoding_leaky(conv_planes[5], conv_planes[6])

        upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv7 = upconv_leaky(conv_planes[6],   upconv_planes[0])
        self.upconv6 = upconv_leaky(upconv_planes[0], upconv_planes[1])
        self.upconv5 = upconv_leaky(upconv_planes[1], upconv_planes[2])
        self.upconv4 = upconv_leaky(upconv_planes[2], upconv_planes[3])
        self.upconv3 = upconv_leaky(upconv_planes[3], upconv_planes[4])
        self.upconv2 = upconv_leaky(upconv_planes[4], upconv_planes[5])
        self.upconv1 = upconv_leaky(upconv_planes[5], upconv_planes[6])

        self.iconv7 = conv_leaky(upconv_planes[0] + conv_planes[5], upconv_planes[0])
        self.iconv6 = conv_leaky(upconv_planes[1] + conv_planes[4], upconv_planes[1])
        self.iconv5 = conv_leaky(upconv_planes[2] + conv_planes[3], upconv_planes[2])
        self.iconv4 = conv_leaky(upconv_planes[3] + conv_planes[2], upconv_planes[3])
        self.iconv3 = conv_leaky(1 + upconv_planes[4] + conv_planes[1], upconv_planes[4])
        self.iconv2 = conv_leaky(1 + upconv_planes[5] + conv_planes[0], upconv_planes[5])
        self.iconv1 = conv_leaky(1 + upconv_planes[6], upconv_planes[6])

        self.predict_disp4 = predict_disp(upconv_planes[3])
        self.predict_disp3 = predict_disp(upconv_planes[4])
        self.predict_disp2 = predict_disp(upconv_planes[5])
        self.predict_disp1 = predict_disp(upconv_planes[6])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        # For sparsity check
        total_count_activation = 0
        zero_count_activation = 0

        x = Quant.apply(x, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        
        # Encoder
        out_conv1 = self.conv1(x, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv1(x, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv1(x, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv1 = Quant.apply(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv1 = bin_fxp(out_conv1, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += encoding(bin_out_conv1, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_conv1, self.n_exp_en + self.n_man_en, self.slice_width)[1]


        out_conv2 = self.conv2(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv2(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv2(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv2 = Quant.apply(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv2 = bin_fxp(out_conv2, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += encoding(bin_out_conv2, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_conv2, self.n_exp_en + self.n_man_en, self.slice_width)[1]


        out_conv3 = self.conv3(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv3(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv3(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv3 = Quant.apply(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv3 = bin_fxp(out_conv3, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += encoding(bin_out_conv3, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_conv3, self.n_exp_en + self.n_man_en, self.slice_width)[1]


        out_conv4 = self.conv4(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv4(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv4(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv4 = Quant.apply(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv4 = bin_fxp(out_conv4, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += encoding(bin_out_conv4, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_conv4, self.n_exp_en + self.n_man_en, self.slice_width)[1]

        out_conv5 = self.conv5(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv5(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv5(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv5 = Quant.apply(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv5 = bin_fxp(out_conv5, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += encoding(bin_out_conv5, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_conv5, self.n_exp_en + self.n_man_en, self.slice_width)[1]

        out_conv6 = self.conv6(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv6(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv6(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv6 = Quant.apply(out_conv6, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv6 = bin_fxp(out_conv6, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += encoding(bin_out_conv6, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_conv6, self.n_exp_en + self.n_man_en, self.slice_width)[1]
        
        out_conv7 = self.conv7(out_conv6, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv7(out_conv7, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv7(out_conv7, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv7 = Quant.apply(out_conv7, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv7 = bin_fxp(out_conv7, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += encoding(bin_out_conv7, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_conv7, self.n_exp_en + self.n_man_en, self.slice_width)[1]
        

        # Decoder

        out_upconv7 = crop_like(self.upconv7(out_conv7), out_conv6)
        out_upconv7 = Quant.apply(out_upconv7, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_upconv7 = bin_fxp(out_upconv7, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_out_upconv7, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_upconv7, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)
        out_iconv7 = Quant.apply(out_iconv7, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv7 = bin_fxp(out_iconv7, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_out_iconv7, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_iconv7, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv6 = crop_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_iconv6 = self.iconv6(concat6)
        out_iconv6 = Quant.apply(out_iconv6, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv6 = bin_fxp(out_iconv6, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_out_iconv6, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_iconv6, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv5 = crop_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)
        out_iconv5 = Quant.apply(out_iconv5, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv5 = bin_fxp(out_iconv5, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_out_iconv5, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_iconv5, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        


        out_upconv4 = crop_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_iconv4 = self.iconv4(concat4)
        out_iconv4 = Quant.apply(out_iconv4, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv4 = bin_fxp(out_iconv4, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_out_iconv4, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_iconv4, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        disp4 = self.alpha * self.predict_disp4(out_iconv4) + self.beta
        disp4 = Quant.apply(disp4, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp4 = bin_fxp(disp4, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_disp4, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_disp4, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv3 = crop_like(self.upconv3(out_iconv4), out_conv2)
        disp4_up = crop_like(F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        disp4_up = Quant.apply(disp4_up, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp4_up = bin_fxp(disp4_up, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_disp4_up, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_disp4_up, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        concat3 = torch.cat((out_upconv3, out_conv2, disp4_up), 1)
        out_iconv3 = self.iconv3(concat3)
        out_iconv3 = Quant.apply(out_iconv3, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv3 = bin_fxp(out_iconv3, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_out_iconv3, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_iconv3, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        disp3 = self.alpha * self.predict_disp3(out_iconv3) + self.beta
        disp3 = Quant.apply(disp3, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp3 = bin_fxp(disp3, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_disp3, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_disp3, self.n_exp_de + self.n_man_de, self.slice_width)[1]
    

        out_upconv2 = crop_like(self.upconv2(out_iconv3), out_conv1)
        disp3_up = crop_like(F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        disp3_up = Quant.apply(disp3_up, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp3_up = bin_fxp(disp3_up, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_disp3_up, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_disp3_up, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        
        concat2 = torch.cat((out_upconv2, out_conv1, disp3_up), 1)
        out_iconv2 = self.iconv2(concat2)
        out_iconv2 = Quant.apply(out_iconv2, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv2 = bin_fxp(out_iconv2, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_out_iconv2, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_out_iconv2, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        disp2 = self.alpha * self.predict_disp2(out_iconv2) + self.beta
        disp2 = Quant.apply(disp2, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp2 = bin_fxp(disp2, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_disp2, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_disp2, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv1 = crop_like(self.upconv1(out_iconv2), x)
        disp2_up = crop_like(F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=False), x)
        disp2_up = Quant.apply(disp2_up, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp2_up = bin_fxp(disp2_up, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_disp2_up, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_disp2_up, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        concat1 = torch.cat((out_upconv1, disp2_up), 1)
        out_iconv1 = self.iconv1(concat1)
        out_iconv1 = Quant.apply(out_iconv1, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_iconv1 = bin_fxp(out_iconv1, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_iconv1, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_iconv1, self.n_exp_de + self.n_man_de, self.slice_width)[1]
    
        disp1 = self.alpha * self.predict_disp1(out_iconv1) + self.beta
        disp1 = Quant.apply(disp1, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp1 = bin_fxp(disp1, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += encoding(bin_disp1, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += encoding(bin_disp1, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        if self.training:
            return disp1, disp2, disp3, disp4
        else:
            return disp1, disp2, disp3, disp4, total_count_activation, zero_count_activation
        


class DispNetS_Q_full_signed_encoding_leaky(nn.Module):

    def __init__(self, alpha=10, beta=0.01, type = "fxp", n_exp_en = 5, n_man_en = 10,  n_exp_de = 5, n_man_de = 10, slice_width = 4, mode = "trunc", device = "cuda" ):
        super(DispNetS_Q_full_signed_encoding_leaky, self).__init__()

        self.type = type
        self.n_exp_en = n_exp_en
        self.n_man_en = n_man_en
        self.n_exp_de = n_exp_de
        self.n_man_de = n_man_de
        self.slice_width = slice_width

        self.mode = mode
        self.device = device
        self.alpha = alpha
        self.beta = beta
        

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv1 = downsample_conv_quant_signed_encoding_leaky(3,              conv_planes[0], kernel_size=7)
        self.conv2 = downsample_conv_quant_signed_encoding_leaky(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = downsample_conv_quant_signed_encoding_leaky(conv_planes[1], conv_planes[2])
        self.conv4 = downsample_conv_quant_signed_encoding_leaky(conv_planes[2], conv_planes[3])
        self.conv5 = downsample_conv_quant_signed_encoding_leaky(conv_planes[3], conv_planes[4])
        self.conv6 = downsample_conv_quant_signed_encoding_leaky(conv_planes[4], conv_planes[5])
        self.conv7 = downsample_conv_quant_signed_encoding_leaky(conv_planes[5], conv_planes[6])

        upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv7 = upconv_leaky(conv_planes[6],   upconv_planes[0])
        self.upconv6 = upconv_leaky(upconv_planes[0], upconv_planes[1])
        self.upconv5 = upconv_leaky(upconv_planes[1], upconv_planes[2])
        self.upconv4 = upconv_leaky(upconv_planes[2], upconv_planes[3])
        self.upconv3 = upconv_leaky(upconv_planes[3], upconv_planes[4])
        self.upconv2 = upconv_leaky(upconv_planes[4], upconv_planes[5])
        self.upconv1 = upconv_leaky(upconv_planes[5], upconv_planes[6])

        self.iconv7 = conv_leaky(upconv_planes[0] + conv_planes[5], upconv_planes[0])
        self.iconv6 = conv_leaky(upconv_planes[1] + conv_planes[4], upconv_planes[1])
        self.iconv5 = conv_leaky(upconv_planes[2] + conv_planes[3], upconv_planes[2])
        self.iconv4 = conv_leaky(upconv_planes[3] + conv_planes[2], upconv_planes[3])
        self.iconv3 = conv_leaky(1 + upconv_planes[4] + conv_planes[1], upconv_planes[4])
        self.iconv2 = conv_leaky(1 + upconv_planes[5] + conv_planes[0], upconv_planes[5])
        self.iconv1 = conv_leaky(1 + upconv_planes[6], upconv_planes[6])

        self.predict_disp4 = predict_disp(upconv_planes[3])
        self.predict_disp3 = predict_disp(upconv_planes[4])
        self.predict_disp2 = predict_disp(upconv_planes[5])
        self.predict_disp1 = predict_disp(upconv_planes[6])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        # For sparsity check
        total_count_activation = 0
        zero_count_activation = 0

        x = Quant.apply(x, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        
        # Encoder
        out_conv1 = self.conv1(x, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv1(x, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv1(x, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv1 = Quant.apply(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv1 = bin_fxp(out_conv1, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_conv1, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_conv1, self.n_exp_en + self.n_man_en, self.slice_width)[1]


        out_conv2 = self.conv2(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv2(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv2(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv2 = Quant.apply(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv2 = bin_fxp(out_conv2, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_conv2, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_conv2, self.n_exp_en + self.n_man_en, self.slice_width)[1]


        out_conv3 = self.conv3(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv3(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv3(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv3 = Quant.apply(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv3 = bin_fxp(out_conv3, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_conv3, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_conv3, self.n_exp_en + self.n_man_en, self.slice_width)[1]


        out_conv4 = self.conv4(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv4(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv4(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv4 = Quant.apply(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv4 = bin_fxp(out_conv4, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_conv4, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_conv4, self.n_exp_en + self.n_man_en, self.slice_width)[1]

        out_conv5 = self.conv5(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv5(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv5(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv5 = Quant.apply(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv5 = bin_fxp(out_conv5, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_conv5, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_conv5, self.n_exp_en + self.n_man_en, self.slice_width)[1]

        out_conv6 = self.conv6(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv6(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv6(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv6 = Quant.apply(out_conv6, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv6 = bin_fxp(out_conv6, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_conv6, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_conv6, self.n_exp_en + self.n_man_en, self.slice_width)[1]
        
        out_conv7 = self.conv7(out_conv6, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv7(out_conv7, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv7(out_conv7, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv7 = Quant.apply(out_conv7, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv7 = bin_fxp(out_conv7, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_conv7, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_conv7, self.n_exp_en + self.n_man_en, self.slice_width)[1]
        

        # Decoder

        out_upconv7 = crop_like(self.upconv7(out_conv7), out_conv6)
        out_upconv7 = Quant.apply(out_upconv7, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_upconv7 = bin_fxp(out_upconv7, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_upconv7, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_upconv7, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)
        out_iconv7 = Quant.apply(out_iconv7, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv7 = bin_fxp(out_iconv7, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_iconv7, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_iconv7, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv6 = crop_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_iconv6 = self.iconv6(concat6)
        out_iconv6 = Quant.apply(out_iconv6, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv6 = bin_fxp(out_iconv6, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_iconv6, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_iconv6, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv5 = crop_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)
        out_iconv5 = Quant.apply(out_iconv5, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv5 = bin_fxp(out_iconv5, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_iconv5, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_iconv5, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        


        out_upconv4 = crop_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_iconv4 = self.iconv4(concat4)
        out_iconv4 = Quant.apply(out_iconv4, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv4 = bin_fxp(out_iconv4, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_iconv4, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_iconv4, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        disp4 = self.alpha * self.predict_disp4(out_iconv4) + self.beta
        disp4 = Quant.apply(disp4, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp4 = bin_fxp(disp4, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_disp4, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_disp4, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv3 = crop_like(self.upconv3(out_iconv4), out_conv2)
        disp4_up = crop_like(F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        disp4_up = Quant.apply(disp4_up, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp4_up = bin_fxp(disp4_up, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_disp4_up, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_disp4_up, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        concat3 = torch.cat((out_upconv3, out_conv2, disp4_up), 1)
        out_iconv3 = self.iconv3(concat3)
        out_iconv3 = Quant.apply(out_iconv3, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv3 = bin_fxp(out_iconv3, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_iconv3, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_iconv3, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        disp3 = self.alpha * self.predict_disp3(out_iconv3) + self.beta
        disp3 = Quant.apply(disp3, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp3 = bin_fxp(disp3, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_disp3, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_disp3, self.n_exp_de + self.n_man_de, self.slice_width)[1]
    

        out_upconv2 = crop_like(self.upconv2(out_iconv3), out_conv1)
        disp3_up = crop_like(F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        disp3_up = Quant.apply(disp3_up, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp3_up = bin_fxp(disp3_up, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_disp3_up, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_disp3_up, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        
        concat2 = torch.cat((out_upconv2, out_conv1, disp3_up), 1)
        out_iconv2 = self.iconv2(concat2)
        out_iconv2 = Quant.apply(out_iconv2, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv2 = bin_fxp(out_iconv2, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_out_iconv2, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_out_iconv2, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        disp2 = self.alpha * self.predict_disp2(out_iconv2) + self.beta
        disp2 = Quant.apply(disp2, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp2 = bin_fxp(disp2, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_disp2, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_disp2, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv1 = crop_like(self.upconv1(out_iconv2), x)
        disp2_up = crop_like(F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=False), x)
        disp2_up = Quant.apply(disp2_up, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp2_up = bin_fxp(disp2_up, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_disp2_up, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_disp2_up, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        concat1 = torch.cat((out_upconv1, disp2_up), 1)
        out_iconv1 = self.iconv1(concat1)
        out_iconv1 = Quant.apply(out_iconv1, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_iconv1 = bin_fxp(out_iconv1, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_iconv1, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_iconv1, self.n_exp_de + self.n_man_de, self.slice_width)[1]
    
        disp1 = self.alpha * self.predict_disp1(out_iconv1) + self.beta
        disp1 = Quant.apply(disp1, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp1 = bin_fxp(disp1, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += signed_encoding(bin_disp1, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += signed_encoding(bin_disp1, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        if self.training:
            return disp1, disp2, disp3, disp4
        else:
            return disp1, disp2, disp3, disp4, total_count_activation, zero_count_activation
        


class DispNetS_Q_full_no_encoding(nn.Module):

    def __init__(self, alpha=10, beta=0.01, type = "fxp", n_exp_en = 5, n_man_en = 10,  n_exp_de = 5, n_man_de = 10, slice_width = 4, mode = "trunc", device = "cuda" ):
        super(DispNetS_Q_full_no_encoding, self).__init__()

        self.type = type
        self.n_exp_en = n_exp_en
        self.n_man_en = n_man_en
        self.n_exp_de = n_exp_de
        self.n_man_de = n_man_de
        self.slice_width = slice_width

        self.mode = mode
        self.device = device
        self.alpha = alpha
        self.beta = beta
        

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv1 = downsample_conv(3,              conv_planes[0], kernel_size=7)
        self.conv2 = downsample_conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = downsample_conv(conv_planes[1], conv_planes[2])
        self.conv4 = downsample_conv(conv_planes[2], conv_planes[3])
        self.conv5 = downsample_conv(conv_planes[3], conv_planes[4])
        self.conv6 = downsample_conv(conv_planes[4], conv_planes[5])
        self.conv7 = downsample_conv(conv_planes[5], conv_planes[6])

        upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv7 = upconv(conv_planes[6],   upconv_planes[0])
        self.upconv6 = upconv(upconv_planes[0], upconv_planes[1])
        self.upconv5 = upconv(upconv_planes[1], upconv_planes[2])
        self.upconv4 = upconv(upconv_planes[2], upconv_planes[3])
        self.upconv3 = upconv(upconv_planes[3], upconv_planes[4])
        self.upconv2 = upconv(upconv_planes[4], upconv_planes[5])
        self.upconv1 = upconv(upconv_planes[5], upconv_planes[6])

        self.iconv7 = conv(upconv_planes[0] + conv_planes[5], upconv_planes[0])
        self.iconv6 = conv(upconv_planes[1] + conv_planes[4], upconv_planes[1])
        self.iconv5 = conv(upconv_planes[2] + conv_planes[3], upconv_planes[2])
        self.iconv4 = conv(upconv_planes[3] + conv_planes[2], upconv_planes[3])
        self.iconv3 = conv(1 + upconv_planes[4] + conv_planes[1], upconv_planes[4])
        self.iconv2 = conv(1 + upconv_planes[5] + conv_planes[0], upconv_planes[5])
        self.iconv1 = conv(1 + upconv_planes[6], upconv_planes[6])

        self.predict_disp4 = predict_disp(upconv_planes[3])
        self.predict_disp3 = predict_disp(upconv_planes[4])
        self.predict_disp2 = predict_disp(upconv_planes[5])
        self.predict_disp1 = predict_disp(upconv_planes[6])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        # For sparsity check
        total_count_activation = 0
        zero_count_activation = 0

        x = Quant.apply(x, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        
        # Encoder
        out_conv1 = self.conv1(x)
        out_conv1 = Quant.apply(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv1 = bin_fxp(out_conv1, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_conv1, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_conv1, self.n_exp_en + self.n_man_en, self.slice_width)[1]


        out_conv2 = self.conv2(out_conv1)
        out_conv2 = Quant.apply(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv2 = bin_fxp(out_conv2, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_conv2, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_conv2, self.n_exp_en + self.n_man_en, self.slice_width)[1]


        out_conv3 = self.conv3(out_conv2)
        out_conv3 = Quant.apply(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv3 = bin_fxp(out_conv3, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_conv3, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_conv3, self.n_exp_en + self.n_man_en, self.slice_width)[1]


        out_conv4 = self.conv4(out_conv3)
        out_conv4 = Quant.apply(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv4 = bin_fxp(out_conv4, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_conv4, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_conv4, self.n_exp_en + self.n_man_en, self.slice_width)[1]

        out_conv5 = self.conv5(out_conv4)
        out_conv5 = Quant.apply(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv5 = bin_fxp(out_conv5, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_conv5, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_conv5, self.n_exp_en + self.n_man_en, self.slice_width)[1]

        out_conv6 = self.conv6(out_conv5)
        out_conv6 = Quant.apply(out_conv6, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv6 = bin_fxp(out_conv6, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_conv6, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_conv6, self.n_exp_en + self.n_man_en, self.slice_width)[1]
        
        out_conv7 = self.conv7(out_conv6)
        out_conv7 = Quant.apply(out_conv7, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv7 = bin_fxp(out_conv7, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_conv7, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_conv7, self.n_exp_en + self.n_man_en, self.slice_width)[1]
        

        # Decoder

        out_upconv7 = crop_like(self.upconv7(out_conv7), out_conv6)
        out_upconv7 = Quant.apply(out_upconv7, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_upconv7 = bin_fxp(out_upconv7, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_upconv7, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_upconv7, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)
        out_iconv7 = Quant.apply(out_iconv7, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv7 = bin_fxp(out_iconv7, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_iconv7, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_iconv7, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv6 = crop_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_iconv6 = self.iconv6(concat6)
        out_iconv6 = Quant.apply(out_iconv6, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv6 = bin_fxp(out_iconv6, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_iconv6, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_iconv6, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv5 = crop_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)
        out_iconv5 = Quant.apply(out_iconv5, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv5 = bin_fxp(out_iconv5, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_iconv5, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_iconv5, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        


        out_upconv4 = crop_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_iconv4 = self.iconv4(concat4)
        out_iconv4 = Quant.apply(out_iconv4, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv4 = bin_fxp(out_iconv4, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_iconv4, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_iconv4, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        disp4 = self.alpha * self.predict_disp4(out_iconv4) + self.beta
        disp4 = Quant.apply(disp4, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp4 = bin_fxp(disp4, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_disp4, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_disp4, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv3 = crop_like(self.upconv3(out_iconv4), out_conv2)
        disp4_up = crop_like(F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        disp4_up = Quant.apply(disp4_up, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp4_up = bin_fxp(disp4_up, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_disp4_up, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_disp4_up, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        concat3 = torch.cat((out_upconv3, out_conv2, disp4_up), 1)
        out_iconv3 = self.iconv3(concat3)
        out_iconv3 = Quant.apply(out_iconv3, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv3 = bin_fxp(out_iconv3, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_iconv3, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_iconv3, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        disp3 = self.alpha * self.predict_disp3(out_iconv3) + self.beta
        disp3 = Quant.apply(disp3, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp3 = bin_fxp(disp3, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_disp3, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_disp3, self.n_exp_de + self.n_man_de, self.slice_width)[1]
    

        out_upconv2 = crop_like(self.upconv2(out_iconv3), out_conv1)
        disp3_up = crop_like(F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        disp3_up = Quant.apply(disp3_up, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp3_up = bin_fxp(disp3_up, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_disp3_up, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_disp3_up, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        
        concat2 = torch.cat((out_upconv2, out_conv1, disp3_up), 1)
        out_iconv2 = self.iconv2(concat2)
        out_iconv2 = Quant.apply(out_iconv2, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv2 = bin_fxp(out_iconv2, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_iconv2, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_iconv2, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        disp2 = self.alpha * self.predict_disp2(out_iconv2) + self.beta
        disp2 = Quant.apply(disp2, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp2 = bin_fxp(disp2, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_disp2, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_disp2, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv1 = crop_like(self.upconv1(out_iconv2), x)
        disp2_up = crop_like(F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=False), x)
        disp2_up = Quant.apply(disp2_up, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp2_up = bin_fxp(disp2_up, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_disp2_up, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_disp2_up, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        concat1 = torch.cat((out_upconv1, disp2_up), 1)
        out_iconv1 = self.iconv1(concat1)
        out_iconv1 = Quant.apply(out_iconv1, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_iconv1 = bin_fxp(out_iconv1, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_iconv1, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_iconv1, self.n_exp_de + self.n_man_de, self.slice_width)[1]
    
        disp1 = self.alpha * self.predict_disp1(out_iconv1) + self.beta
        disp1 = Quant.apply(disp1, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp1 = bin_fxp(disp1, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_disp1, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_disp1, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        if self.training:
            return disp1, disp2, disp3, disp4
        else:
            return disp1, disp2, disp3, disp4, total_count_activation, zero_count_activation
        


class DispNetS_Q_full_no_encoding_leaky(nn.Module):

    def __init__(self, alpha=10, beta=0.01, type = "fxp", n_exp_en = 5, n_man_en = 10,  n_exp_de = 5, n_man_de = 10, slice_width = 4, mode = "trunc", device = "cuda" ):
        super(DispNetS_Q_full_no_encoding_leaky, self).__init__()

        self.type = type
        self.n_exp_en = n_exp_en
        self.n_man_en = n_man_en
        self.n_exp_de = n_exp_de
        self.n_man_de = n_man_de
        self.slice_width = slice_width

        self.mode = mode
        self.device = device
        self.alpha = alpha
        self.beta = beta
        

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv1 = downsample_conv_quant_no_encoding_leaky(3,              conv_planes[0], kernel_size=7)
        self.conv2 = downsample_conv_quant_no_encoding_leaky(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = downsample_conv_quant_no_encoding_leaky(conv_planes[1], conv_planes[2])
        self.conv4 = downsample_conv_quant_no_encoding_leaky(conv_planes[2], conv_planes[3])
        self.conv5 = downsample_conv_quant_no_encoding_leaky(conv_planes[3], conv_planes[4])
        self.conv6 = downsample_conv_quant_no_encoding_leaky(conv_planes[4], conv_planes[5])
        self.conv7 = downsample_conv_quant_no_encoding_leaky(conv_planes[5], conv_planes[6])

        upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv7 = upconv_leaky(conv_planes[6],   upconv_planes[0])
        self.upconv6 = upconv_leaky(upconv_planes[0], upconv_planes[1])
        self.upconv5 = upconv_leaky(upconv_planes[1], upconv_planes[2])
        self.upconv4 = upconv_leaky(upconv_planes[2], upconv_planes[3])
        self.upconv3 = upconv_leaky(upconv_planes[3], upconv_planes[4])
        self.upconv2 = upconv_leaky(upconv_planes[4], upconv_planes[5])
        self.upconv1 = upconv_leaky(upconv_planes[5], upconv_planes[6])

        self.iconv7 = conv_leaky(upconv_planes[0] + conv_planes[5], upconv_planes[0])
        self.iconv6 = conv_leaky(upconv_planes[1] + conv_planes[4], upconv_planes[1])
        self.iconv5 = conv_leaky(upconv_planes[2] + conv_planes[3], upconv_planes[2])
        self.iconv4 = conv_leaky(upconv_planes[3] + conv_planes[2], upconv_planes[3])
        self.iconv3 = conv_leaky(1 + upconv_planes[4] + conv_planes[1], upconv_planes[4])
        self.iconv2 = conv_leaky(1 + upconv_planes[5] + conv_planes[0], upconv_planes[5])
        self.iconv1 = conv_leaky(1 + upconv_planes[6], upconv_planes[6])

        self.predict_disp4 = predict_disp(upconv_planes[3])
        self.predict_disp3 = predict_disp(upconv_planes[4])
        self.predict_disp2 = predict_disp(upconv_planes[5])
        self.predict_disp1 = predict_disp(upconv_planes[6])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        # For sparsity check
        total_count_activation = 0
        zero_count_activation = 0

        x = Quant.apply(x, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        
        # Encoder
        out_conv1 = self.conv1(x, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv1(x, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv1(x, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv1 = Quant.apply(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv1 = bin_fxp(out_conv1, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_conv1, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_conv1, self.n_exp_en + self.n_man_en, self.slice_width)[1]


        out_conv2 = self.conv2(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv2(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv2(out_conv1, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv2 = Quant.apply(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv2 = bin_fxp(out_conv2, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_conv2, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_conv2, self.n_exp_en + self.n_man_en, self.slice_width)[1]


        out_conv3 = self.conv3(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv3(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv3(out_conv2, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv3 = Quant.apply(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv3 = bin_fxp(out_conv3, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_conv3, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_conv3, self.n_exp_en + self.n_man_en, self.slice_width)[1]


        out_conv4 = self.conv4(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv4(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv4(out_conv3, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv4 = Quant.apply(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv4 = bin_fxp(out_conv4, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_conv4, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_conv4, self.n_exp_en + self.n_man_en, self.slice_width)[1]

        out_conv5 = self.conv5(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv5(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv5(out_conv4, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv5 = Quant.apply(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv5 = bin_fxp(out_conv5, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_conv5, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_conv5, self.n_exp_en + self.n_man_en, self.slice_width)[1]

        out_conv6 = self.conv6(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv6(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv6(out_conv5, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv6 = Quant.apply(out_conv6, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv6 = bin_fxp(out_conv6, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_conv6, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_conv6, self.n_exp_en + self.n_man_en, self.slice_width)[1]
        
        out_conv7 = self.conv7(out_conv6, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[0]
        total_count_activation += self.conv7(out_conv7, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[1]
        zero_count_activation += self.conv7(out_conv7, self.type, self.n_exp_en, self.n_man_en, self.slice_width, self.mode, self.device)[2]
        out_conv7 = Quant.apply(out_conv7, self.type, self.n_exp_en, self.n_man_en, self.mode, self.device)
        bin_out_conv7 = bin_fxp(out_conv7, self.n_exp_en, self.n_man_en, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_conv7, self.n_exp_en + self.n_man_en, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_conv7, self.n_exp_en + self.n_man_en, self.slice_width)[1]
        

        # Decoder

        out_upconv7 = crop_like(self.upconv7(out_conv7), out_conv6)
        out_upconv7 = Quant.apply(out_upconv7, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_upconv7 = bin_fxp(out_upconv7, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_upconv7, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_upconv7, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)
        out_iconv7 = Quant.apply(out_iconv7, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv7 = bin_fxp(out_iconv7, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_iconv7, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_iconv7, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv6 = crop_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_iconv6 = self.iconv6(concat6)
        out_iconv6 = Quant.apply(out_iconv6, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv6 = bin_fxp(out_iconv6, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_iconv6, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_iconv6, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv5 = crop_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)
        out_iconv5 = Quant.apply(out_iconv5, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv5 = bin_fxp(out_iconv5, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_iconv5, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_iconv5, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        


        out_upconv4 = crop_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_iconv4 = self.iconv4(concat4)
        out_iconv4 = Quant.apply(out_iconv4, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv4 = bin_fxp(out_iconv4, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_iconv4, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_iconv4, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        disp4 = self.alpha * self.predict_disp4(out_iconv4) + self.beta
        disp4 = Quant.apply(disp4, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp4 = bin_fxp(disp4, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_disp4, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_disp4, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv3 = crop_like(self.upconv3(out_iconv4), out_conv2)
        disp4_up = crop_like(F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        disp4_up = Quant.apply(disp4_up, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp4_up = bin_fxp(disp4_up, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_disp4_up, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_disp4_up, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        concat3 = torch.cat((out_upconv3, out_conv2, disp4_up), 1)
        out_iconv3 = self.iconv3(concat3)
        out_iconv3 = Quant.apply(out_iconv3, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv3 = bin_fxp(out_iconv3, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_iconv3, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_iconv3, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        disp3 = self.alpha * self.predict_disp3(out_iconv3) + self.beta
        disp3 = Quant.apply(disp3, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp3 = bin_fxp(disp3, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_disp3, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_disp3, self.n_exp_de + self.n_man_de, self.slice_width)[1]
    

        out_upconv2 = crop_like(self.upconv2(out_iconv3), out_conv1)
        disp3_up = crop_like(F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        disp3_up = Quant.apply(disp3_up, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp3_up = bin_fxp(disp3_up, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_disp3_up, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_disp3_up, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        
        concat2 = torch.cat((out_upconv2, out_conv1, disp3_up), 1)
        out_iconv2 = self.iconv2(concat2)
        out_iconv2 = Quant.apply(out_iconv2, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_out_iconv2 = bin_fxp(out_iconv2, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_out_iconv2, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_out_iconv2, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        disp2 = self.alpha * self.predict_disp2(out_iconv2) + self.beta
        disp2 = Quant.apply(disp2, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp2 = bin_fxp(disp2, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_disp2, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_disp2, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        out_upconv1 = crop_like(self.upconv1(out_iconv2), x)
        disp2_up = crop_like(F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=False), x)
        disp2_up = Quant.apply(disp2_up, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp2_up = bin_fxp(disp2_up, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_disp2_up, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_disp2_up, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        
        concat1 = torch.cat((out_upconv1, disp2_up), 1)
        out_iconv1 = self.iconv1(concat1)
        out_iconv1 = Quant.apply(out_iconv1, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_iconv1 = bin_fxp(out_iconv1, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_iconv1, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_iconv1, self.n_exp_de + self.n_man_de, self.slice_width)[1]
    
        disp1 = self.alpha * self.predict_disp1(out_iconv1) + self.beta
        disp1 = Quant.apply(disp1, self.type, self.n_exp_de, self.n_man_de, self.mode, self.device)
        bin_disp1 = bin_fxp(disp1, self.n_exp_de, self.n_man_de, self.mode, self.device)
        total_count_activation += no_encoding(bin_disp1, self.n_exp_de + self.n_man_de, self.slice_width)[0]
        zero_count_activation += no_encoding(bin_disp1, self.n_exp_de + self.n_man_de, self.slice_width)[1]
        

        if self.training:
            return disp1, disp2, disp3, disp4
        else:
            return disp1, disp2, disp3, disp4, total_count_activation, zero_count_activation