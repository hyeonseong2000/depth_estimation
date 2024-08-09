import torch
from torch.autograd import Function
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import argparse
import os


# Floating point to Fixed point
def fxp_quant(input, n_int, n_frac, mode="trunc", device = "cpu"):
    # n_int includes Sign Bit (2's complement)
    max_val = (2** (n_int)) - (2**(-n_frac))
    min_val = -max_val
    sf = 2 ** n_frac #scaling factor = 2**(n_frac)

    assert mode in ["trunc", "round", "stochastic"] , "Quantize Mode Must be 'trunc' or 'round' or 'stochastic'"
    
    input = input.to(device)
    #abs_input = torch.abs(input)
    #max_input = torch.max(abs_input)
    #norm_input = input / max_input # Normalized input
    #input = norm_input * (2 ** (n_int - 1)) # Scaling Norm. input, if n_int = 5 -> -16 < input < 16
    
    # Restrict the number with given bit precision (Fractional Width)
    # Quantization Rules
    if(mode == "trunc"):
        input_trunc = torch.floor(input * sf)/sf # Truncate Fractional Bit
    elif(mode == "round"):
        input_trunc = torch.round(input * sf)/sf  # Round to Nearest
    elif(mode == "stochastic"):
        rdn = torch.rand_like(input) 
        input_trunc = torch.floor(input * sf + rdn)/sf

    
    # Saturate Overflow Vals
    clipped_input = torch.clamp(input_trunc, min_val, max_val)

    return clipped_input

# Floating point to Floating point
def fp_quant(input, n_exp, n_man, mode="trunc", device = "cpu"):
    bias = (2 ** (n_exp - 1)) -1
    exp_min = (2** ((-bias) + 1))
    #exp_max = (2** (bias + 1))
    exp_max = torch.pow(torch.tensor(2.0, device = device) , (bias + 1))
    #man_max = 2 - (2**(-n_man))
    man_max = (1 + (2**n_man -1 )/2**n_man)
    man_min = (2 ** (-n_man))
    min_val = exp_min * man_min
    max_val = exp_max * man_max
    epsilon = 1e-16
    man_sf = 2 ** n_man  # Mantissa Scaling factor


    # Again, Check Overflow
    input_clipped = torch.clamp(input, min = -max_val, max = max_val)
    
    mask = (input_clipped>0) & (input_clipped < min_val)
    input_clipped [mask] = 0
    mask = (input_clipped<0) & (input_clipped > -min_val)
    input_clipped [mask] = 0

    zero_mask = (input_clipped == 0)

    input_abs = torch.abs(input_clipped)


    assert mode in ["trunc", "round", "stochastic"] , "Quantize Mode Must be 'trunc' or 'round' or 'stochastic'"
    


    # Extract exp value
    input_exp = torch.log2(input_abs).to(device)
    input_exp = torch.floor(input_exp).to(device)


    # For Denenormalize
    input_exp = torch.where(input_exp <torch.tensor((-bias)+1).float().to(device), torch.tensor((-bias)+1).float().to(device), input_exp).to(device) 
    
    
       
    
    # When input_exp < (-bias + 1), Denorm, and input_man < 1 (Denormalized number!)
    input_man = input_abs / (2 ** input_exp)  # If Denorm, input_man = 0.xxxx , else input_man == 1.xxxx

    # Same with Fixed point, Restrict the number with given bit precision (Mantissa Width)
    # Mantissa Quantization
    if(mode == "trunc"):
        man_trunc = torch.floor(input_man * man_sf)/man_sf # Truncate Fractional Bit
    elif(mode == "round"):
        man_trunc = torch.round(input_man * man_sf)/man_sf  # Round to Nearest
    elif(mode == "stochastic"):
        #decimal_part = input_man * man_sf - torch.floor(input_man * man_sf)
        #rdn = torch.where(torch.rand_like(decimal_part) < decimal_part , 0 , 1)
        rdn = torch.rand_like(input_man)
        man_trunc = torch.floor(input_man * man_sf + rdn)/man_sf
    
 

    # Value restore ( mantissa * 2^(exp)) if Denorm case, mantissa = 0.xxxx, exp = (-bias + 1)
    input_quantized = man_trunc * (2 ** input_exp)

    

    # Attach Sign Bit
    signed_input = input_quantized * torch.sign(input)

    return signed_input

def quantization(input, type="fxp", n_exp = 5 , n_man = 10, mode = "trunc", device = "cpu" ):
    if (type == "fxp") :
        input = input.to(device)
        quant_output = fxp_quant(input, n_man, n_exp, mode , device)
    elif (type == "fp"):
        input = input.to(device)
        quant_output = fp_quant(input, n_exp, n_man, mode, device)

    return quant_output


class Quant(Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, input, type="fxp", n_exp =5 , n_man = 10, mode = "trunc", device = "cpu"):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.type = type
        ctx.n_exp = n_exp
        ctx.n_man = n_man
        ctx.mode = mode
        ctx.device = device
        input = input.to(device)
        output = quantization(input, type, n_exp, n_man, mode, device)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        
        
        
        return grad_output ,None, None, None, None, None


class PrecisionLayer(torch.nn.Module):
    def __init__(self, type = "fxp", n_exp = 5, n_man = 10, mode = "trunc", device = "cuda" ):
        super(PrecisionLayer,self).__init__()
        self.type = type
        self.n_exp = n_exp
        self.n_man = n_man
        self.mode = mode
        self.device = device

    
    def forward(self, input):
        input = input.to(self.device)
        return Quant.apply(input, self.type, self.n_exp, self.n_man, self.mode, self.device)
    



    def extra_repr(self):
        # (optional) Set the extra information about this module.
        # You can test it by printing an object of this class
        return f'type={self.type}, n_exp={self.n_exp}, n_man={self.n_man}, mode={self.mode}, device={self.device}'

