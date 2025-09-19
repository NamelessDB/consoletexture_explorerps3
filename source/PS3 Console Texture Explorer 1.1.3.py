import os
import io
import struct
import threading
import time
from functools import partial
import tkinter as tk
from tkinter import filedialog, font, ttk, Menu
from tkinter import StringVar, IntVar, BooleanVar
from PIL import Image, ImageTk
import numpy as np


CG_TEX_SZ = 0x00
CG_TEX_LN = 0x20
CG_TEX_NR = 0x00
CG_TEX_UN = 0x40


CG_TEX_B8 = 0x81
CG_TEX_A1R5G5B5 = 0x82
CG_TEX_A4R4G4B4 = 0x83
CG_TEX_R5G6B5 = 0x84
CG_TEX_A8R8G8B8 = 0x85
CG_TEX_COMPRESSED_DXT1 = 0x86
CG_TEX_COMPRESSED_DXT23 = 0x87
CG_TEX_COMPRESSED_DXT45 = 0x88
CG_TEX_G8B8 = 0x8B
CG_TEX_COMPRESSED_B8R8_G8R8 = 0x8D
CG_TEX_COMPRESSED_R8B8_R8G8 = 0x8E
CG_TEX_R6G5B5 = 0x8F
CG_TEX_DEPTH24_D8 = 0x90
CG_TEX_DEPTH24_D8_FLOAT = 0x91
CG_TEX_DEPTH16 = 0x92
CG_TEX_DEPTH16_FLOAT = 0x93
CG_TEX_X16 = 0x94
CG_TEX_Y16_X16 = 0x95
CG_TEX_R5G5B5A1 = 0x97
CG_TEX_COMPRESSED_HILO8 = 0x98
CG_TEX_COMPRESSED_HILO_S8 = 0x99
CG_TEX_ARGB32 = 0xA5
CG_TEX_W16_Z16_Y16_X16_FLOAT = 0x9A
CG_TEX_W32_Z32_Y32_X32_FLOAT = 0x9B
CG_TEX_X32_FLOAT = 0x9C
CG_TEX_D1R5G5B5 = 0x9D
CG_TEX_D8R8G8B8 = 0x9E
CG_TEX_Y16_X16_FLOAT = 0x9F

FORMAT_MAPPING = {
    'A8R8G8B8': CG_TEX_A8R8G8B8,
    'DXT1': CG_TEX_COMPRESSED_DXT1,
    'DXT2': CG_TEX_COMPRESSED_DXT23,
    'DXT3': CG_TEX_COMPRESSED_DXT23,
    'DXT4': CG_TEX_COMPRESSED_DXT45,
    'DXT5': CG_TEX_COMPRESSED_DXT45,
    'R5G6B5': CG_TEX_R5G6B5,
    'R6G5B5': CG_TEX_R6G5B5,
    'R5G5B5A1': CG_TEX_R5G5B5A1,
    'A1R5G5B5': CG_TEX_A1R5G5B5,
    'D8R8G8B8': CG_TEX_D8R8G8B8,
    'B8': CG_TEX_B8,
    'A4R4G4B4': CG_TEX_A4R4G4B4,
    'G8B8': CG_TEX_G8B8,
    'B8R8_G8R8': CG_TEX_COMPRESSED_B8R8_G8R8,
    'R8B8_R8G8': CG_TEX_COMPRESSED_R8B8_R8G8,
    'DEPTH24_D8': CG_TEX_DEPTH24_D8,
    'DEPTH24_D8_FLOAT': CG_TEX_DEPTH24_D8_FLOAT,
    'DEPTH16': CG_TEX_DEPTH16,
    'DEPTH16_FLOAT': CG_TEX_DEPTH16_FLOAT,
    'X16': CG_TEX_X16,
    'Y16_X16': CG_TEX_Y16_X16,
    'HILO8': CG_TEX_COMPRESSED_HILO8,
    'HILO_S8': CG_TEX_COMPRESSED_HILO_S8,
    'ARGB32': CG_TEX_ARGB32,
    'W16_Z16_Y16_X16_FLOAT': CG_TEX_W16_Z16_Y16_X16_FLOAT,
    'W32_Z32_Y32_X32_FLOAT': CG_TEX_W32_Z32_Y32_X32_FLOAT,
    'X32_FLOAT': CG_TEX_X32_FLOAT,
    'D1R5G5B5': CG_TEX_D1R5G5B5,
    'Y16_X16_FLOAT': CG_TEX_Y16_X16_FLOAT,
}


# ----------------- Decoders -----------------

def srgb_gamma_correction(img):
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array[:, :, :3] = np.where(
        img_array[:, :, :3] <= 0.04045,
        img_array[:, :, :3] / 12.92,
        ((img_array[:, :, :3] + 0.055) / 1.055) ** 2.4
    )
    img_array = (img_array * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_array, 'RGBA')

def decode_b8(data, use_alpha=True):
    return [(v, v, v, 255) for v in data]

def decode_a4r4g4b4(data, use_alpha=True):
    pixels = []
    for i in range(0, len(data), 2):
        val = struct.unpack(">H", data[i:i+2])[0]
        a = ((val >> 12) & 0xF) * 17 if use_alpha else 255
        r = ((val >> 8) & 0xF) * 17
        g = ((val >> 4) & 0xF) * 17
        b = (val & 0xF) * 17
        pixels.append((r, g, b, a))
    return pixels

def decode_g8b8(data, use_alpha=True):
    pixels = []
    for i in range(0, len(data), 2):
        g, b = data[i:i+2]
        pixels.append((0, g, b, 255))
    return pixels

def decode_hilo8(data, use_alpha=True):
    pixels = []
    for i in range(0, len(data), 2):
        h, l = data[i:i+2]
        
        pixels.append((l, h, 0, 255))
    return pixels

def decode_hilo_s8(data, use_alpha=True):
    pixels = []
    for i in range(0, len(data), 2):
        h, l = struct.unpack(">bb", data[i:i+2])
        r = max(0, min(255, h + 128))
        g = max(0, min(255, l + 128))
        pixels.append((r, g, 0, 255))
    return pixels

def decode_d1r5g5b5(data, use_alpha=True):
    pixels = []
    for i in range(0, len(data), 2):
        val = struct.unpack(">H", data[i:i+2])[0]
        
        r = ((val >> 10) & 0x1F) << 3
        g = ((val >> 5) & 0x1F) << 3
        b = (val & 0x1F) << 3
        pixels.append((r, g, b, 255))
    return pixels


def decode_rgb565(data, use_alpha=True):
    pixels = []
    for i in range(0, len(data), 2):
        val = struct.unpack(">H", data[i:i+2])[0]
        r = ((val >> 11) & 0x1F) << 3
        g = ((val >> 5) & 0x3F) << 2
        b = (val & 0x1F) << 3
        a = 255 if use_alpha else 255
        pixels.append((r, g, b, a))
    return pixels

def decode_a8r8g8b8(data, use_alpha=True):
    pixels=[]
    for i in range(0,len(data),4):
        a,r,g,b = data[i:i+4]
        if not use_alpha:
            a = 255
        pixels.append((r,g,b,a))
    return pixels

def decode_d8r8g8b8(data, use_alpha=True):
    pixels=[]
    for i in range(0,len(data),4):
        d,r,g,b = data[i:i+4]
        a = 255
        pixels.append((r,g,b,a))
    return pixels

def decode_r5g5b5a1(data, use_alpha=True):
    pixels=[]
    for i in range(0,len(data),2):
        val = struct.unpack(">H", data[i:i+2])[0]
        r = ((val>>11)&0x1F)<<3
        g = ((val>>6)&0x1F)<<3
        b = ((val>>1)&0x1F)<<3
        a = (val & 1) * 255 if use_alpha else 255
        pixels.append((r,g,b,a))
    return pixels

def decode_r6g5b5(data, use_alpha=True):
    pixels=[]
    for i in range(0,len(data),2):
        val = struct.unpack(">H", data[i:i+2])[0]
        r = ((val >> 10) & 0x3F) << 2
        g = ((val >> 5) & 0x1F) << 3
        b = (val & 0x1F) << 3
        pixels.append((r,g,b,255))
    return pixels

def decode_r5g6b5_helper(data,use_alpha=True):
    pixels=[]
    for i in range(0,len(data),2):
        val = struct.unpack('>H', data[i:i+2])[0]
        r = ((val>>11)&0x1F)<<3
        g = ((val>>5)&0x3F)<<2
        b = (val&0x1F)<<3
        pixels.append((r,g,b,255))
    return pixels

def decode_a1r5g5b5(data, use_alpha=True):
    pixels=[]
    for i in range(0,len(data),2):
        val = struct.unpack(">H", data[i:i+2])[0]
        a = ((val>>15)&1)*255 if use_alpha else 255
        r = ((val>>10)&0x1F)<<3
        g = ((val>>5)&0x1F)<<3
        b = (val&0x1F)<<3
        pixels.append((r,g,b,a))
    return pixels

def decode_a8(data, use_alpha=True):
    return [(0,0,0,b if use_alpha else 255) for b in data]

def decode_l8(data, use_alpha=True):
    return [(v,v,v,255) for v in data]

def decode_la8(data, use_alpha=True):
    pixels = []
    for i in range(0,len(data),2):
        l,a = data[i:i+2]
        pixels.append((l,l,l,a if use_alpha else 255))
    return pixels

def decode_rgb10a2(data, use_alpha=True, endian='>'):
    pixels = []
    for i in range(0,len(data),4):
        val = struct.unpack(endian+"I",data[i:i+4])[0]
        r = (val & 0x3FF) >> 2
        g = ((val >> 10) & 0x3FF) >> 2
        b = ((val >> 20) & 0x3FF) >> 2
        a = ((val >> 30) & 0x3) * 85 if use_alpha else 255
        pixels.append((r,g,b,a))
    return pixels

def decode_r8(data, use_alpha=True):
    return [(v,0,0,255) for v in data]

def decode_rg8(data, use_alpha=True):
    pixels = []
    for i in range(0,len(data),2):
        r,g = data[i:i+2]
        pixels.append((r,g,0,255))
    return pixels

def decode_r16(data, use_alpha=True, endian='>'):
    pixels=[]
    for i in range(0,len(data),2):
        r = struct.unpack(endian+"H", data[i:i+2])[0]>>8
        pixels.append((r,0,0,255))
    return pixels

def decode_rg16(data, use_alpha=True, endian='>'):
    pixels=[]
    for i in range(0,len(data),4):
        r = struct.unpack(endian+"H", data[i:i+2])[0]>>8
        g = struct.unpack(endian+"H", data[i+2:i+4])[0]>>8
        pixels.append((r,g,0,255))
    return pixels

def decode_rgba16(data, use_alpha=True, endian='>'):
    pixels=[]
    for i in range(0,len(data),8):
        r = struct.unpack(endian+"H", data[i:i+2])[0]>>8
        g = struct.unpack(endian+"H", data[i+2:i+4])[0]>>8
        b = struct.unpack(endian+"H", data[i+4:i+6])[0]>>8
        a = struct.unpack(endian+"H", data[i+6:i+8])[0]>>8 if use_alpha else 255
        pixels.append((r,g,b,a))
    return pixels

def decode_depth(data, bpp, use_float=False, endian='>'):
    pixels = []
    if bpp == 2:
        for i in range(0, len(data), 2):
            val = struct.unpack(endian + 'H', data[i:i+2])[0]
            
            lum = int(val / 257)
            pixels.append((lum, lum, lum, 255))
    elif bpp == 4:
        for i in range(0, len(data), 4):
            if use_float:
                val = struct.unpack(endian + 'f', data[i:i+4])[0]
                lum = max(0, min(255, int(val * 255))) 
            else:
                val = struct.unpack(endian + 'I', data[i:i+4])[0]
                lum = int(val / 16777216) 
            pixels.append((lum, lum, lum, 255))
    elif bpp == 8: 
        for i in range(0, len(data), 8):
            x = struct.unpack(endian + 'f', data[i+6:i+8])[0] 
            lum = max(0, min(255, int(x * 255)))
            pixels.append((lum, lum, lum, 255))
    elif bpp == 16: 
        for i in range(0, len(data), 16):
            x = struct.unpack(endian + 'f', data[i+12:i+16])[0] 
            lum = max(0, min(255, int(x * 255)))
            pixels.append((lum, lum, lum, 255))
    return pixels


def decode_linear(data, fmt, use_alpha=True, endian='>'):
    if fmt in ('RGBA8','ARGB32','BGRA8','ABGR8','B8G8R8A8'):
        pixels = []
        for i in range(0, len(data), 4):
            b0,b1,b2,b3 = data[i:i+4]
            if not use_alpha: b3=255
            if fmt=='RGBA8': pixels.append((b0,b1,b2,b3))
            elif fmt=='ARGB32': pixels.append((b1,b2,b3,b0))
            elif fmt=='BGRA8' or fmt=='B8G8R8A8': pixels.append((b2,b1,b0,b3))
            elif fmt=='A8R8G8B8': pixels.append((b1,b2,b3,b0))
            elif fmt=='ABGR8': pixels.append((b3,b2,b1,b0))
        return pixels
    elif fmt == 'A8R8G8B8':
        return decode_a8r8g8b8(data, use_alpha)
    elif fmt=='RGB24':
        return [(data[i],data[i+1],data[i+2],255) for i in range(0,len(data),3)]
    elif fmt=='BGR24':
        return [(data[i+2],data[i+1],data[i],255) for i in range(0,len(data),3)]
    elif fmt=='RGB565':
        return decode_rgb565(data,use_alpha)
    elif fmt=='R5G6B5':
        return decode_r5g6b5_helper(data,use_alpha)
    elif fmt=='R6G5B5':
        return decode_r6g5b5(data,use_alpha)
    elif fmt=='R5G5B5A1':
        return decode_r5g5b5a1(data,use_alpha)
    elif fmt=='A1R5G5B5':
        return decode_a1r5g5b5(data,use_alpha)
    elif fmt=='A4R4G4B4':
        return decode_a4r4g4b4(data,use_alpha)
    elif fmt=='A8':
        return decode_a8(data,use_alpha)
    elif fmt=='B8':
        return decode_b8(data,use_alpha)
    elif fmt=='L8':
        return decode_l8(data,use_alpha)
    elif fmt=='LA8':
        return decode_la8(data,use_alpha)
    elif fmt=='RGB10A2':
        return decode_rgb10a2(data,use_alpha,endian)
    elif fmt=='R8':
        return decode_r8(data,use_alpha)
    elif fmt=='RG8':
        return decode_rg8(data,use_alpha)
    elif fmt=='G8B8':
        return decode_g8b8(data,use_alpha)
    elif fmt=='R16':
        return decode_r16(data,use_alpha,endian)
    elif fmt=='RG16':
        return decode_rg16(data,use_alpha,endian)
    elif fmt=='RGBA16':
        return decode_rgba16(data,use_alpha,endian)
    elif fmt=='D1R5G5B5':
        return decode_d1r5g5b5(data,use_alpha)
    elif fmt=='D8R8G8B8':
        return decode_d8r8g8b8(data,use_alpha)
    elif fmt in ('DEPTH16','X16'): 
        return decode_depth(data,2,False,endian)
    elif fmt in ('Y16_X16','RG16'): 
        return decode_depth(data,4,False,endian)
    elif fmt in ('DEPTH24_D8','DEPTH24_D8_FLOAT', 'X32_FLOAT','Y16_X16_FLOAT'): 
        return decode_depth(data,4,'FLOAT' in fmt,endian)
    elif fmt in ('W16_Z16_Y16_X16_FLOAT'):
        return decode_depth(data,8,True,endian)
    elif fmt in ('W32_Z32_Y32_X32_FLOAT'):
        return decode_depth(data,16,True,endian)
    elif fmt in ('HILO8', 'HILO_S8'):
        if fmt == 'HILO8':
            return decode_hilo8(data, use_alpha)
        else:
            return decode_hilo_s8(data, use_alpha)
    else:
        raise ValueError("Unsupported format:" + str(fmt))


def decode_dxt_block(alpha_bytes, color_bytes, fmt, use_alpha=True):
    alpha = [255]*16
    if 'DXT2' in fmt or 'DXT3' in fmt and use_alpha:
        for i in range(0, 8, 2):
            val = struct.unpack("<H", alpha_bytes[i:i+2])[0]  # Little-endian
            for j in range(4):
                alpha[i*2//2 + j] = ((val >> (j*4)) & 0xF) * 17
    elif 'DXT4' in fmt or 'DXT5' in fmt and use_alpha:
        a0,a1 = alpha_bytes[0],alpha_bytes[1]
        bits = int.from_bytes(alpha_bytes[2:8],'little')  # Always little-endian
        alpha_vals = [a0,a1]
        if a0>a1:
            for i in range(1,6):
                alpha_vals.append(( (6-i)*a0 + i*a1)//7)
        else:
            for i in range(1,4):
                alpha_vals.append(( (4-i)*a0 + i*a1)//5)
            alpha_vals += [0,255]
        for i in range(16):
            idx = (bits >> (3*i)) & 0x7
            alpha[i] = alpha_vals[idx]

    c0,c1 = struct.unpack("<HH", color_bytes[:4])  # Little-endian
    r0 = ((c0>>11)&0x1F)<<3; g0 = ((c0>>5)&0x3F)<<2; b0 = (c0&0x1F)<<3
    r1 = ((c1>>11)&0x1F)<<3; g1 = ((c1>>5)&0x3F)<<2; b1 = (c1&0x1F)<<3
    colors = [(r0,g0,b0),(r1,g1,b1),((2*r0+r1)//3,(2*g0+g1)//3,(2*b0+b1)//3),((r0+2*r1)//3,(g0+2*g1)//3,(b0+2*b1)//3)]
    
    indices = struct.unpack("<I", color_bytes[4:8])[0]  # Little-endian
    block = [(colors[(indices>>(2*i))&0x3][0], colors[(indices>>(2*i))&0x3][1], colors[(indices>>(2*i))&0x3][2], alpha[i]) for i in range(16)]  # RGB order
    return block

def decode_dxt(data,width,height,fmt,use_alpha=True):
    img = np.zeros((height,width,4),dtype=np.uint8)
    blocks_w = (width + 3)//4
    blocks_h = (height + 3)//4
    pos = 0
    for by in range(blocks_h):
        for bx in range(blocks_w):
            if fmt=='DXT1':
                alpha_bytes = bytes(8)
                color_bytes = data[pos:pos+8]
                pos += 8
            else:
                alpha_bytes = data[pos:pos+8]
                color_bytes = data[pos+8:pos+16]
                pos += 16
            block = decode_dxt_block(alpha_bytes,color_bytes,fmt,use_alpha)
            for y in range(4):
                for x in range(4):
                    if by*4+y<height and bx*4+x<width:
                        img[by*4+y,bx*4+x] = block[y*4+x]
    return Image.fromarray(img,'RGBA')



def morton2d(x, y, bits):
    m = 0
    for i in range(bits):
        m |= ((x >> i) & 1) << (2 * i)
        m |= ((y >> i) & 1) << (2 * i + 1)
    return m


def bit_reverse(v, bits):
    r = 0
    for i in range(bits):
        r = (r << 1) | ((v >> i) & 1)
    return r


def reorder_dxt_blocks_variant(data, width, height, fmt, mode='SZ'):
    block_w = (width + 3) // 4
    block_h = (height + 3) // 4
    total = block_w * block_h
    block_size = 8 if fmt == 'DXT1' else 16

    
    blocks = [
        data[i*block_size:(i+1)*block_size]
        for i in range(total)
        if (i+1)*block_size <= len(data)
    ]

    
    new_blocks = [None] * total
    bits = max(block_w, block_h).bit_length()

    for y in range(block_h):
        for x in range(block_w):
            idx = y * block_w + x
            if mode == 'SZ':    
                dst = morton2d(x, y, bits)
            elif mode == 'LN':    
                dst = idx
            elif mode == 'NR':    
                dst = bit_reverse(morton2d(x, y, bits), bits*2)
            elif mode == 'UN':    
                dst = (x << bits) | y
            elif mode == 'Desactivado': 
                dst = idx
            else:
                dst = idx
            if dst < total and idx < len(blocks):
                new_blocks[dst] = blocks[idx]

    return b''.join([b for b in new_blocks if b is not None])



def deswizzle_pixels_morton(img, mode='SZ'):
    w,h = img.size
    maxdim = max(w-1,h-1)
    bits = maxdim.bit_length() or 1
    src = img.load()
    out = Image.new('RGBA',(w,h))
    out_px = out.load()
    total = w*h
    for y in range(h):
        for x in range(w):
            if mode=='LN' or mode=='NORMAL' or mode=='Desactivado':
                sx,sy = x,y
            elif mode=='SZ':
                m = morton2d(x,y,bits)
                sx = m % w; sy = m // w
            elif mode=='NR':
                rx = bit_reverse(x,bits); ry = bit_reverse(y,bits)
                m = morton2d(rx,ry,bits)
                sx = m % w; sy = m // w
            elif mode=='UN':
                m = 0
                for i in range(bits):
                    xb = (x>>i)&1; yb=(y>>i)&1
                    if i%2==0:
                        m |= xb << (2*i)
                        m |= yb << (2*i+1)
                    else:
                        m |= yb << (2*i)
                        m |= xb << (2*i+1)
                sx = m % w; sy = m // w
            else:
                sx,sy = x,y
            if sx<w and sy<h:
                out_px[x,y] = src[sx,sy]
    return out



class ThemeManager:
    themes = {
        'Dark Blue': {
            'bg': '#161616', 'panel': '#1f1f1f', 'fg': '#e6e6e6', 'accent': '#3a7bd5', 'button_bg': '#3a7bd5'
        },
        'Solarized Dark': {
            'bg': '#002b36', 'panel': '#073642', 'fg': '#839496', 'accent': '#268bd2', 'button_bg': '#268bd2'
        },
        'Monokai': {
            'bg': '#272822', 'panel': '#3e3d32', 'fg': '#f8f8f2', 'accent': '#ae81ff', 'button_bg': '#ae81ff'
        },
        'Matrix': {
            'bg': '#000000', 'panel': '#0c0c0c', 'fg': '#00ff41', 'accent': '#00ff41', 'button_bg': '#005500'
        },
        'Gruvbox': {
            'bg': '#282828', 'panel': '#3c3836', 'fg': '#ebdbb2', 'accent': '#458588', 'button_bg': '#458588'
        },
        'Aura': {
            'bg': '#212436', 'panel': '#242a42', 'fg': '#ede9e6', 'accent': '#64ffda', 'button_bg': '#64ffda'
        },
        'Dracula': {
            'bg': '#282a36', 'panel': '#44475a', 'fg': '#f8f8f2', 'accent': '#bd93f9', 'button_bg': '#bd93f9'
        },
        'Nord': {
            'bg': '#2e3440', 'panel': '#3b4252', 'fg': '#d8dee9', 'accent': '#88c0d0', 'button_bg': '#88c0d0'
        },
        'Tomorrow Night Blue': {
            'bg': '#002451', 'panel': '#003366', 'fg': '#ffffff', 'accent': '#5c80e6', 'button_bg': '#5c80e6'
        },
        'Oceanic': {
            'bg': '#1b2b34', 'panel': '#2c3e50', 'fg': '#a7adba', 'accent': '#4ecdc4', 'button_bg': '#4ecdc4'
        },
        'Ubuntu': {
            'bg': '#310034', 'panel': '#431746', 'fg': '#f3f1f5', 'accent': '#e95420', 'button_bg': '#e95420'
        },
        'Windows 11': {
            'bg': '#1e1e1e', 'panel': '#2d2d2d', 'fg': '#ffffff', 'accent': '#0078d4', 'button_bg': '#0078d4'
        },
        'Atom': {
            'bg': '#282c34', 'panel': '#31333b', 'fg': '#abb2bf', 'accent': '#61afef', 'button_bg': '#61afef'
        },
        'One Dark': {
            'bg': '#282c34', 'panel': '#3d4048', 'fg': '#abb2bf', 'accent': '#c678dd', 'button_bg': '#c678dd'
        },
        'Material': {
            'bg': '#263238', 'panel': '#37474f', 'fg': '#eceff1', 'accent': '#42a5f5', 'button_bg': '#42a5f5'
        },
        'Ocean': {
            'bg': '#1b2b34', 'panel': '#2c3e50', 'fg': '#a7adba', 'accent': '#5fb2c8', 'button_bg': '#5fb2c8'
        },
        'Forest': {
            'bg': '#1c2120', 'panel': '#272d2c', 'fg': '#c0dfd7', 'accent': '#5e8d83', 'button_bg': '#5e8d83'
        },
        'Rose Pine': {
            'bg': '#191724', 'panel': '#26233a', 'fg': '#e0def4', 'accent': '#c4a7e7', 'button_bg': '#c4a7e7'
        },
    }
    
    def __init__(self, root):
        self.root = root
        self.current_theme = StringVar(value='Dark Blue')
        self.apply_theme()
    
    def apply_theme(self):
        theme_name = self.current_theme.get()
        colors = self.themes.get(theme_name, self.themes['Dark Blue'])
        self.root.configure(bg=colors['bg'])
        
        
        app_instance = self.root.app
        app_instance.bg = colors['bg']
        app_instance.panel = colors['panel']
        app_instance.fg = colors['fg']
        app_instance.accent = colors['accent']
        app_instance.button_bg = colors['button_bg']
        
        
        for widget in self.root.winfo_children():
            self._update_widget_colors(widget, colors)
        
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TCombobox',
                        fieldbackground=colors['panel'],
                        background=colors['panel'],
                        foreground=colors['fg'],
                        selectbackground=colors['accent'],
                        selectforeground=colors['fg'],
                        bordercolor=colors['accent'])
        style.map('TCombobox',
                  fieldbackground=[('readonly', colors['panel'])],
                  background=[('readonly', colors['panel'])],
                  foreground=[('readonly', colors['fg'])])

    def _update_widget_colors(self, widget, colors):
        try:
            widget.configure(bg=colors['bg'], fg=colors['fg'])
        except tk.TclError:
            pass
        try:
            if isinstance(widget, (tk.Entry, tk.Frame, tk.LabelFrame)):
                widget.configure(bg=colors['panel'])
            if isinstance(widget, tk.Entry):
                widget.configure(fg=colors['fg'], insertbackground=colors['fg'])
            if isinstance(widget, tk.Button):
                widget.configure(bg=colors['button_bg'], fg='white')
            if isinstance(widget, tk.Label) and widget.cget('text').startswith("Creado por"):
                 widget.configure(bg=colors['bg'])
        except tk.TclError:
            pass
        
        for child in widget.winfo_children():
            self._update_widget_colors(child, colors)




class App:
    def __init__(self, root):
        self.root = root
        root.title('PS3 Texture Explorer - Dark')
        
        
        self.bg = '#161616'
        self.panel = '#1f1f1f'
        self.fg = '#e6e6e6'
        self.accent = '#3a7bd5'
        self.button_bg = '#3a7bd5'

       
        self.root.app = self 
        self.theme_manager = ThemeManager(root)

        
        self.filepath = StringVar()
        self.width = IntVar(value=128)
        self.height = IntVar(value=128)
        self.offset = IntVar(value=0)
        self.palette_offset = IntVar(value=0)
        self.palette_size = IntVar(value=256)
        self.palette_order = StringVar(value='RGBA')
        self.format = StringVar(value='A8R8G8B8')
        self.swizzle = StringVar(value='Desactivado')
        self.normalization = StringVar(value='NR')
        self.deswizzle = StringVar(value='Desactivado')
        self.use_alpha = BooleanVar(value=True)
        self.export_fmt = StringVar(value='png')
        self.endianness = StringVar(value='Big Endian')
        self.srgb = BooleanVar(value=False)

        self._last_change = 0
        self._render_job = None
        
        self.credit_label_color_state = False

        self._last_render_img = None
        self._last_render_meta = {}

        
        self._build_ui()
        self.theme_manager.apply_theme()
        self.animate_credit_label()

        for var in (self.filepath,self.width,self.height,self.offset,self.format,self.swizzle,self.normalization,self.deswizzle,self.use_alpha,self.palette_offset,self.palette_size,self.palette_order,self.endianness, self.srgb):
            try:
                var.trace_add('write', lambda *a: self.on_change())
            except Exception:
                pass


    def _label(self, parent, text, **kw):
        lbl = tk.Label(parent, text=text, bg=self.bg, fg=self.fg, anchor='w')
        if 'row' in kw:
            lbl.grid(row=kw['row'], column=kw.get('column',0), sticky='w', padx=6, pady=3)
        return lbl

    def _entry(self, parent, var, width=10, **kw):
        e = tk.Entry(parent, textvariable=var, width=width, bg=self.panel, fg=self.fg, insertbackground=self.fg, relief='flat')
        e.grid(row=kw.get('row',0), column=kw.get('column',0), padx=6, pady=3)
        return e

    def _combobox(self, parent, var, values, row, column):
        cb = ttk.Combobox(parent, textvariable=var, values=values, state='readonly', width=12)
        cb.grid(row=row, column=column, padx=6, pady=3)
        return cb
    
    def _checkbox(self, parent, text, var, row, column):
        cb = tk.Checkbutton(parent, text=text, variable=var, bg=self.bg, fg=self.fg, selectcolor=self.panel, activebackground=self.bg, activeforeground=self.fg)
        cb.grid(row=row, column=column, columnspan=2, sticky='w', padx=6, pady=3)
        return cb


    def _build_ui(self):
        
        menubar = Menu(self.root, bg=self.bg, fg=self.fg)
        self.root.config(menu=menubar)
        
        theme_menu = Menu(menubar, tearoff=0, bg=self.panel, fg=self.fg)
        menubar.add_cascade(label="Themes", menu=theme_menu)
        for theme_name in self.theme_manager.themes.keys():
            theme_menu.add_command(label=theme_name, command=lambda name=theme_name: self.set_theme(name))


        top = tk.Frame(self.root, bg=self.bg)
        top.pack(side='top', fill='x', padx=8, pady=8)

        
        self._label(top, 'PS3 Binary:', row=0, column=0)
        tk.Entry(top, textvariable=self.filepath, bg=self.panel, fg=self.fg, insertbackground=self.fg, width=50, relief='flat').grid(row=0,column=1,columnspan=3,padx=6)
        tk.Button(top, text='📂 Browse', command=self.browse, bg=self.button_bg, fg='white', relief='flat', font=('Segoe UI', 10, 'bold')).grid(row=0,column=4,padx=6)

        
        self.filepath_label = tk.Label(top, text='File Opened: None', bg=self.bg, fg=self.fg, anchor='w')
        self.filepath_label.grid(row=1,column=0,columnspan=5, sticky='w', padx=6, pady=2)

        
        self._label(top, 'Width:', row=2, column=0)
        self._entry(top, self.width, row=2, column=1)
        self._label(top, 'Height:', row=2, column=2)
        self._entry(top, self.height, row=2, column=3)
        self._label(top, 'Offset:', row=3, column=0)
        self._entry(top, self.offset, row=3, column=1)

        
        self._label(top, 'Palette offset (opt):', row=3, column=2)
        self._entry(top, self.palette_offset, row=3, column=3)
        self._label(top, 'Palette size:', row=4, column=2)
        self._entry(top, self.palette_size, row=4, column=3)
        self._label(top, 'Palette order:', row=4, column=0)
        self._combobox(top, self.palette_order, ['RGBA','BGRA','ARGB','ABGR','GRAB','BGA','RGB','BGR'], row=4, column=1)

        
        formats = sorted(list(FORMAT_MAPPING.keys()))
        self._label(top, 'Format:', row=5, column=0)
        self._combobox(top, self.format, formats, row=5, column=1)

       
        sw_modes = ['Desactivado','NORMAL','SZ','LN','NR','UN']
        self._label(top, 'Swizzle (blocks):', row=5, column=2)
        self._combobox(top, self.swizzle, sw_modes, row=5, column=3)
        
        self._label(top, 'Normalization flag:', row=6, column=0)
        self._combobox(top, self.normalization, ['NR','UN'], row=6, column=1)
        
        self._label(top, 'Deswizzle (pixels):', row=6, column=2)
        self._combobox(top, self.deswizzle, sw_modes, row=6, column=3)
        
        self._label(top, 'Endianness:', row=7, column=0)
        self._combobox(top, self.endianness, ['Big Endian', 'Little Endian'], row=7, column=1)
        
        
        self.flags_label = tk.Label(top, text='Flags: 0x00', bg=self.bg, fg=self.fg)
        self.flags_label.grid(row=7,column=2,columnspan=2, sticky='w', padx=6)
        self.update_flags_label()

        
        self._checkbox(top, 'Use alpha:', self.use_alpha, row=8, column=0)
        self._checkbox(top, 'Apply sRGB:', self.srgb, row=8, column=2)

        self._label(top, 'Export as:', row=9, column=0)
        self._combobox(top, self.export_fmt, ['png','dds'], row=9, column=1)
        
        
        tk.Button(top, text='Export Texture', command=self.export_current,
                  bg=self.button_bg, fg='white', relief='flat', font=('Segoe UI', 10, 'bold')).grid(row=10, column=0, columnspan=5, pady=6)
                  
        
        self.credit_label = tk.Label(self.root, text="Made By Huziad", bg=self.bg, fg="#e6e6e6", font=('Segoe UI', 12, 'bold'))
        self.credit_label.pack(side='bottom', pady=5)


        
        preview_frame = tk.Frame(self.root, bg=self.panel, bd=1, relief='sunken')
        preview_frame.pack(side='top', fill='both', expand=True, padx=8, pady=8)
        self.preview_label = tk.Label(preview_frame, bg=self.panel)
        self.preview_label.pack(fill='both', expand=True)

        
        bot = tk.Frame(self.root, bg=self.bg)
        bot.pack(side='bottom', fill='x', padx=8, pady=6)
        tk.Button(bot, text='Reload', command=self.trigger_render, bg=self.button_bg, fg='white', relief='flat', font=('Segoe UI', 10)).pack(side='right')

    def set_theme(self, name):
        self.theme_manager.current_theme.set(name)
        self.theme_manager.apply_theme()
        
    def animate_credit_label(self):
        current_colors = self.theme_manager.themes.get(self.theme_manager.current_theme.get())
        if self.credit_label_color_state:
            self.credit_label.config(fg=current_colors['fg'])
        else:
            self.credit_label.config(fg=current_colors['accent'])
        self.credit_label_color_state = not self.credit_label_color_state
        self.root.after(750, self.animate_credit_label)


    def update_flags_label(self):
        fmt_name = self.format.get().upper()
        flags = FORMAT_MAPPING.get(fmt_name, 0)
        
        sw = self.swizzle.get()
        norm = self.normalization.get()
        sw_flag = CG_TEX_SZ if sw in ('LN', 'Desactivado') else CG_TEX_SZ
        norm_flag = CG_TEX_UN if norm=='UN' else CG_TEX_NR
        
        flags |= sw_flag | norm_flag
        self.flags_label.config(text=f'Flags: 0x{flags:02X} (fmt:{fmt_name}, sw:{sw}, norm:{norm})')

    def browse(self):
        p = filedialog.askopenfilename()
        if p:
            self.filepath.set(p)
            self.filepath_label.config(text=f'File Opened: {p}')

    def on_change(self):
        self._last_change = time.time()
        if self._render_job is None:
            self._render_job = self.root.after(200, self._debounced_render_check)

    def _debounced_render_check(self):
        if time.time() - self._last_change >= 0.18:
            threading.Thread(target=self.render_preview, daemon=True).start()
        self._render_job = None

    def trigger_render(self):
        threading.Thread(target=self.render_preview, daemon=True).start()

    def read_texture_bytes(self):
        path = self.filepath.get()
        if not path or not os.path.exists(path):
            return None
        w = int(self.width.get()); h = int(self.height.get())
        off = int(self.offset.get())
        fmt = self.format.get().upper()
        
        endian_str = self.endianness.get()
        endian_char = '<' if endian_str == 'Little Endian' else '>'
        
        with open(path,'rb') as f:
            f.seek(off)
            if 'DXT' in fmt or 'HILO' in fmt or 'R8B8_R8G8' in fmt or 'B8R8_G8R8' in fmt:
                blocks_w = (w + 3)//4; blocks_h = (h + 3)//4
                if 'DXT' in fmt:
                    size = blocks_w * blocks_h * (8 if 'DXT1' in fmt else 16)
                elif 'HILO' in fmt:
                    size = blocks_w * blocks_h * 16 
                else: 
                    size = blocks_w * blocks_h * 8
                
                data = f.read(size)
                mode = self.swizzle.get()
                if mode != 'NORMAL' and mode != 'Desactivado' and data:
                    data = reorder_dxt_blocks_variant(data,w,h,fmt,mode)
                return ('DXT', data, endian_char)
            else:
                bpp_map = {'RGBA8':4,'ARGB32':4,'A8R8G8B8':4,'BGRA8':4,'ABGR8':4,'B8G8R8A8':4,'RGBA16':8,'RGB10A2':4,
                           'RGB24':3,'BGR24':3,'RGB565':2,'R5G6B5':2,'R6G5B5':2,'R5G5B5A1':2,'A1R5G5B5':2,'A8':1,'L8':1,'LA8':2,
                           'R8':1,'RG8':2,'R16':2,'RG16':4,'D8R8G8B8':4, 'B8':1,'A4R4G4B4':2,'G8B8':2,
                           'DEPTH24_D8':4,'DEPTH24_D8_FLOAT':4,'DEPTH16':2,'DEPTH16_FLOAT':2,
                           'X16':2,'Y16_X16':4,'W16_Z16_Y16_X16_FLOAT':8,'W32_Z32_Y32_X32_FLOAT':16,'X32_FLOAT':4,
                           'D1R5G5B5':2,'Y16_X16_FLOAT':4}
                bpp = bpp_map.get(fmt,4)
                data = f.read(w*h*bpp)
                
                palette = None
                pal_off = int(self.palette_offset.get()) if self.palette_offset.get() else None
                if pal_off and bpp == 1:
                    cur = f.tell()
                    f.seek(pal_off)
                    pal_entries = int(self.palette_size.get())
                    pal_bytes = f.read(pal_entries * 4)
                    f.seek(cur)
                    palette = []
                    order = self.palette_order.get()
                    for i in range(0, len(pal_bytes), 4):
                        if i+4 <= len(pal_bytes):
                            a,b,c,d = pal_bytes[i:i+4]
                            if order == 'RGBA':
                                palette.append((a,b,c,d))
                            elif order == 'BGRA':
                                palette.append((b,c,a,d))
                            elif order == 'ARGB':
                                palette.append((b,c,d,a))
                            elif order == 'ABGR':
                                palette.append((d,c,b,a))
                    while len(palette) < pal_entries:
                        palette.append((0,0,0,0))
                return ('LINEAR', data, endian_char)

    
    def _render_image(self):
        res = self.read_texture_bytes()
        if res is None:
            return None
        
        kind, data, endian_char = res
        w = int(self.width.get()); h = int(self.height.get())
        fmt = self.format.get().upper()
        use_alpha = bool(self.use_alpha.get())
        
        img = None
        if kind == 'DXT':
             img = decode_dxt(data, w, h, fmt, use_alpha) 
        else:
            pixels = decode_linear(data, fmt, use_alpha, endian_char)
            img = Image.new('RGBA', (w, h))
            img.putdata(pixels)
            
           
            pix_mode = self.deswizzle.get()
            if pix_mode != 'NORMAL' and pix_mode != 'LN' and pix_mode != 'Desactivado':
                img = deswizzle_pixels_morton(img, pix_mode)
        
        
        if self.srgb.get():
            img = srgb_gamma_correction(img)
            
        return img

    def render_preview(self):
        try:
            
            img = self._render_image()
            if img is None:
                self._update_preview_image(None)
                return
            
            
            self._last_render_img = img.copy()
            self._last_render_meta = {
                'filepath': self.filepath.get(),
                'width': int(self.width.get()),
                'height': int(self.height.get()),
                'offset': int(self.offset.get()),
                'format': self.format.get().upper(),
                'swizzle': self.swizzle.get(),
                'normalization': self.normalization.get(),
                'deswizzle': self.deswizzle.get(),
                'use_alpha': bool(self.use_alpha.get()),
                'palette_offset': int(self.palette_offset.get()) if self.palette_offset.get() else 0,
                'palette_size': int(self.palette_size.get()) if self.palette_size.get() else 0,
                'palette_order': self.palette_order.get(),
                'export_fmt': self.export_fmt.get(),
                'endianness': self.endianness.get(),
                'srgb': self.srgb.get()
            }
            
            
            max_preview = 512
            sx = min(max_preview / img.width, 1.0)
            if sx < 1.0:
                img_thumb = img.resize((int(img.width * sx), int(img.height * sx)), Image.NEAREST)
            else:
                img_thumb = img
            
            self._update_preview_image(img_thumb)
            
        except Exception as e:
            print('Render error:', e)

    def _update_preview_image(self, pil_img):
        if pil_img is None:
            self.preview_label.config(image='', text='No preview', fg=self.fg, bg=self.panel)
            return
        self._last_img = ImageTk.PhotoImage(pil_img)
        def onmain():
            self.preview_label.config(image=self._last_img)
        self.root.after(0, onmain)

    def export_current(self):
        cur_meta = {
            'filepath': self.filepath.get(),
            'width': int(self.width.get()),
            'height': int(self.height.get()),
            'offset': int(self.offset.get()),
            'format': self.format.get().upper(),
            'swizzle': self.swizzle.get(),
            'normalization': self.normalization.get(),
            'deswizzle': self.deswizzle.get(),
            'use_alpha': bool(self.use_alpha.get()),
            'palette_offset': int(self.palette_offset.get()) if self.palette_offset.get() else 0,
            'palette_size': int(self.palette_size.get()) if self.palette_size.get() else 0,
            'palette_order': self.palette_order.get(),
            'export_fmt': self.export_fmt.get(),
            'endianness': self.endianness.get(),
            'srgb': self.srgb.get()
        }

       
        if self._last_render_img is not None and self._last_render_meta == cur_meta:
            img = self._last_render_img.copy()
        else:
            
            img = self._render_image()
            if img is None:
                print('No file/data')
                return
            
        export = self.export_fmt.get().lower()
        outname = filedialog.asksaveasfilename(defaultextension='.'+export, filetypes=[(export.upper(), '*.'+export)])
        if not outname: return
        
        if export=='png':
            img.save(outname,'PNG')
        else:
            try:
                img.save(outname,'DDS')
            except Exception:
                img.save(outname,'PNG')
        print('Exported to', outname)



def decode_r5g6b5_helper(data,use_alpha=True):
    pixels=[]
    for i in range(0,len(data),2):
        val = struct.unpack('>H', data[i:i+2])[0]
        r = ((val>>11)&0x1F)<<3
        g = ((val>>5)&0x3F)<<2
        b = (val&0x1F)<<3
        pixels.append((r,g,b,255))
    return pixels


decode_r5g6b5 = decode_r5g6b5_helper




def main():
    root = tk.Tk()
    root.geometry('1000x750')
    app = App(root)
    root.mainloop()

if __name__=='__main__':
    main()
