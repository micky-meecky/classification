import copy
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SingleDeconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.up(x)


class SingleConv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class Conv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv2DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv2DBlock(in_planes, out_planes),
            SingleConv2DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class SelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

        self.vis = False

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, in_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1()
        x = self.act(x)
        x = self.drop(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=786, d_ff=2048, dropout=0.1):
        super().__init__()
        # Torch linears have a `b` by default.
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, dropout):
        super().__init__()
        self.n_patches = int((cube_size[0] * cube_size[1]) / (patch_size * patch_size))
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv2d(in_channels=input_dim, out_channels=embed_dim,
                                          kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))  # 这是一个可学习参数
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)    # 2表示从第二个维度开始
        x = x.transpose(-1, -2)  # 交换最后两个维度
        embeddings = x + self.position_embeddings   # resnet
        embeddings = self.dropout(embeddings)   # dropout
        return embeddings


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, cube_size, patch_size):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_dim = int((cube_size[0] * cube_size[1]) / (patch_size * patch_size))
        self.mlp = PositionwiseFeedForward(embed_dim, 2048)
        self.attn = SelfAttention(num_heads, embed_dim, dropout)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        h = x

        x = self.mlp_norm(x)
        x = self.mlp(x)

        x = x + h
        return x, weights


class Transformer(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, num_heads, num_layers, dropout, extract_layers):
        super().__init__()
        self.embeddings = Embeddings(input_dim, embed_dim, cube_size, patch_size, dropout)
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.extract_layers = extract_layers
        for _ in range(num_layers):
            layer = TransformerBlock(embed_dim, num_heads, dropout, cube_size, patch_size)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        extract_layers = []
        hidden_states = self.embeddings(x)

        for depth, layer_block in enumerate(self.layer):
            hidden_states, _ = layer_block(hidden_states)
            if depth + 1 in self.extract_layers:
                extract_layers.append(hidden_states)

        return extract_layers


class UNETR(nn.Module):
    def __init__(self, img_shape=(224, 224), input_dim=1, output_dim=1, embed_dim=768, patch_size=16, num_heads=12,
                 dropout=0.1, batch_size=10):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]
        self.linear = nn.Linear(embed_dim * 1, self.output_dim, bias=True)  # bias=True 是指是否使用偏置

        self.patch_dim = [int(x / patch_size) for x in img_shape]

        # Transformer Encoder
        self.transformer = Transformer(input_dim, embed_dim, img_shape, patch_size, num_heads, self.num_layers, dropout,
                                       self.ext_layers)

        # U-Net Decoder
        self.decoder0 = nn.Sequential(Conv2DBlock(input_dim, 32, 3),Conv2DBlock(32, 64, 3))
        self.decoder3 = nn.Sequential(Deconv2DBlock(embed_dim, 512), Deconv2DBlock(512, 256), Deconv2DBlock(256, 128))
        self.decoder6 = nn.Sequential(Deconv2DBlock(embed_dim, 512), Deconv2DBlock(512, 256),)
        self.decoder9 = Deconv2DBlock(embed_dim, 512)
        self.decoder12_upsampler = SingleDeconv2DBlock(embed_dim, 512)
        self.decoder9_upsampler = nn.Sequential(Conv2DBlock(1024, 512), Conv2DBlock(512, 512), Conv2DBlock(512, 512),
                                                SingleDeconv2DBlock(512, 256))
        self.decoder6_upsampler = nn.Sequential(Conv2DBlock(512, 256), Conv2DBlock(256, 256),
                                                SingleDeconv2DBlock(256, 128))
        self.decoder3_upsampler = nn.Sequential(Conv2DBlock(256, 128), Conv2DBlock(128, 128),
                                                SingleDeconv2DBlock(128, 64))
        self.decoder0_header = nn.Sequential(Conv2DBlock(128, 64), Conv2DBlock(64, 64),
                                             SingleConv2DBlock(64, output_dim, 1))

    def forward(self, x):
        z = self.transformer(x)
        z0, z3, z6, z9, z12 = x, *z
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)   # shape: (batch_size, 768, 16, 16)
        # 将z12用nn.AdaptiveAvgPool2d(1)降维
        z12c = nn.AdaptiveAvgPool2d(1)(z12)  # shape: (batch_size, 768, 1, 1)
        # flatten
        z12c = z12c.view(z12c.size(0), -1)  # shape: (batch_size, 768),-1表示自动计算
        # dropout
        # z12c = F.dropout(z12c, p=self.dropout, training=self.training)
        # linear
        z12c = self.linear(z12c)  # shape: (batch_size, 3)

        # z3 = torch.mean(z3.view(z3.size(0), z3.size(1), -1), dim=2)  # shape: (batch_size, 768)
        # z6 = torch.mean(z6.view(z6.size(0), z6.size(1), -1), dim=2)  # shape: (batch_size, 768)
        # z9 = torch.mean(z9.view(z9.size(0), z9.size(1), -1), dim=2)  # shape: (batch_size, 768)
        # z12 = torch.mean(z12.view(z12.size(0), z12.size(1), -1), dim=2)  # shape: (batch_size, 768)
        # features = torch.cat((z3, z6, z9, z12), dim=1)  # shape: (batch_size, 768*4)
        # input_size = features.size(1)
        # print('input_size is ', input_size)

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))

        return z12c, output


class UNETRcls(nn.Module):
    def __init__(self, img_shape=(224, 224), input_dim=1, output_dim=1, embed_dim=768, patch_size=16, num_heads=12,
                 dropout=0.1, batch_size=10):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]
        # self.fc = nn.Linear(embed_dim, self.output_dim, bias=True)  # bias=True 是指是否使用偏置
        self.fc1 = nn.Linear(embed_dim * 1, 512)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, self.output_dim)

        self.patch_dim = [int(x / patch_size) for x in img_shape]

        # Transformer Encoder
        self.transformer = Transformer(input_dim, embed_dim, img_shape, patch_size, num_heads, self.num_layers,
                                       dropout,
                                       self.ext_layers)

    def forward(self, x):
        z = self.transformer(x)
        z12 = z[-1]
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)  # shape: (batch_size, 768, 16, 16)
        # 将z12用nn.AdaptiveAvgPool2d(1)降维
        z12 = nn.AdaptiveAvgPool2d(1)(z12)  # shape: (batch_size, 768, 1, 1)
        z12 = z12.view(z12.size(0), -1)  # shape: (batch_size, 768),-1表示自动计算
        z12c = F.dropout(z12, p=self.dropout, training=self.training)

        z12c = self.fc1(z12c)
        z12c = self.dropout1(z12c)
        out = self.fc2(z12c)

        # z3 = torch.mean(z3.view(z3.size(0), z3.size(1), -1), dim=2)  # shape: (batch_size, 768)
        # z6 = torch.mean(z6.view(z6.size(0), z6.size(1), -1), dim=2)  # shape: (batch_size, 768)
        # z9 = torch.mean(z9.view(z9.size(0), z9.size(1), -1), dim=2)  # shape: (batch_size, 768)
        # z12 = torch.mean(z12.view(z12.size(0), z12.size(1), -1), dim=2)  # shape: (batch_size, 768)
        # features = torch.cat((z3, z6, z9, z12), dim=1)  # shape: (batch_size, 768*4)
        # input_size = features.size(1)
        # print('input_size is ', input_size)

        return out


class UNETRseg(nn.Module):
    def __init__(self, img_shape=(224, 224), input_dim=1, output_dim=1, embed_dim=768, patch_size=16, num_heads=12,
                 dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]

        self.patch_dim = [int(x / patch_size) for x in img_shape]

        # Transformer Encoder
        self.transformer = Transformer(input_dim, embed_dim, img_shape, patch_size, num_heads, self.num_layers, dropout,
                                       self.ext_layers)

        # U-Net Decoder
        self.decoder0 = nn.Sequential(Conv2DBlock(input_dim, 32, 3),Conv2DBlock(32, 64, 3))
        self.decoder3 = nn.Sequential(Deconv2DBlock(embed_dim, 512), Deconv2DBlock(512, 256), Deconv2DBlock(256, 128))
        self.decoder6 = nn.Sequential(Deconv2DBlock(embed_dim, 512), Deconv2DBlock(512, 256),)
        self.decoder9 = Deconv2DBlock(embed_dim, 512)
        self.decoder12_upsampler = SingleDeconv2DBlock(embed_dim, 512)
        self.decoder9_upsampler = nn.Sequential(Conv2DBlock(1024, 512), Conv2DBlock(512, 512), Conv2DBlock(512, 512),
                                                SingleDeconv2DBlock(512, 256))
        self.decoder6_upsampler = nn.Sequential(Conv2DBlock(512, 256), Conv2DBlock(256, 256),
                                                SingleDeconv2DBlock(256, 128))
        self.decoder3_upsampler = nn.Sequential(Conv2DBlock(256, 128), Conv2DBlock(128, 128),
                                                SingleDeconv2DBlock(128, 64))
        self.decoder0_header = nn.Sequential(Conv2DBlock(128, 64), Conv2DBlock(64, 64),
                                             SingleConv2DBlock(64, output_dim, 1))

    def forward(self, x):
        # z = self.transformer(x)
        # z0, z3, z6, z9, z12 = x, *z
        # # print('z0 shape is ', z0.shape)
        # # print('z3 shape is ', z3.shape)
        # # print('z6 shape is ', z6.shape)
        # # print('z9 shape is ', z9.shape)
        # # print('z12 shape is ', z12.shape)
        #
        # z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)  # *self.
        # z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        # z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        # z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)   # shape: (batch_size, 768, 16, 16)
        # # print('z0 shape is ', z0.shape)
        # # print('z3 shape is ', z3.shape)
        # # print('z6 shape is ', z6.shape)
        # # print('z9 shape is ', z9.shape)
        # # print('z12 shape is ', z12.shape)
        #
        # z12 = self.decoder12_upsampler(z12)
        # # print('z12 shape is ', z12.shape)
        # z9 = self.decoder9(z9)
        # # print('z9 shape is ', z9.shape)
        # z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        # # print('z9 shape is ', z9.shape)
        # z6 = self.decoder6(z6)
        # # print('z6 shape is ', z6.shape)
        # z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        # # print('z6 shape is ', z6.shape)
        # z3 = self.decoder3(z3)
        # # print('z3 shape is ', z3.shape)
        # z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        # # print('z3 shape is ', z3.shape)
        # z0 = self.decoder0(z0)
        # # print('z0 shape is ', z0.shape)
        # output = self.decoder0_header(torch.cat([z0, z3], dim=1))
        # # print('output shape is ', output.shape)
        #
        # return output

        z = self.transformer(x)
        z0, z3, z6, z9, z12 = x, *z
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))

        return output


class SwinEmbeddings(nn.Module):
    def __init__(self, input_dim, embed_dim, img_size, patch_size, dropout):
        super().__init__()
        self.n_patches = int((img_size[0] * img_size[1]) / (patch_size * patch_size))
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv2d(in_channels=input_dim, out_channels=embed_dim,
                                          kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class SwinSelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output


class WindowAttention(nn.Module):
    def __init__(self, window_size, num_heads, embed_dim, dropout):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.attn = SwinSelfAttention(num_heads, embed_dim, dropout)

    def forward(self, x, H, W, attention_mask=None):
        B, N, C = x.shape
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, self.window_size * self.window_size, C)

        if attention_mask is not None:
            attention_mask = attention_mask.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size)
            attention_mask = attention_mask.permute(0, 1, 3, 2, 4).contiguous().view(B, -1, self.window_size * self.window_size)

        attn_out = self.attn(x, attention_mask)
        attn_out = attn_out.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, C)
        attn_out = attn_out.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, N, C)

        return attn_out


class SwinTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = WindowAttention(window_size, num_heads, embed_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = PositionwiseFeedForward(embed_dim, 2048, dropout)

    def forward(self, x, H, W, attention_mask=None):
        attn_out = self.attn(self.norm1(x), H, W, attention_mask)
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        return x


class SwinTransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, img_shape, patch_size, num_heads, num_layers, dropout, window_size, extract_layers):
        super().__init__()
        self.embeddings = SwinEmbeddings(input_dim, embed_dim, img_shape, patch_size, dropout)
        self.layers = nn.ModuleList()
        self.extract_layers = extract_layers
        for _ in range(num_layers):
            layer = SwinTransformerBlock(embed_dim, num_heads, window_size, dropout)
            self.layers.append(layer)

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]
        x = self.embeddings(x)
        H, W = H // self.embeddings.patch_size, W // self.embeddings.patch_size
        x = x.view(x.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()

        extract_layers = []
        for depth, layer in enumerate(self.layers):
            x = layer(x, H, W)
            if depth + 1 in self.extract_layers:
                extract_layers.append(x)

        return extract_layers


class UNETswin(nn.Module):
    def __init__(self, img_shape=(224, 224), input_dim=1, output_dim=1, embed_dim=768, patch_size=32, num_heads=12,
                 dropout=0.1, batch_size=10):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]

        self.patch_dim = [int(x / patch_size) for x in img_shape]

        # Transformer Encoder
        self.swinformer = timm.create_model('swin_small_patch4_window7_224', pretrained=True, in_chans=input_dim)
        self.transformer = Transformer(input_dim, embed_dim, img_shape, patch_size, num_heads, self.num_layers, dropout,
                                       self.ext_layers)
        # U-Net Decoder
        self.decoder0 = nn.Sequential(Conv2DBlock(input_dim, 32, 3), Conv2DBlock(32, 64, 3))
        self.decoder3 = nn.Sequential(Deconv2DBlock(embed_dim, 512), Deconv2DBlock(512, 256),
                                      Deconv2DBlock(256, 128))
        self.decoder6 = nn.Sequential(Deconv2DBlock(embed_dim, 512), Deconv2DBlock(512, 256), )
        self.decoder9 = Deconv2DBlock(embed_dim, 512)
        self.decoder12_upsampler = SingleDeconv2DBlock(embed_dim, 512)
        self.decoder9_upsampler = nn.Sequential(Conv2DBlock(1024, 512), Conv2DBlock(512, 512),
                                                Conv2DBlock(512, 512),
                                                SingleDeconv2DBlock(512, 256))
        self.decoder6_upsampler = nn.Sequential(Conv2DBlock(512, 256), Conv2DBlock(256, 256),
                                                SingleDeconv2DBlock(256, 128))
        self.decoder3_upsampler = nn.Sequential(Conv2DBlock(256, 128), Conv2DBlock(128, 128),
                                                SingleDeconv2DBlock(128, 64))
        self.decoder0_header = nn.Sequential(Conv2DBlock(128, 64), Conv2DBlock(64, 64),
                                             SingleConv2DBlock(64, output_dim, 1))

    def forward(self, x):
        z = self.transformer(x)
        feature = self.swinformer.forward_features(x)
        print(feature.shape)

        z0, z3, z6, z9, z12 = x, *z

        # z12 和 feature在第二个维度进行相加得到新的z12

        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        feature = feature.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        # z12 = z12 + feature

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))

        return output


class UNETRSwin(nn.Module):
    def __init__(self, img_shape=(224, 224), input_dim=1, output_dim=1, embed_dim=768, patch_size=16, num_heads=12,
                 dropout=0.1, batch_size=10, window_size=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]
        self.window_size = window_size

        # Swin Transformer Encoder
        self.swin_transformer = SwinTransformerEncoder(input_dim, embed_dim, img_shape, patch_size, num_heads,
                                                       self.num_layers, dropout,
                                                       window_size, self.ext_layers)

        # U-Net Decoder
        self.decoder0 = nn.Sequential(Conv2DBlock(input_dim, 32, 3), Conv2DBlock(32, 64, 3))
        self.decoder3 = nn.Sequential(Deconv2DBlock(embed_dim, 512), Deconv2DBlock(512, 256),
                                      Deconv2DBlock(256, 128))
        self.decoder6 = nn.Sequential(Deconv2DBlock(embed_dim, 512), Deconv2DBlock(512, 256), )
        self.decoder9 = Deconv2DBlock(embed_dim, 512)
        self.decoder12_upsampler = SingleDeconv2DBlock(embed_dim, 512)
        self.decoder9_upsampler = nn.Sequential(Conv2DBlock(1024, 512), Conv2DBlock(512, 512),
                                                Conv2DBlock(512, 512),
                                                SingleDeconv2DBlock(512, 256))
        self.decoder6_upsampler = nn.Sequential(Conv2DBlock(512, 256), Conv2DBlock(256, 256),
                                                SingleDeconv2DBlock(256, 128))
        self.decoder3_upsampler = nn.Sequential(Conv2DBlock(256, 128), Conv2DBlock(128, 128),
                                                SingleDeconv2DBlock(128, 64))
        self.decoder0_header = nn.Sequential(Conv2DBlock(128, 64), Conv2DBlock(64, 64),
                                             SingleConv2DBlock(64, output_dim, 1))

    def forward(self, x):
        z = self.swin_transformer(x)
        z0, z3, z6, z9, z12 = x, *z

        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))

        return output


if __name__ == '__main__':
    model = UNETRcls()
    x = torch.randn(2, 1, 224, 224)
    y = model(x)
    print(y.shape)
