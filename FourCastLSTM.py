import torch
import torch.nn as nn
from timm.models.layers import to_2tuple

from modules.Afnonet import Block
from modules.Moudles import DoubleConv, GFU, CASA


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class PatchInflated(nn.Module):
    def __init__(self, in_chans, embed_dim, input_resolution, stride=2, padding=1, output_padding=1):
        super(PatchInflated, self).__init__()

        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        output_padding = to_2tuple(output_padding)
        self.input_resolution = input_resolution

        self.ConvT = nn.ConvTranspose2d(in_channels=embed_dim, out_channels=in_chans, kernel_size=(3, 3),
                                        stride=stride, padding=padding, output_padding=output_padding)
        self.Conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 3), stride=stride, padding=padding,
                output_padding=output_padding),
            nn.GroupNorm(16, embed_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        # x = self.ConvT(x)
        x = self.Conv(x)
        x = self.ConvT(x)
        return x


class FourCastNetBlocks(nn.Module):

    def __init__(self, dim, input_resolution, depth, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super(FourCastNetBlocks, self).__init__()
        drop_rate = 0.
        drop_path_rate = 0.
        sparsity_threshold = 0.01
        hard_thresholding_fraction = 1.0
        num_blocks = 16
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = nn.ModuleList([
            Block(dim=dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                  num_blocks=num_blocks, sparsity_threshold=sparsity_threshold,
                  hard_thresholding_fraction=hard_thresholding_fraction, input_resolution=input_resolution)
            for i in range(depth)])

    def forward(self, xt, hx):

        outputs = []

        for index, layer in enumerate(self.layers):
            if index == 0:
                x = layer(xt, hx)
                outputs.append(x)

            else:
                if index % 2 == 0:
                    x = layer(outputs[-1], xt)
                    outputs.append(x)

                if index % 2 == 1:
                    x = layer(outputs[-1], None)
                    outputs.append(x)

        return outputs[-1]


class FourCastLSTMCell(nn.Module):

    def __init__(self, dim, input_resolution, depth,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super(FourCastLSTMCell, self).__init__()

        self.Block = FourCastNetBlocks(dim=dim, input_resolution=input_resolution, depth=depth,
                                       mlp_ratio=mlp_ratio, norm_layer=norm_layer)

    def forward(self, xt, hidden_states):
        if hidden_states is None:
            B, L, C = xt.shape
            hx = torch.zeros(B, L, C).to(xt.device)
            cx = torch.zeros(B, L, C).to(xt.device)

        else:
            hx, cx = hidden_states

        Ft = self.Block(xt, hx)

        gate = torch.sigmoid(Ft)
        cell = torch.tanh(Ft)

        cy = gate * (cx + cell)
        hy = gate * torch.tanh(cy)
        hx = hy
        cx = cy

        return hx, (hx, cx)


class Func(nn.Module):

    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm):

        super(Func, self).__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.img_size = img_size
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed_rst = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                          in_chans=in_chans, embed_dim=in_chans)
        patches_resolution = self.patch_embed.patches_resolution

        self.PatchInflated = PatchInflated(in_chans=in_chans, embed_dim=embed_dim, input_resolution=patches_resolution)
        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = FourCastLSTMCell(dim=embed_dim,
                                     input_resolution=(patches_resolution[0], patches_resolution[1]),
                                     depth=depths[i_layer],
                                     mlp_ratio=self.mlp_ratio,
                                     norm_layer=norm_layer)

            self.layers.append(layer)
        self.cnn = DoubleConv(embed_dim, embed_dim)
        self.casa = CASA(embed_dim)
        self.h = img_size // patch_size
        self.gfu = GFU(embed_dim)

    def forward(self, x, h):

        x = self.patch_embed(x)
        B, L, C = x.shape
        local_x = x.reshape(B, self.h, self.h, C).permute(0, 3, 2, 1)
        local_x = self.cnn(local_x)
        local_x = self.casa(local_x)
        x = local_x
        x = x.flatten(2).transpose(1, 2)
        hidden_states = []
        for index, layer in enumerate(self.layers):
            x, hidden_state = layer(x, h[index])
            hidden_states.append(hidden_state)

        global_x = x.reshape(B, self.h, self.h, C).permute(0, 3, 2, 1)
        x = self.gfu(global_x, local_x)
        x = x.flatten(2).transpose(1, 2)

        x = torch.sigmoid(self.PatchInflated(x))

        return hidden_states, x


class FourCastLSTM(nn.Module):

    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths,
                 ):
        super(FourCastLSTM, self).__init__()

        self.func = Func(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                       embed_dim=embed_dim, depths=depths,
                       )

    def forward(self, inputs, states):
        states_next, output = self.func(inputs, states)

        return output, states_next
