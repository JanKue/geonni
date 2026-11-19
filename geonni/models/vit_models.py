import torch
from torch import Tensor, nn


class TransformerEncoderBlock(nn.Module):
    """General-purpose Transformer encoder block."""
    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float = 0.0,
            activation_layer: nn.Module = nn.ReLU, # alternative: nn.GeLU
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
                                               dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            activation_layer(inplace=True),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout, inplace=True)
        )


    def forward(self, input: Tensor) -> Tensor:
        emb1: Tensor = self.norm1(input)
        emb1, _ = self.self_attention(query=emb1, key=emb1, value=emb1, need_weights=False)

        emb2: Tensor = self.norm2(emb1 + input)
        emb2 = self.mlp(emb2)

        output = emb1 + emb2
        return output


class TransformerEncoder(nn.Module):
    """General-purpose Transformer encoder composed of multiple layers."""
    def __init__(
            self,
            seq_len: int,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float = 0.1,
    ):
        super().__init__()

        self.pos_encoding = nn.Parameter(data=torch.empty(1, seq_len, hidden_dim).normal_(std=0.02))

        layers = nn.Sequential()
        for _ in range(num_layers):
            layers.append(
                TransformerEncoderBlock(num_heads=num_heads, hidden_dim=hidden_dim,
                                        mlp_dim=mlp_dim, dropout=dropout, activation_layer=nn.ReLU)
            )
        self.layers = layers
        self.norm = nn.LayerNorm(hidden_dim)


    def forward(self, input: Tensor) -> Tensor:
        emb = input + self.pos_encoding
        output: Tensor = self.layers(emb)
        output = self.norm(output)
        return output


class ImageToSequence(nn.Module):
    """
    Module to transform a given image (batch) into a sequence of embeddings which encode patches extracted from the image.

    The input image must be square with sides of `img_size`. The patches are squares with sides `patch_size`.
    `img_size` must be divisible by `patch_size`.
    """
    def __init__(self, img_size: int, patch_size: int, hidden_dim: int):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.seq_len = (img_size // patch_size) ** 2
        self.hidden_dim = hidden_dim

        self.projection = nn.Conv2d(in_channels=3, out_channels=hidden_dim,
                                    kernel_size=patch_size, stride=patch_size)


    def forward(self, input: Tensor) -> Tensor: # [B, C, H, W] -> [B, seq_len, hidden_dim]
        n_batch, channels, height, width = input.shape
        assert height == width == self.img_size
        assert channels == 3

        output: Tensor = self.projection(input) # [B, hidden_dim, H//patch_size, W//patch_size]
        output = output.reshape(n_batch, self.hidden_dim, self.seq_len) # [B, hidden_dim, seq_len]
        output = output.permute([0, 2, 1]) # [B, seq_len, hidden_dim]

        return output


class ClassificationViT(nn.Module):
    """VisionTransformer for classification"""
    def __init__(
            self,
            num_classes: int,
            img_size: int,
            patch_size: int,
            hidden_dim: int = 512,
            num_heads: int = 16,
            num_encoder_layers: int = 12,
            dropout: float = 0.1,
            **kwargs
    ):
        super().__init__()
        
        self.image_to_sequence = ImageToSequence(img_size=img_size, patch_size=patch_size, hidden_dim=hidden_dim)
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_len_cls = self.image_to_sequence.seq_len + 1 # +1 represents class token

        self.encoder = TransformerEncoder(seq_len=seq_len_cls, hidden_dim=hidden_dim, mlp_dim=4 * hidden_dim,
                                          num_layers=num_encoder_layers, num_heads=num_heads, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)


    def forward(self, input: Tensor) -> Tensor:
        n_batch, *_ = input.shape # expected input shape: [n_batch, 3, img_size, img_size]
        input_seq: Tensor = self.image_to_sequence(input) # [n_batch, seq_len, hidden_dim]
        # add the class token to the start of the sequence
        exp_class_token = self.class_token.expand(n_batch, -1, -1) # [n_batch, seq_len_cls, hidden_dim]
        emb: Tensor = torch.cat([exp_class_token, input_seq], dim=-2) # [n_batch, seq_len_cls, hidden_dim]
        # pass through the encoder
        emb = self.encoder(emb) # [n_batch, seq_len_cls, hidden_dim]
        # use just the class token as classifier input
        output: Tensor = self.classifier(emb[..., 0, :]) # [n_batch, num_classes]
        return output
