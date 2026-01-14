from dataclasses import dataclass, field


@dataclass(frozen=True)
class VisionConfig:
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    hidden_dim: int = 384
    num_layers: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    layer_norm_eps: float = 1e-6
    include_cls_token: bool = True

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size:
            raise ValueError(f"image_size {self.image_size} not divisible by patch_size {self.patch_size}")
        if self.hidden_dim % self.num_heads:
            raise ValueError(f"hidden_dim {self.hidden_dim} not divisible by num_heads {self.num_heads}")

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def num_visual_tokens(self) -> int:
        return self.num_patches + int(self.include_cls_token)


@dataclass(frozen=True)
class TextConfig:
    vocab_size: int = 16384
    hidden_dim: int = 512
    intermediate_dim: int = 1376  # ~8/3 * hidden, rounded to a multiple of 32
    num_layers: int = 8
    num_q_heads: int = 8
    num_kv_heads: int = 2  # 4 query heads share each kv head
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.hidden_dim % self.num_q_heads:
            raise ValueError(f"hidden_dim {self.hidden_dim} not divisible by num_q_heads {self.num_q_heads}")
        if self.num_q_heads % self.num_kv_heads:
            raise ValueError(f"num_q_heads {self.num_q_heads} not divisible by num_kv_heads {self.num_kv_heads}")

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_q_heads

    @property
    def num_kv_groups(self) -> int:
        return self.num_q_heads // self.num_kv_heads


@dataclass(frozen=True)
class VLMConfig:
    vision: VisionConfig = field(default_factory=VisionConfig)
    text: TextConfig = field(default_factory=TextConfig)
    projector_hidden_dim: int = 512
    token_pooling_factor: int = 1  # >1 concatenates groups of visual tokens before projection, see projector.py
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    @property
    def num_projected_visual_tokens(self) -> int:
        return self.vision.num_visual_tokens // self.token_pooling_factor

    # Parameter counts below are with tied embeddings.

    @classmethod
    def tiny(cls) -> "VLMConfig":
        """~2.1M params, 64x64 images. For tests and the walkthrough notebook."""
        return cls(
            vision=VisionConfig(image_size=64, patch_size=16, hidden_dim=128, num_layers=2, num_heads=4),
            text=TextConfig(
                vocab_size=2048, hidden_dim=192, intermediate_dim=512,
                num_layers=3, num_q_heads=6, num_kv_heads=2, max_seq_len=512,
            ),
            projector_hidden_dim=192,
        )

    @classmethod
    def compact(cls) -> "VLMConfig":
        """~42M params. Fits comfortably on a 4GB card (GTX 1650)."""
        return cls(
            vision=VisionConfig(image_size=224, patch_size=16, hidden_dim=384, num_layers=6, num_heads=6),
            text=TextConfig(
                vocab_size=16384, hidden_dim=512, intermediate_dim=1376,
                num_layers=8, num_q_heads=8, num_kv_heads=2, max_seq_len=1024,
            ),
            projector_hidden_dim=512,
        )

    @classmethod
    def base(cls) -> "VLMConfig":
        """~127M params. Needs a bigger GPU."""
        return cls(
            vision=VisionConfig(image_size=224, patch_size=16, hidden_dim=512, num_layers=8, num_heads=8),
            text=TextConfig(
                vocab_size=32000, hidden_dim=768, intermediate_dim=2048,
                num_layers=12, num_q_heads=12, num_kv_heads=4, max_seq_len=2048,
            ),
            projector_hidden_dim=768,
        )

    @classmethod
    def from_name(cls, name: str) -> "VLMConfig":
        presets = {"tiny": cls.tiny, "compact": cls.compact, "base": cls.base}
        if name not in presets:
            raise ValueError(f"unknown profile {name!r}, pick one of {sorted(presets)}")
        return presets[name]()
