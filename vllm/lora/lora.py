from typing import List, Optional
from typing import Sequence as GenericSequence

import torch
import torch.types

from vllm.lora.peft_helper import PEFTHelper
from vllm.utils import is_pin_memory_available


class LoRALayerWeights:
    """LoRA weights for a layer composed of two low rank matrixes."""

    def __init__(
        self,
        module_name: str,
        rank: Optional[int],
        lora_alpha: int,
        lora_a: Optional[torch.Tensor],
        lora_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        embeddings_tensor: Optional[torch.Tensor] = None,
        scaling: Optional[float] = None,
    ) -> None:
        '''
        rank == None means that we have full rank tensors (ModulesToSave)
        in this case:
            lora_a=None
            lora_b=full rank tensor
        '''

        self.module_name = module_name
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.bias = bias
        self.embeddings_tensor = embeddings_tensor

        self.scaling = scaling

        if (scaling is None) and (self.rank is not None):
            self.scaling = self.lora_alpha / self.rank

    def optimize(self) -> "LoRALayerWeights":
        """Optimize the LoRA by merging the scaling into lora_b."""
        if self.scaling == 1:
            return self
        if self.scaling is not None:
            self.lora_b *= self.scaling
        self.scaling = 1
        return self

    @property
    def input_dim(self) -> int:
        if self.lora_a is not None:
            return self.lora_a.shape[0]

        return self.lora_b.shape[0]

    @property
    def output_dim(self) -> int:
        return self.lora_b.shape[1]

    @property
    def is_packed(self) -> bool:
        return False

    @property
    def extra_vocab_size(self) -> int:
        return self.embeddings_tensor.shape[
            0] if self.embeddings_tensor is not None else 0

    @classmethod
    def from_config(
        cls,
        module_name: str,
        peft_helper: PEFTHelper,
        embeddings_tensor: Optional[torch.Tensor] = None,
    ) -> "LoRALayerWeights":
        return cls(module_name, peft_helper.r, peft_helper.lora_alpha, None,
                   None, None, embeddings_tensor,
                   peft_helper.vllm_lora_scaling_factor)

    @classmethod
    def create_dummy_lora_weights(
            cls,
            module_name: str,
            input_dim: int,
            output_dim: int,
            rank: Optional[int],
            dtype: torch.dtype,
            device: torch.types.Device,
            embeddings_tensor_dim: Optional[int] = None,
            bias_enabled: Optional[bool] = False) -> "LoRALayerWeights":
        pin_memory = str(device) == "cpu" and is_pin_memory_available()
        if rank is None:
            lora_a = None
            lora_b = torch.zeros([input_dim, output_dim],
                                 dtype=dtype,
                                 device=device,
                                 pin_memory=pin_memory)
            embeddings_tensor = None
            bias=None
        else:
            lora_a = torch.zeros([input_dim, rank],
                                 dtype=dtype,
                                 device=device,
                                 pin_memory=pin_memory)
            lora_b = torch.zeros([rank, output_dim],
                                 dtype=dtype,
                                 device=device,
                                 pin_memory=pin_memory)
            if bias_enabled:
                bias = torch.zeros([output_dim],
                                   dtype=dtype,
                                   device=device,
                                   pin_memory=pin_memory)
            else:
                bias = None

            embeddings_tensor = torch.rand(
                10,
                embeddings_tensor_dim,
                dtype=dtype,
                device=device,
                pin_memory=pin_memory) if embeddings_tensor_dim else None
        return cls(
            module_name,
            rank=rank,
            lora_alpha=1,
            lora_a=lora_a,
            lora_b=lora_b,
            bias=bias,
            embeddings_tensor=embeddings_tensor,
        )

    def lora_a_pin_memory(self):
        if self.lora_a is not None:
            self.lora_a = self.lora_a.pin_memory()

    def lora_b_pin_memory(self):
        self.lora_b = self.lora_b.pin_memory()


class PackedLoRALayerWeights(LoRALayerWeights):
    """LoRA used for packed layers (eg. qkv_proj)."""

    def __init__(
        self,
        module_name: str,
        rank: Optional[int],
        lora_alphas: List[Optional[int]],
        lora_a: List[Optional[torch.Tensor]],
        lora_b: List[Optional[torch.Tensor]],
        bias: Optional[List[Optional[torch.Tensor]]] = None,
        scaling: Optional[List[float]] = None,
    ) -> None:
        super().__init__(
            module_name=module_name,
            rank=rank,
            lora_alpha=0,
            lora_a=lora_a,
            lora_b=lora_b,
            bias=bias,
            scaling=scaling,  # type: ignore
            embeddings_tensor=None,
        )
        self.lora_alphas = lora_alphas
        if (scaling is None) and (self.rank is not None):
            self.scaling = [  # type: ignore
                lora_alpha / self.rank  # type: ignore # noqa
                for lora_alpha in self.lora_alphas
            ]

    @classmethod
    def pack(
        cls, loras: GenericSequence[Optional["LoRALayerWeights"]]
    ) -> "PackedLoRALayerWeights":
        """Pack a list of LoRAs into a single LoRA.

        If LoRA is None, it signifies that the submodule does not have a LoRA.
        """
        first_lora = next(lora for lora in loras if lora is not None)
        for lora in loras:
            if lora is None:
                continue
            lora.optimize()
        rank = first_lora.rank
        module_name = first_lora.module_name
        obj = cls(
            module_name,
            rank,
            [lora.lora_alpha if lora is not None else None for lora in loras],
            [lora.lora_a if lora is not None else None for lora in loras],
            [lora.lora_b if lora is not None else None for lora in loras],
            [lora.bias if lora is not None else None for lora in loras],
            scaling=[
                1 if lora is not None else None  # type: ignore
                for lora in loras
            ])
        return obj

    def optimize(self) -> "PackedLoRALayerWeights":
        """Optimize the LoRA by merging the scaling into lora_b."""
        for i in range(len(self.lora_b)):
            if self.scaling[i] == 1 or self.lora_b[i] is None:  # type: ignore
                continue
            self.lora_b[i] *= self.scaling[i]  # type: ignore
            self.scaling[i] = 1  # type: ignore
        return self

    @property
    def input_dim(self) -> int:
        raise NotImplementedError()

    @property
    def output_dim(self) -> int:
        raise NotImplementedError()

    @property
    def is_packed(self) -> bool:
        return True
