import numpy as np

from esm.utils.structure.protein_chain import ProteinChain

from src.protein_chain import WrappedProteinChain
from esm.utils.structure.protein_complex import ProteinComplex
from esm.utils.constants import esm3 as esm3_c

from biotite.structure.io.pdbx import CIFFile, convert
from biotite.structure.io.pdb import PDBFile
from biotite.structure import concatenate

import biotite.structure as bs
from Bio.Data import PDBData
from esm.utils import residue_constants as RC
import torch

from dataclasses import asdict, dataclass, replace
from esm.utils.structure.normalize_coordinates import normalize_coordinates

import io
from pathlib import Path
from cloudpathlib import CloudPath
from typing import Sequence, TypeVar, Union
PathLike = Union[str, Path, CloudPath]
PathOrBuffer = Union[PathLike, io.StringIO]

from esm.utils import residue_constants
from esm.utils.structure.protein_chain import infer_CB

@dataclass
class ProteinComplexMetadata:
    entity_lookup: dict[int, int]
    chain_lookup: dict[int, str]
    chain_boundaries: list[tuple[int, int]]


class WrappedProteinComplex(ProteinComplex):
    """
    https://github.com/evolutionaryscale/esm/blob/main/esm/utils/structure/protein_complex.py
    """
    def to_structure_encoder_inputs(
        self,
        device="cpu",
        should_normalize_coordinates: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coords = torch.tensor(self.atom37_positions, dtype=torch.float32, device=device)
        plddt = torch.tensor(self.confidence, dtype=torch.float32, device=device)
        residue_index = torch.tensor(self.residue_index, dtype=torch.long, device=device)
        chain_index = torch.tensor(self.chain_id, dtype=torch.long, device=device)

        if should_normalize_coordinates:
            coords = normalize_coordinates(coords)
        return coords.unsqueeze(0), plddt.unsqueeze(0), residue_index.unsqueeze(0), chain_index.unsqueeze(0)

    @classmethod
    def from_chains(cls, chains: Sequence[ProteinChain]):
        if not chains:
            raise ValueError(
                "Cannot create a ProteinComplex from an empty list of chains"
            )

        # TODO: Make a proper protein complex class
        def join_arrays(arrays: Sequence[np.ndarray], sep: np.ndarray):
            full_array = []
            for array in arrays:
                full_array.append(array)
                full_array.append(sep)
            full_array = full_array[:-1]
            return np.concatenate(full_array, 0)

        sep_tokens = {
            "residue_index": np.array([-1]),
            "insertion_code": np.array([""]),
            "atom37_positions": np.full([1, 37, 3], np.nan),
            "atom37_mask": np.zeros([1, 37], dtype=bool),
            "confidence": np.array([0]),
        }

        array_args: dict[str, np.ndarray] = {
            name: join_arrays([getattr(chain, name) for chain in chains], sep)
            for name, sep in sep_tokens.items()
        }

        multimer_arrays = []
        chain2num_max = -1
        chain2num = {}
        ent2num_max = -1
        ent2num = {}
        total_index = 0
        chain_boundaries = []
        for i, c in enumerate(chains):
            num_res = c.residue_index.shape[0]
            if c.chain_id not in chain2num:
                chain2num[c.chain_id] = (chain2num_max := chain2num_max + 1)
            chain_id_array = np.full([num_res], chain2num[c.chain_id], dtype=np.int64)

            if c.entity_id is None:
                entity_num = (ent2num_max := ent2num_max + 1)
            else:
                if c.entity_id not in ent2num:
                    ent2num[c.entity_id] = (ent2num_max := ent2num_max + 1)
                entity_num = ent2num[c.entity_id]
            entity_id_array = np.full([num_res], entity_num, dtype=np.int64)

            sym_id_array = np.full([num_res], i, dtype=np.int64)

            multimer_arrays.append(
                {
                    "chain_id": chain_id_array,
                    "entity_id": entity_id_array,
                    "sym_id": sym_id_array,
                }
            )

            chain_boundaries.append((total_index, total_index + num_res))
            total_index += num_res + 1

        sep = np.array([-1])
        update = {
            name: join_arrays([dct[name] for dct in multimer_arrays], sep=sep)
            for name in ["chain_id", "entity_id", "sym_id"]
        }
        array_args.update(update)

        metadata = ProteinComplexMetadata(
            chain_boundaries=chain_boundaries,
            chain_lookup={v: k for k, v in chain2num.items()},
            entity_lookup={v: k for k, v in ent2num.items()},
        )

        return cls(
            id=chains[0].id,
            sequence=esm3_c.CHAIN_BREAK_STR.join(chain.sequence for chain in chains),
            metadata=metadata,
            **array_args,
        )

    @classmethod
    def from_pdb(
        cls,
        path: PathOrBuffer,
        chain_list,
        id: str | None = None
    ):

        atom_array = PDBFile.read(path).get_structure(
            model=1, extra_fields=["b_factor"]
        )

        atom_array_list = []
        for chain_id in chain_list:
            atom_array_list.append(atom_array[
                bs.filter_amino_acids(atom_array)
                & ~atom_array.hetero
                & np.isin(atom_array.chain_id, chain_id)
            ])
        atom_array = concatenate(atom_array_list)

        chains = []
        for chain in bs.chain_iter(atom_array):
            chain = chain[~chain.hetero]
            if len(chain) == 0:
                continue
            chains.append(WrappedProteinChain.from_atomarray(chain, id))
        return WrappedProteinComplex.from_chains(chains)
    
    

    def infer_cbeta(self, infer_cbeta_for_glycine: bool = False) -> ProteinComplex:
        """Return a new chain with inferred CB atoms at all residues except GLY.

        Args:
            infer_cbeta_for_glycine (bool): If True, infers a beta carbon for glycine
                residues, even though that residue doesn't have one.  Default off.

                NOTE(rverkuil): The reason for having this switch in the first place
                is that sometimes we want a (inferred) CB coordinate for every residue,
                for example for making a pairwise distance matrix, or doing an RMSD
                calculation between two designs for a given structural template, w/
                CB atoms.
        """


        atom37_positions = self.atom37_positions.copy()
        atom37_mask = self.atom37_mask.copy()

        N, CA, C = np.moveaxis(self.atoms[["N", "CA", "C"]], 1, 0)
        # See usage in trDesign codebase.
        # https://github.com/gjoni/trDesign/blob/f2d5930b472e77bfacc2f437b3966e7a708a8d37/02-GD/utils.py#L140
        inferred_cbeta_positions = infer_CB(C, N, CA, 1.522, 1.927, -2.143)
        if not infer_cbeta_for_glycine:
            inferred_cbeta_positions[np.array(list(self.sequence)) == "G", :] = np.nan

        atom37_positions[:, residue_constants.atom_order["CB"]] = (
            inferred_cbeta_positions
        )
        atom37_mask[:, residue_constants.atom_order["CB"]] = ~np.isnan(
            atom37_positions[:, residue_constants.atom_order["CB"]]
        ).any(-1)
        new_chain = replace(
            self, atom37_positions=atom37_positions, atom37_mask=atom37_mask
        )
        return new_chain