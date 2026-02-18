# import os
# # import time
# from tqdm import tqdm
# # from collections import Counter, defaultdict

# import torch
# import torch.distributed as dist

# from protein_chain import WrappedProteinChain
# import util
# from dataset.base import BaseDataset, convert_chain_id
# # from dataset.cath import CATHLabelMappingDataset
# import json

# from tokenizer import *

import os
import time
import json
from tqdm import tqdm
import gc
from glob import glob

import torch
import numpy as np
import torch.distributed as dist

from sklearn.model_selection import train_test_split

from Bio import PDB
import biotite
from biotite.sequence import Alphabet, Sequence, GeneralSequence
from biotite.sequence.align import align_optimal, SubstitutionMatrix

from dataset.base import BaseDataset
from esm.utils.constants import esm3 as C
import util
from protein_chain import WrappedProteinChain
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

class RAbDDataset(BaseDataset):
    """
    Though this class inherents from BaseDataset, they are implemented with 
    different logic of processing, especially indicated by __init__(). Functions
    not designed to be used in this class has "assert" warnings.
    """

    SPLIT_NAME = {
        "train": ["train"],
        "valid": ["valid"],
        "test": ["test"]
    }

    def __init__(self, *args, **kwargs):
        self.data_path = kwargs["data_path"]
        self.data_version = kwargs["data_version"]
        self.truncation_length = kwargs["truncation_length"]
        self.filter_length = kwargs["filter_length"]
        self.split = kwargs["split"]
        self.py_logger = kwargs["py_logger"]

        self.PDB_DATA_DIR = kwargs["pdb_data_dir"]
        self.fast_dev_run = kwargs.get("fast_dev_run", False)
        self.data_name = kwargs["data_name"]
        self.seq_tokenizer = kwargs.get("seq_tokenizer", EsmSequenceTokenizer())

        self.use_continuous = kwargs.get("use_continuous", False)
        self.use_sequence = kwargs.get("use_sequence", False)
        self.target_field = kwargs.get("target_field", None)

        if kwargs.get("tokenizer", None) is not None:
            self.structure_pad_token_id = kwargs["tokenizer"].pad_token_id #-1
        else:
            self.structure_pad_token_id = -1

        self.debug = kwargs.get("debug", False)

        if self.split in ["train", "validation"]:
            # load pre-processed data
            target_split_file = os.path.join(self.data_path, self.split, f"processed_structured_{self.data_version}")
            processed_flag = os.path.exists(target_split_file)
            if not processed_flag:
                NotImplementedError
            else:
                self.data = torch.load(target_split_file, weights_only=False)
                if self.debug:
                    self.data = self.data[:100]

                self.py_logger.info(f"Loading from processed file {target_split_file},"
                                f"structured data of {len(self.data)} entries.")
                self._get_coords_from_pdb_chain()
        else:
            self.py_logger.info(f"Loading all test datasets")
            file_name = os.path.join(self.data_path, self.split, f"processed_structured_{self.data_version}")
            self.py_logger.info(f">>> path: {file_name}")
            assert os.path.exists(file_name), "Test data not preprocessed by ESM3 tokenizers, please run code for 'Code Usage Frequency'"
            self.data = torch.load(file_name, map_location="cpu", weights_only=False)
            self._get_coords_from_pdb_chain()

    
    def sanity_check(self):
        assert 0, "Not Needed"
    
    def _get_init_cnt_stats(self):
        return {
            "cnt_chain_fail": 0
        }

    def _get_coords_from_pdb_chain(self, ):

        # get ESM3's input for VQ-VAE's encoder
        new_data = []
        for i in tqdm(range(len(self.data))):
            pdb_chain = self.data[i]["pdb_chain"]
            coords, plddt, residue_index, chain_index = pdb_chain.to_structure_encoder_inputs()

            if len(coords[0]) > self.filter_length:
                coords = coords[:,:self.filter_length]
                plddt = plddt[:,:self.filter_length]
                residue_index = residue_index[:,:self.filter_length]
                chain_index = chain_index[:,:self.filter_length]
                pdb_chain = pdb_chain[:self.filter_length]
                self.data[i]["pdb_chain"] = pdb_chain
                # continue
            
            # filter out residues with nan for N, Ca and C coords
            sequence = pdb_chain.sequence
            chainbreak_boolean = torch.tensor([s == '|' for s in sequence])
            is_coord_nan = coords[0][:, :3, :].isnan().any(dim=-1).any(dim=-1) # [L, 3, 3] -> [L]
            invalid_boolean = torch.logical_and(~chainbreak_boolean, is_coord_nan)

            if invalid_boolean.any():
                indices = invalid_boolean.nonzero().squeeze(-1) #[0]
                if len(indices) > 5:
                    print(f"Skipping entry {pdb_chain.id} due to too many NaN coords: {len(indices)}")
                    continue
                    # raise ValueError
                else:
                    new_data.append(self.data[i])

                pdb_chain = pdb_chain[~is_coord_nan.numpy()]
                coords, plddt, residue_index, chain_index = pdb_chain.to_structure_encoder_inputs()
                new_data[-1]["pdb_chain"] = pdb_chain
            
            else:
                new_data.append(self.data[i])

            new_data[-1]["coords"] = coords[0] # [1, L, 37, 3] -> [L, 37, 3]
            new_data[-1]["plddt"] = plddt[0] # [1, L] -> [L]
            new_data[-1]["residue_index"] = residue_index[0] # [1, L] -> [L]
            new_data[-1]["chain_index"] = chain_index[0] # [1, L] -> [L]

            sequence = pdb_chain.sequence
            # Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/utils/encoding.py#L48
            if "_" in sequence:
                self.py_logger.info("Somehow character - is in protein sequence")
                raise ValueError

            if len(sequence) > self.filter_length:
                sequence = sequence[:self.filter_length]

            sequence = sequence.replace(C.MASK_STR_SHORT, "<mask>")
            seq_ids = self.seq_tokenizer.encode(sequence, add_special_tokens=False)
            seq_ids = torch.tensor(seq_ids, dtype=torch.int64)

            assert len(seq_ids) == len(coords[0])
            new_data[-1]["seq_ids"] = seq_ids # [L]

        self.py_logger.info(f"After pre-processing, get {len(new_data)} entries")

        self.data = new_data

    def collate_fn(self, batch):
        """passed to DataLoader as collate_fn argument"""
        if self.target_field is None:

            batch = list(filter(lambda x: x is not None, batch))

            coords, residue_index, chain_index, seq_ids, pdb_chain = tuple(zip(*batch))
            
            coords = util.pad_structures(coords, 
                            constant_value=torch.inf,
                            truncation_length=self.truncation_length)
                
            attention_mask = coords[:, :, 0, 0] == torch.inf
            residue_index = util.pad_structures(residue_index, constant_value=0,
                            truncation_length=self.truncation_length,
                            pad_length=coords.shape[1])

            # chain_ids
            chain_index = util.pad_structures(chain_index, constant_value=-2, # chain index는 0, 1, 2, ...로 시작함, -1은 chainbreak
                            truncation_length=self.truncation_length,
                            pad_length=coords.shape[1]
                        )

            assert C.SEQUENCE_PAD_TOKEN == 1
            seq_ids = util.pad_structures(seq_ids, constant_value=1, # pad_token_id not work anymore, Jan 14
                            truncation_length=self.truncation_length,
                            pad_length=coords.shape[1])
            
            return {
                "input_list": (coords, attention_mask, residue_index, chain_index, seq_ids, pdb_chain)
            }
    

    def load_all_structures(self, ):
        assert 0, "Not in use"

    def _get_item_residue_tokens(self, index):
        assert 0, "Not in use"
    
    def _get_item_structural_tokens(self, index):
        assert 0, "Not in use"
    
    def __getitem__(self, index: int):
        if self.target_field is None:
            item = self.data[index]
            coords, residue_index, chain_index, seq_ids, pdb_chain = item["coords"], item["residue_index"], item["chain_index"], item["seq_ids"], item["pdb_chain"]
            return coords, residue_index, chain_index, seq_ids, pdb_chain


    def cache_all_tokenized(self):
        # pre-checking
        for index in tqdm(range(len(self))):
            try:
                self[index]
            except:
                raise IndexError

    # def _get_item_structural_tokens(self, index, skip_check=False):
        
    #     item = self.data[index]
    #     pdb_chain = item["pdb_chain"]
    #     # if isinstance(self.tokenizer, WrappedOurComplexPretrainedTokenizer):

    #     if pdb_chain.id == '7yar':
    #         import pdb; pdb.set_trace()

    #     token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_chain, self.use_continuous) # torch.Tensors


    #     assert len(token_ids) == len(residue_index)
    #     token_ids = token_ids.detach()
    #     assert len(residue_index) == len(seqs)
        
    #     assigned_labels = item[self.target_field] #np.array(item[self.target_field])

    #     if assigned_labels.dim() == 1:
    #         selected_indices = torch.where(assigned_labels!=-1)
    #         token_ids = token_ids[selected_indices]
    #         assigned_labels = assigned_labels[selected_indices]
    #         seqs = np.array(seqs)[selected_indices].tolist()
    #         residue_index = residue_index[selected_indices]

    #     elif assigned_labels.dim() == 2:
    #         selected_indices = torch.where(assigned_labels.sum(dim=1)!=-assigned_labels.size(1))[0]
    #         token_ids = token_ids[selected_indices]
    #         # assigned_labels = assigned_labels[selected_indices]
    #         assigned_labels = assigned_labels[selected_indices[:, None], selected_indices[None, :]]            
    #         seqs = np.array(seqs)[selected_indices].tolist()
    #         residue_index = residue_index[selected_indices]
    #         # print(f"Contact labels after filtering: pos {torch.sum(assigned_labels==1)}, neg {torch.sum(assigned_labels==0)}")

    #     # filter chainbreak residues
    #     # valid_indices = np.where(residue_index!=-1)[0]
    #     valid_indices = np.where(residue_index>=0)[0]
    #     token_ids = token_ids[valid_indices]
    #     assigned_labels = assigned_labels[valid_indices] 
    #     seqs = np.array(seqs)[valid_indices].tolist()
    #     residue_index = residue_index[valid_indices]

    #     # cache the tokens
    #     self.data[index]["token_ids"] = token_ids.to("cpu").detach().clone()
    #     self.data[index][self.target_field] = assigned_labels
    #     self.data[index]["real_seqs"] = seqs
    #     self.data[index]["residue_index"] = residue_index #.to("cpu").detach().clone()

        return token_ids, assigned_labels, seqs, residue_index # torch.Tensor, List