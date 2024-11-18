import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import dok_matrix
import re
import os
import os.path
import time
import shutil
import concurrent.futures
import subprocess
import networkx as nx
import json
from io import StringIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio.Blast.Applications import NcbiblastpCommandline
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio.Align import MultipleSeqAlignment
from Bio.Align import AlignInfo
from Bio.SVDSuperimposer import SVDSuperimposer
from pathlib import Path

from pdbecif.mmcif_io import CifFileReader
#from Bio.SubsMat import MatrixInfo as matlist

#from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Polypeptide import three_to_one


import logging
from datetime import datetime
#from .utils import * # relatvie improt will cause issues when running from commandline
from pipeline.utils import *


# query_acc = '_QUERY' # taken from utils

class HomologyRing:
    def __init__(self, query_file, blast_DB, struct_source='AlphaFold', log_level = logging.INFO, uses_PDB_ids = False, interchain_edges = 'single'):
        self.name = os.path.basename(query_file).split('.')[0]
        self.query_file =  query_file
        self.msa = None
        self.blast_df = None
        self.contact_df = None
        self.hRIN_dict = None
        self.entity_contact_df = None
        self.EIDN_dict = None
        self.rot_tran_dict = None
        self.interchain_edges = interchain_edges

        self.query_acc = None

        with open('pipeline/config.json') as config_file:
            config = json.load(config_file)
        self.config = config
        self.cifDir = config['HOM_AFOLD_DIR'] #'cif/AlphaFold/'
        self.AFillDir = config['HOM_AFILL_DIR'] #'cif/AlphaFill/'
        self.pdbDir = config['HOM_PDB_DIR']#: "cif/PDB/"
        self.BLAST_DB_UNP_acc_idx = config['BLAST_DB_UNP_acc_idx']
        self.BLAST_DB_PDB_acc_idx = config['BLAST_DB_PDB_acc_idx']
        self.BLAST_DB_PDB_chain_idx = config['BLAST_DB_PDB_chain_idx']
        self.uses_PDB_ids = uses_PDB_ids
        self.built_from_list = None
        self.verbose = True

        self.blast_DB = blast_DB #'blast_db/uniprot_sprot.fasta' 
        self.struct_source = struct_source
        self.struct_dir = f'cif/{self.struct_source}'
        self.use_label_asym_id = False # ring output will give label chain identifiers instead of auth.
        self.ring_dir = f'ring_out/{self.struct_source}' #init in build

        self.contact_img_base = None
        self.contact_image_base_cmap = None
        self.prob_df = None
        self.G = None
        self.accs = []
        self.pos_dict = None

        self.q_atom_pos = None # used for planting the contactNet graphs. NOTE: tagged for deprication
        self.trimmed_pth = None # used by old version where only one chain of query was considered. This is nolonger the case and should be removed.

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_name = f"{self.name}_{timestamp}"
        self.logger = logging.getLogger(log_name)
        logging.basicConfig(filename=f'log/{log_name}.log',level=log_level, datefmt='%Y-%m-%d %H:%M:%S')
        self.logger.info(f"Object initialized. Query path: {self.query_file}")
        logging.getLogger('matplotlib').setLevel(logging.WARNING) # prevent matplotlib from logging a bunch of garbage
        init_file_structure(self.logger)

    def build(self, chain_id, state_id, max_results = 32, eval = 0.001, remote_BLAST=False, force_download = False, force_RING = False, all_models = False, use_label_asym_id = False, prob_normalization='strong'):
        """Create an hRIN for results of a BLAST search given a query chain

        Args:
            chain_id (str): asym(chain)_id of query chain e.g. 'C', 'AA'. Affected by use_label_asym_id.
            state_id (_type_): Not currently used default: use state 1. #TODO
            max_results (int, optional): _description_. Defaults to 32.
            eval (float, optional): _description_. Defaults to 0.001.
            remote_BLAST (bool, optional): _description_. Defaults to False.
            force_download (bool, optional): _description_. Defaults to False.
            force_RING (bool, optional): _description_. Defaults to False.
            use_label_asym_id (bool, optional): _description_. Defaults to False.
            prob_normalization (str, optional): _description_. Defaults to 'strong'.
        """
        self.logger.info(f"Bild initialized on file: {self.query_file}.")
        self.built_from_list = False
        self.use_label_asym_id = use_label_asym_id
        self.multi_state_ring = all_models
        self.asym_id_source = 'label' if use_label_asym_id else 'auth'
        if self.use_label_asym_id:
            self.ring_dir = f'{self.ring_dir}/label_asym_id'# = f'ring_out/{self.struct_source}' # RING will need to make different predictions for label and auth chain ids
        self.query_acc = query_acc
        self.q_chain, self.q_state = chain_id, state_id
        self.q_acc_chn = query_acc+'_'+self.q_chain
        # Auth is used here because that is what is returned from BLAST
        self.query_seq, _, _ = seq_from_cif(self.query_file, target_chain_id=chain_id, seq_id_source = 'auth', asym_id_source = self.asym_id_source, state_id=state_id)
        self.BLAST_is_remote = remote_BLAST
        self.blast_df = self._get_homologs(max_results=max_results, eval=eval)

        ### Look here if you run itno seq issues you thought you fixed before
        # if self.uses_PDB_ids: # set config depended values for useing the PDB blast_DB or the one for UNP
        #     blast_Qs = list(self.blast_df['acc_chn']) # + '_' + self.blast_df['pdb_chain']
        #     sep = self.config['BLAST_DB_PDB_delim']
        #     acc_idx = self.config['BLAST_DB_PDB_acc_idx']
        # else:
        #     blast_Qs = list(self.blast_df['acc'])
        #     sep = self.config['BLAST_DB_UNP_delim']
        #     acc_idx = self.config['BLAST_DB_UNP_acc_idx']
        # self.blast_df['full_seq'] = get_full_sequences(blast_Qs, self.blast_DB, acc_idx, sep=sep)


        # for few sequences, the uniprot sequence differs from the sequence in the AlphaFold structure.
        self.fetch_structures(source = self.struct_source, force = force_download) # this create self.file_df
        ### NOTE: auth_asym_id is not a mistake. Blast_DB list entries by auth_id, so entries in file.df will use this as well.
        self.authId_to_seqId_map = get_struct_seqs(self, seq_id_source='auth') # populate blast_df with the AA sequences taken directly from the mmCIF
        
        # self.fix_blast_alignment()
        # self._remove_AF_UNP_desc()
        
        self.run_ring(force=force_RING) # will remove sequences with empty blast output
        self.accs = [query_acc] + self.blast_df['acc'].tolist()
        if len(self.accs) <= 1:
            print(f"Insufficient number of homologs to perform alignment: {len(self.accs)}")
            return
        self.msa = self._build_msa()
        self.consensus_seq = dumber_consensus(self.msa) #str(AlignInfo.SummaryInfo(self.msa).dumb_consensus())
        
        self.seqId_to_alnIdx_maps = {rec.id:build_seqId_to_alnIdx_map(rec.seq) for rec in self.msa}
        
        self.contact_df, self.node_df = self.build_ring_dfs()
        self._interaction_types = list(self.contact_df['inter'].unique())
        self.atom_df = self.build_atom_df()
        self.create_hRIN(inter_class='all', set_attr=True, normalize=prob_normalization)
        return


    def parse_CIF_polyEnt(self, cfr, pth, gene_info = True):
        cif_obj = cfr.read(pth)
        acc, cif_data = list(cif_obj.items())[0]
        poly_ent = cif_data['_entity_poly']
        #_entity_src_gen
        poly_ent = {k: v if isinstance(v, list) else [v] for k,v in poly_ent.items()} # ensure dict values are lists for cast to df
        poly_ent['pdbx_seq_one_letter_code_can'] = [re.sub(r'[^A-Z]', '', seq) for seq in poly_ent['pdbx_seq_one_letter_code_can']] # remove non-alpha characters (newlines) from sequences
        poly_ent_df = pd.DataFrame(poly_ent)
        poly_ent_df['src_acc'] = acc
        poly_ent_df['struct_pth'] = pth

        if gene_info:
            # Join gene information for protein chains
            try:
                ent_src_gen = cif_data['_entity_src_gen']
                ent_src_gen = {k: v if isinstance(v, list) else [v] for k,v in ent_src_gen.items()} # ensure dict values are lists for cast to df
                ent_src_gen_df = pd.DataFrame(ent_src_gen)[['entity_id', 'pdbx_gene_src_gene']]
                poly_ent_df = pd.merge(poly_ent_df, ent_src_gen_df, on='entity_id', how='left')
            except:
                print(f'No Gene information for {pth}. Structure Skipped')
                return None # this probably wont work correctly
        return poly_ent_df

    def build_fromFamily(self, family, chain_id, state_id, force_RING = False, use_label_asym_id = False, prob_normalization='strong', all_models = False):
        self.multi_state_ring = all_models
        cfr = CifFileReader()
        poly_ent_dfs = []
        if isinstance(family, str): # family is a path
            if os.path.isdir(family): 
                #family points to a directory
                # expected to contain cif files
                for root, dirs, files in os.walk(family):
                    for file in files:
                        if file[-4:] != '.cif':
                            continue
                        pth = os.path.join(root,file)
                        poly_ent_df = self.parse_CIF_polyEnt(cfr, pth)
                        poly_ent_dfs.append(poly_ent_df)
                poly_ent_df = pd.concat(poly_ent_dfs)
                poly_ent_df = poly_ent_df[['src_acc', 'struct_pth'] + [col for col in poly_ent_df.columns if col not in ['src_acc', 'struct_pth']]] # reorder col
                poly_ent_df = poly_ent_df[poly_ent_df['type'] == 'polypeptide(L)'] # Filter just the protein chains

                ### FILTERING OF CHAINS 
                # Alt method of filtering chains
                # poly_ent_df.loc[poly_ent_df.groupby('src_acc')['pairiwise_identity'].idxmax()]
                poly_ent_df = poly_ent_df[poly_ent_df['pdbx_seq_one_letter_code_can'].str.contains('DEAD')] # REMOVE ME!
                poly_ent_df['pdbx_strand_id'] = poly_ent_df['pdbx_strand_id'].str.split(',')
                poly_ent_df = poly_ent_df.explode('pdbx_strand_id').reset_index(drop=True)
                poly_ent_df['acc_chn'] =  poly_ent_df['src_acc'] + '_' + poly_ent_df['pdbx_strand_id']
                self.file_df = poly_ent_df[['src_acc', 'acc_chn', 'pdbx_strand_id', 'struct_pth']]
                self.file_df = self.file_df.rename(columns={'src_acc':'acc', 'pdbx_strand_id':'pdb_chain'})
                self.blast_df = poly_ent_df
                self.blast_df = self.blast_df.rename(columns={'src_acc':'acc', 'pdbx_strand_id':'pdb_chain', 'pdbx_seq_one_letter_code_can':'struct_seq'})
            elif os.path.isfile(family): 
                family_df = pd.read_csv(family, sep='\t')
                if family_df['chain'].str.contains(',').any():
                    # Explode the dataframe if the chain columns has list of chains.
                    # This would indicate there is more than one query chain for a given file.
                    family_df['chain'] = family_df['chain'].str.split(',')
                    family_df = family_df.explode('chain')
                family_df['chain'] = family_df['chain'].str.strip()
                family_df['acc'] = family_df['PDB'] if self.uses_PDB_ids else family_df['UNP']
                family_df['acc_chn'] = family_df['acc'] + '_' + family_df['chain']
                family_df = family_df.rename(columns={'chain':'pdb_chain', 'path':'struct_pth'})
                self.file_df = family_df[['acc','acc_chn','pdb_chain', 'struct_pth']]
                self.blast_df = family_df[['acc','acc_chn','pdb_chain']]
                # processed_paths = []
                # poly_ent_dfs = []
                # for _, r in family_df.iterrows():
                #     pth = r['struct_path']
                #     if pth not in processed_paths:
                #         poly_ent_dfs.append()
                #         processed_paths.append(pth)
                #     else:
                #         continue
            else:
                raise Exception("Family must be a path to directory containing files to be included in family or a csv file with paths to CIF structures and the cain identifier to be included in the family.")
        else:
            raise Exception(f"Invalid path {family}.")

        self.logger.info(f"Bild initialized on file: {self.query_file}.")
        self.built_from_list = True
        self.use_label_asym_id = use_label_asym_id
        self.asym_id_source = 'label' if use_label_asym_id else 'auth'
        if self.use_label_asym_id:
            self.ring_dir = f'{self.ring_dir}/label_asym_id'# = f'ring_out/{self.struct_source}' # RING will need to make different predictions for label and auth chain ids
        self.q_chain, self.q_state = chain_id, state_id
        # cfr = CifFileReader()
        self.query_acc = list(cfr.read(self.query_file).keys())[0]
        self.q_acc_chn = self.query_acc+'_'+self.q_chain
        self.query_seq, _, _ = seq_from_cif(self.query_file, target_chain_id=chain_id, seq_id_source = 'auth', asym_id_source = self.asym_id_source, state_id=state_id)

        # See what chains in structures are homologs to the query
        # No longer used
        # q = self.query_seq
        # scores = []
        # for i, r in poly_ent_df.iterrows():
        #     s = r['pdbx_seq_one_letter_code_can']
        #     score = pairwise2.align.localxx(q, s)[0].score / min(len(q), len(s))
        #     scores.append(score)
        # poly_ent_df['pairiwise_identity'] = scores

        # DDX_sele = poly_ent_df['pdbx_gene_src_gene'].str.contains('DDX', na=False)
        # if self.verbose and DDX_sele.sum():
        #     print(f"Removed {(~DDX_sele).sum()} chains do not list DDX in gene name.")
        # poly_ent_df = poly_ent_df[DDX_sele]

        # self.blast_df['full_seq'] = get_full_sequences(blast_Qs, self.blast_DB, acc_idx, sep=sep)
        # # for few sequences, the uniprot sequence differs from the sequence in the AlphaFold structure.
        # self.fetch_structures(source = self.struct_source, force = force_download) # this create self.file_df
        

        # ### NOTE: auth_asym_id is not a mistake. Blast_DB list entries by auth_id, so entries in file.df will use this as well.
        self.authId_to_seqId_map = get_struct_seqs(self, seq_id_source='auth') # populate blast_df with the AA sequences taken directly from the mmCIF
        
        # self.fix_blast_alignment()
        # #self._remove_AF_UNP_desc()

        self.run_ring(force=force_RING) # will remove sequences with empty blast output
        self.accs = [query_acc] + self.blast_df['acc'].tolist()
        if len(self.accs) <= 1:
            print(f"Insufficient number of homologs to perform alignment: {len(self.accs)}")
            return
        self.msa = self._build_msa()
        self.consensus_seq = dumber_consensus(self.msa) #str(AlignInfo.SummaryInfo(self.msa).dumb_consensus())
        
        self.seqId_to_alnIdx_maps = {rec.id:build_seqId_to_alnIdx_map(rec.seq) for rec in self.msa}
        
        self.contact_df, self.node_df = self.build_ring_dfs()
        self._interaction_types = list(self.contact_df['inter'].unique())
        self.atom_df = self.build_atom_df()
        self.create_hRIN(inter_class='all', set_attr=True, normalize=prob_normalization)
        return #poly_ent_df
    
    def save_family(self, out):
        """Saves members of protein family to tsv. Can be used to rebuild the family with build_fromFamily

        Args:
            out (str): Path/name of the family tsv file to output

        Returns:
            pandas.DataFrame: Content of the Family tsv file.
        """
        out_df = self.file_df.copy()
        out_df = out_df[['acc','struct_pth','pdb_chain']]
        if self.uses_PDB_ids:
            out_df = out_df.rename(columns={'acc':'PDB'})
        else:
            out_df = out_df.rename(columns={'acc':'UNP'})
        out_df = out_df.rename(columns={'struct_pth':'path', 'pdb_chain':'chain'})
        out_df['struct_pth'] = out_df['struct_pth'].apply(lambda x: Path(x).resolve())
        out_df.to_csv(out, sep='\t')
        print(f"Saved family to {Path(out).resolve()}")
        return out_df

    # def buildFromPDBs(self, state_id, q_acc, q_chain, PDB_accs = None, PDB_pths = None, max_results = 32, eval = 0.001, force_download = False, force_RING = False, use_label_asym_id=False):
    #     """
    #     Performs analysis on list of user supplied PDBs instead of performing a homolgy search.
    #     Analysis requires a set of corresponding CHAINS accross proteins, so you spesify a chain of interest q_chain (auth_asym_id) in one protein q_acc that must be represented in 
    #     PDB_accs or PDB_pths.
    #     """
    #     self.built_from_list = True
    #     self.use_label_asym_id = use_label_asym_id
    #     self.asym_id_source = 'label' if use_label_asym_id else 'auth'
    #     if self.use_label_asym_id:
    #         self.ring_dir = f'{self.ring_dir}/label_asym_id'
    #     if not ((PDB_accs is None) ^ (PDB_pths is None)):
    #         raise Exception("You must either provide a list of PDB accs with PDB_accs or paths to mmCIF files with PDB_pths. PDB_pths may also be a string to a folder containing all queries.")
        
    #     self.logger.info(f"Bild initialized on file: {self.query_file}.")
    #     self.q_acc = q_acc
    #     self.q_chain, self.q_state = q_chain, state_id
    #     self.q_acc_chn = q_acc+'_'+self.q_chain
    #     #self.query_seq, _, _ = seq_from_cif(self.query_file, target_chain_id=q_chain, seq_id_source = 'auth', asym_id_source = 'auth', state_id=state_id)

    #     if PDB_accs is not None:
    #         # need to get the structures for the PDBs provided by the user.
    #         # When running on user supplied structures, this needs to be done before getting homologs
    #         self.fetch_structures(force=force_download, source=self.struct_source, on_accs=PDB_accs) # result: self.file_df is constructed
    #     elif PDB_pths is not None:
    #         # need to construct the file_df because it was not done by fetch_structures
    #         if isinstance(PDB_pths, str) and os.path.isdir(PDB_pths):
    #             # List comprehension to gather all file paths in the directory
    #             PDB_pths = [os.path.join(PDB_pths, f) for f in os.listdir(PDB_pths)]
    #         else:
    #             raise Exception("PDB_pths must be a string of a dir containing query mmCIF files or a list of paths to mmCIF files.")
    #         if type(PDB_pths) == list:
    #             file_dicts = []
    #             for cif_pth in PDB_pths:
    #                 f_basename = os.path.basename(cif_pth).split('.')[0]
    #                 if not os.path.isfile(cif_pth):
    #                     raise Exception(FileNotFoundError(cif_pth))
    #                 if not os.path.splitext(cif_pth)[1].lower():
    #                     raise Exception("Provided structure files must be .cif")
    #                 file_dicts.append({
    #                     'acc': f_basename,
    #                     'pdb_chain': None,
    #                     'acc_chn': None,
    #                     'struct_pth': cif_pth
    #                 })
    #             self.file_df = pd.DataFrame(file_dicts)
    #         else:
    #             raise Exception("PDB_pths must be a string of a dir containing query mmCIF files or a list of paths to mmCIF files.")


        # self.blast_df = self._get_homolog_chains(max_results=max_results, eval=eval)
        # # at this point file df has one entry per entity in mmCIF (unique at sequecne level)
        # # This can combine multiple chains to one entry.
        # # We want one entry per chain
        # self.file_df = pd.merge(self.file_df[['acc','struct_pth']], self.blast_df[['acc','pdb_chain','acc_chn']], on='acc', how='inner')[self.file_df.columns.tolist()]
        # self.authId_to_seqId_map = get_struct_seqs(self, seq_id_source='auth')
        # # Housekeeping logic for if acc represent PDB ids or uniprot Accs
        # # these are parsed differently from the blast database
        # if self.uses_PDB_ids: # set config depended values for useing the PDB blast_DB or the one for UNP
        #     blast_Qs = list(self.blast_df['acc_chn']) # + '_' + self.blast_df['pdb_chain']
        #     sep = self.config['BLAST_DB_PDB_delim']
        #     acc_idx = self.config['BLAST_DB_PDB_acc_idx']
        # else:
        #     blast_Qs = list(self.blast_df['acc'])
        #     sep = self.config['BLAST_DB_UNP_delim']
        #     acc_idx = self.config['BLAST_DB_UNP_acc_idx']
        # #self.blast_df['full_seq'] = get_full_sequences(blast_Qs, self.blast_DB, acc_idx, sep=sep)
        # # self.accs = [query_acc] + self.blast_df['acc'].tolist()
        # # for few sequences, the uniprot sequence differs from the sequence in the AlphaFold structure.
        # #self.fetch_structures(source = self.struct_source, force = force_download)
        # #self.authId_to_seqId_map = get_struct_seqs(self, seq_id_source='auth') #seq_id_source previously 'label' # populate blast_df with the AA sequences taken directly from the mmCIF
        # #self.fix_blast_alignment()
        # #self._remove_AF_UNP_desc()

        # # 
        # self.run_ring(force=force_RING) # will remove sequences with empty blast output
        # self.msa = self._build_msa()
        # self.accs = [query_acc] + self.blast_df['acc'].tolist() # should remove refrence to this to make more robust. calculate with [r.id for r in self.msa]
        # self.seqId_to_alnIdx_maps = {rec.id:build_seqId_to_alnIdx_map(rec.seq) for rec in self.msa}
        # #build_index_maps(self)
        
        # self.contact_df, self.node_df = self.build_ring_dfs()
        # self._interaction_types = list(self.contact_df['inter'].unique())
        # self.atom_df = self.build_atom_df()

    # def _get_homologs(self, max_results = 32, eval = 0.001, remote=False):
    #     self.logger.info(f"Retreiving homologs from blast DB: {self.blast_DB}")
    #     self.logger.info('Running BLAST...')
    #     # write query sequence as fasta to be used by BLAST commandline as input
    #     with open('query_ring.fasta', 'w') as q:
    #         q.write(f">query\n{self.query_seq}")
        
    #     blast_header = 'sseqid qstart qend sstart send evalue gaps sseq qseq'
    #     blast_input_f = 'query_ring.fasta'
    #     if self.remote_db is not None:
    #         db = self.remote_db
    #         if db == 'pdb':
    #             self.uses_PDB_ids = True
    #         blast_client = NcbiblastpCommandline(query=blast_input_f, db=db, outfmt=f"6 {blast_header}", evalue=eval, remote=True)
    #     else:
    #         blast_client = NcbiblastpCommandline(query=blast_input_f, db=self.blast_DB, outfmt=f"6 {blast_header}", evalue=eval, remote=False)
    #     stdout, stderr = blast_client() 

    #     blast_df = pd.read_csv(StringIO(stdout), sep='\t', names=blast_header.split()).sort_values('evalue').reset_index(drop=True)#.head(max_results)
    #     blast_df['overlap_length'] = blast_df['qend'] - blast_df['qstart'] - blast_df['gaps']
    #     if self.uses_PDB_ids:
    #         id = blast_df['sseqid']
    #         if '|' in id.iloc[0]: # this is just about the worst way of doing this
    #             pdb_acc = id.str.split('|').str[1]#.str.lower()
    #             pdb_chain = id.str.split('|').str[2]#.str.upper()
    #         else:
    #             pdb_acc = id.str.split('_').str[0]
    #             pdb_chain = id.str.split('_').str[1]
    #         blast_df['pdb_chain'] = pdb_chain
    #         blast_df['acc'] = pdb_acc
    #     else:
    #         blast_df['acc'] = blast_df['sseqid'].str.split('|').str[1]
    #         blast_df['pdb_chain'] = 'A' # all alphafold structures will be on chain A
    #     blast_df['acc_chn'] = blast_df['acc'] + '_' + blast_df['pdb_chain']#pdb_acc + '_' + pdb_chain
    #     df_len = len(blast_df)
    #     self.logger.info(f'BLAST returns {df_len} sequences.')
    #     # remove duplicate enteries (common in PDB)
    #     blast_df = blast_df.drop_duplicates(['acc','pdb_chain'])
    #     if df_len - len(blast_df):
    #         self.logger.warning(f"{df_len - len(blast_df)} map to the same entry. Removing duplicates and keeping best E-value...")
    #     # Remove sequences with missing residues / non-standard AAs. These usually don't have structures in alphafold.
    #     # df_len = len(blast_df)
    #     # blast_df = blast_df[~blast_df['sseq'].apply(contains_non_standard_aa)]
    #     # if df_len - len(blast_df):
    #     #     self.logger.warning(f"{df_len - len(blast_df)} Contain non-standard residues. Removing...")
    #     # take only the top max_values
    #     df_len = len(blast_df)
    #     blast_df = blast_df.head(max_results).reset_index()
    #     if df_len - len(blast_df):
    #          self.logger.info(f"Taking the top {max_results} enteries.")
        
    #     return blast_df

    def _get_homologs(self, max_results = 32, eval = 0.001):
        self.logger.info(f"Retreiving homologs from blast DB: {self.blast_DB}")
        self.logger.info('Running BLAST...')
        # write query sequence as fasta to be used by BLAST commandline as input
        with open('query_ring.fasta', 'w') as q:
            q.write(f">query\n{self.query_seq}")
        
        blast_input_f = 'query_ring.fasta'
        if (self.blast_DB not in ['pdb', 'sprot']) and self.BLAST_is_remote:
            raise Exception("Supported remote Databases are 'sprot' and 'pdb'")
        
        blast_client = NcbiblastpCommandline(query=blast_input_f, db=self.blast_DB, outfmt=15, evalue=eval, remote=self.BLAST_is_remote)
        stdout, stderr = blast_client() 
        blast_json = json.loads(stdout)

        hits = blast_json['BlastOutput2'][0]['report']['results']['search']['hits']

        # check the first result to see if acc formatted like PDB entries or UNP
        # Needed because BLAST does not use a consistent/useful schema for different databases
        test_acc = hits[0]['description'][0]['accession']
        test_acc = test_acc.split('_')
        test_title = hits[0]['description'][0]['title']
        # check if acc are PDB or uniprot
        # This is not a robust way of doing things
        if (len(test_acc) == 2) and (len(test_acc[0]) == 4):
            acc_fmt = 'pdb'
        elif test_title.split('|')[0] == 'sp':
            acc_fmt = 'unp'
        else:
            acc_fmt = None
            raise Exception("Unknown / unsupported BLAST output scheme.")

        hit_dfs = []
        for hit in hits:
            desc_df = pd.json_normalize(hit['description'])
            hsps_df = pd.json_normalize(hit['hsps'])
            if len(hsps_df) > 1:
                raise Exception("Multiple alignments for a hit!")
            hsps_df = pd.concat([hsps_df] * len(desc_df), ignore_index=True)
            hit_df = pd.concat([desc_df, hsps_df], axis=1)
            hit_dfs.append(hit_df)
        blast_df = pd.concat(hit_dfs, ignore_index=True)
        if acc_fmt == 'pdb':
            blast_df = blast_df.rename(
                columns={
                    'id': 'sseqid',
                    'accession': 'acc_chn', 
                    'query_from': 'qstart', 
                    'query_to': 'qend', 
                    'hit_from': 'sstart', 
                    'hit_to': 'send'}
            )
            blast_df['acc'] = blast_df['acc_chn'].str.split('_').str[0]
            blast_df['pdb_chain'] = blast_df['acc_chn'].str.split('_').str[1]
        elif acc_fmt == 'unp':
            blast_df['acc'] = blast_df['title'].str.split('|').str[1]
            # AF structures will all only have chain A
            # dummy cols so errors aren't thrown elsewhere.
            blast_df['pdb_chain'] = 'A'
            blast_df['acc_chn'] = blast_df['acc'] + '_A'
            blast_df = blast_df.rename(
                columns={
                    'id': 'sseqid', 
                    'query_from': 'qstart', 
                    'query_to': 'qend', 
                    'hit_from': 'sstart', 
                    'hit_to': 'send'}
            )

        df_len = len(blast_df)
        self.logger.info(f'BLAST returns {df_len} sequences.')
        # remove duplicate enteries (common in PDB)
        blast_df = blast_df.drop_duplicates(['acc','pdb_chain'])
        if df_len - len(blast_df):
            self.logger.warning(f"{df_len - len(blast_df)} map to the same entry. Removing duplicates and keeping best E-value...")
        # Remove sequences with missing residues / non-standard AAs. These usually don't have structures in alphafold.
        # df_len = len(blast_df)
        # blast_df = blast_df[~blast_df['sseq'].apply(contains_non_standard_aa)]
        # if df_len - len(blast_df):
        #     self.logger.warning(f"{df_len - len(blast_df)} Contain non-standard residues. Removing...")
        # take only the top max_values
        df_len = len(blast_df)
        blast_df = blast_df.head(max_results).reset_index()
        if df_len - len(blast_df):
             self.logger.info(f"Taking the top {max_results} enteries.")
        
        return blast_df
    
    def get_all_file_chain_seq(self):
        """
        Used when providing list of PDB accs/mmCIF files
        """
        cfr = CifFileReader()
        rows = []
        ex_debug = None
        for _,r in self.file_df.iterrows():
            acc = r['acc']
            struct_pth = r['struct_pth']
            cif_obj = cfr.read(struct_pth)
            row = list(cif_obj.values())[0]['_entity_poly']
            if type(row['entity_id']) == list:
                for row_m in [dict(zip(row, t)) for t in zip(*row.values())]:
                    row_m['acc'] = acc
                    rows.append(row_m)
            else:
                row['acc'] = acc
                rows.append(row)
        out_df = pd.DataFrame(rows)
        out_df['pdbx_seq_one_letter_code'] = out_df['pdbx_seq_one_letter_code'].str.replace(r'\s+', '', regex=True)
        out_df['pdbx_seq_one_letter_code_can'] = out_df['pdbx_seq_one_letter_code_can'].str.replace(r'\s+', '', regex=True)
        return out_df
    
    def _get_homolog_chains(self, max_results = 32, eval = 0.001):
        # Modification of the _get_homologs method for user supplied PDBs
        # Performs 1-vs-all pairwise alignemnt of the query against all chains in the provided list of PDBs instead of BLAST search.
        self.logger.info("Processing CIFs for chains homologous to query.")
        
        # parse the provided CIF files to get the PP sequence for each chain.
        out_df = self.get_all_file_chain_seq()
        out_df = out_df[out_df['type'] == 'polypeptide(L)'] # filter just the polypeptide chains. #Lookout: don't know what the (L) means
        
        # find the query sequence in the df of chains and their sequences.
        chain_re = f'(^|,){self.q_chain}(,|$)' # Used for finding the records with the correct chain.
        query_seq = out_df[(out_df['acc'] == self.q_acc) & (out_df['pdbx_strand_id'].str.contains(chain_re, regex=True))]
        if len(query_seq) == 0:
                raise Exception("No match for file_chain {self.q_acc}_{self.q_acc}. Ensure auth_asym_id is used for chain id.")
        query_seq = query_seq.iloc[0]['pdbx_seq_one_letter_code_can']
        self.query_seq = query_seq
        
        # Pairwise align
        #aligner = Align.PairwiseAligner()
        #aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        rows = []
        for i, r in out_df.iterrows():
                # s = subject
                s_acc = r['acc']
                s_chains = r['pdbx_strand_id'].split(',')
                s_seq = r['pdbx_seq_one_letter_code_can']
                #aligner = Align.PairwiseAligner(scoring="blastp")
                
                alignments = pairwise2.align.localxx(self.query_seq, s_seq)#, sub_mat)
                #alignments = aligner.align(self.query_seq, s_seq)
                best_aln = max(alignments, key=lambda x: x.score)
                for chain in s_chains:
                        rows.append({
                                'acc' : s_acc,
                                'pdb_chain': chain,
                                'acc_chn' : f'{s_acc}_{chain}',
                                'sstart': best_aln.start,
                                'send': best_aln.end,
                                'struct_seq_parse' : s_seq,
                                'score': best_aln.score
                        })
        return pd.DataFrame(rows)

    def fix_blast_alignment(self):
        """
        DEPRICATED, no need for pairwise alignment. Keeping becauese can be used for discarding bad sequences later
        """
        self.blast_df[['qseq_x', 'sseq_x', 'sstart_x', 'send_x']] = None, None, None, None
        out = []
        for i, r in self.blast_df.iterrows():
            q = r['qseq'.replace('-','')]
            s_full = r['full_seq']
            s_struct = r['struct_seq']
            alns = pairwise2.align.localxx(q, s_struct, penalize_end_gaps = False)
            # pairwise aligner will return a list of alignments with equal cost
            # pick an alignment that minimizes gaps in the query sequence.
            min_q_gaps = 100000000
            aln = None
            for a in alns:
                #print(len(alns), r['pdb_id'], a)
                gaps = a.seqA.count('-')
                if gaps < min_q_gaps:
                    min_q_gaps = gaps
                    aln = a
                out.append(aln)
            #print(r['pdb_id'])
            #print(format_alignment(*aln))
            #print(r['sstart'], r['send'], aln.start, aln.end)
            self.blast_df.loc[i, ['qseq_x', 'sseq_x', 'sstart_x', 'send_x']] = [aln.seqA, aln.seqB, int(aln.start), int(aln.end)]
        return out

    def _build_msa(self):
        self.logger.info('Building MSA with ClustalO...')

        # save sequences in blast_df to FASTA. Used as input to clustal.
        seq_records = [SeqRecord(Seq(self.query_seq), id=self.q_acc_chn, description='') ]
        for i, r in self.blast_df.iterrows():
            seq_record = SeqRecord(Seq(r['struct_seq']), id=r['acc_chn'], description='') 
            seq_records.append(seq_record)
        with open('blast_results.fasta', "w") as output_handle:
            SeqIO.write(seq_records, output_handle, "fasta")
        #MSA for visualization
        clustalOmega_cline = ClustalOmegaCommandline(infile='blast_results.fasta')
        stdout, stderr = clustalOmega_cline()
        with open("msa.fasta", 'w') as f:
            f.write(stdout)
        seq_records = []
        # Fix clustalo output splitting sequences over multiple lines
        for r in str.split(stdout,'>')[1:]:
            l = str.split(r)
            acc = l[0]
            seq_records.append(SeqRecord(Seq(''.join(l[1:])), id=acc))
        msa = MultipleSeqAlignment(seq_records)
        self.logger.info("MSA created.")
        query_gaps = str(msa[0].seq).count('-')
        if query_gaps:
            self.logger.warning(f"MSA opens {query_gaps} gaps in query.")
        return msa
    
    # Get the SeqRecord obj from MSA given uniprot acc
    def get_seqrecord_by_id(self, seq_id):
        return next((record for record in self.msa if record.id == seq_id), None)
    
    def _build_pos_dict(self):
        """
        Used by InteractionSummary()
        """
        self.pos_dict = {}
        for acc in self.accs:
            if acc == query_acc:
                dir = self.query_file
            else:
                dir = f'cif/trimmed/{acc}.cif'
            try:
                self.pos_dict[acc] = self.get_pos(dir, atoms='CA')
            except:
                print("Requires Trimmed CIF files to be created. Run BuildEnsemble() first.")
                break

    def fetch_structures(self, force=False, source='AlphaFold', on_accs = None):
        """Fetch structures based on the specified source with optional force redownload."""
        status_dict = {}
        cols = ['acc', 'pdb_chain', 'acc_chn']
        if on_accs == None: # if this is none, it assumes that there is already a list of homologs in self.blast_df we get structures for.
            self.file_df = self.blast_df[cols].copy()
            self.file_df['struct_pth'] = None
            # To prevent ring from overwriting the output from the query file because a blast result has the same acc/file name, create a renamed copy.
            new_query_file = shutil.copy(self.query_file, f"{self.struct_dir}/query.cif")
            # record the query file, and add enteries for the homologs returned by blast. 
            ### NOTE: The asym_ids in the blast DB files are all auth ids! as a result, enteries in this df will be auth_asym_ids
            self.file_df = pd.concat([pd.DataFrame([{'acc':query_acc, 'pdb_chain':self.q_chain,'acc_chn': query_acc+'_'+self.q_chain, 'struct_pth':new_query_file}]), self.file_df], ignore_index=True)
            accs_to_fetch = list(self.file_df['acc'])[1:] # don't get the structure for the query, use the one provided
        else:
            accs_to_fetch = on_accs
            self.file_df = pd.DataFrame({'acc':accs_to_fetch})
            # these will be written later when the structure files are parsed to get the PP sequence of each chain
            self.file_df['pdb_chain'] = None 
            self.file_df['acc_chn'] = None
            self.file_df['struct_pth'] = None
        if source == 'AlphaFold':
            self.logger.info("Collecting AlphaFold structures...")
            status_dict = get_alphafold(self, accs_to_fetch, force=force)
        elif source == 'AlphaFill':
            self.logger.info("Collecting AlphaFill structures...")
            status_dict = get_alphafill(self, accs_to_fetch, force=force, out_dir=self.AFillDir)

        elif source == 'PDB':
            self.logger.info("Downloading structures from the PDB...")
            status_dict = get_pdb(self, accs_to_fetch, force=force)

        failed_accs = [acc for acc, status in status_dict.items() if status not in  ['Success', 'Exists']]
        if failed_accs:
            self.logger.warning(f"Failed to download Structures for {failed_accs}. Removing...")
            for acc in failed_accs:
                filter_sequences(self, acc) # removes from blast_df and file_df
            #self.accs = [acc for acc in self.accs if acc not in failed_accs]

        return status_dict
    
    def _start_ring_p(self, input_file, acc, force = False):
        out_name = os.path.basename(input_file).split('.')[0] # this should be the PDB/UNP acc but default ring behaivor is to take the basename of the input file
        if acc == query_acc: # query_acc defined in utils
            force = True # always run for the query, since it is different
        out_edges_pth = f'{self.ring_dir}/{out_name}.cif_ringEdges'
        out_nodes_pth = f'{self.ring_dir}/{out_name}.cif_ringNodes'
        if (os.path.exists(out_edges_pth) and os.path.exists(out_nodes_pth)) and (force == False): #node file, edge file exist. if is query file should re-run
            out = {"acc": acc, "edge_pth": out_edges_pth, "node_pth": out_nodes_pth, "error": None}
            return out
        else:
            self.logger.debug(f"running ring on {input_file} -? {self.ring_dir}")
            command = ['ring', '-i', input_file, '-v', '--out_dir', self.ring_dir]
            if self.use_label_asym_id:
                command.append('--prefer_label_asym_id')
            if self.multi_state_ring:
                command.append('--all_models')
            with open(os.devnull, 'w') as devnull: # dont't care about the output steams
                process = subprocess.Popen(command, stdout=devnull, stderr=devnull)#, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                process.wait()
            #stdout, stderr = process.communicate() # wait for ring to complete before releasing thread
            stdout, stderr = None, None#process.stdout.read(), process.stderr.read()
            # stdout = stdout.decode('utf-8')
            # stderr = stderr.decode('utf-8')  
            out = {"acc": acc, "edge_pth": out_edges_pth, "node_pth": out_nodes_pth, "error": stderr, 'stdout': stdout}
        return out
    
    def run_ring(self, force = False):
        max_threads = os.cpu_count() - 1 or 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = []
            for i,r in self.file_df.drop_duplicates('acc').iterrows():
                acc = r['acc'] #DIFFERECE BETWEEN PDB AND UNP
                cif_file = r['struct_pth']
                future = executor.submit(self._start_ring_p, cif_file, acc, force=force)
                futures.append(future)
            results = []
            time.sleep(1) # prevent race condition where OS has not reistered ring output files yet
            for future in futures:
                result = future.result()
                results.append(result)
        results = pd.DataFrame(results)
        for i,r in results.iterrows():
            with open(r['edge_pth'], 'r') as edge_f, open(r['node_pth'], 'r') as node_f:
                edge_lines = 0
                node_lines = 0
                
                for edge_line, node_line in zip(edge_f, node_f):
                    edge_lines += 1
                    node_lines += 1
                    
                    if edge_lines >= 3 or node_lines >= 3:
                        break
                if (edge_lines < 3) or (node_lines < 3):
                    #self.logger.debug()
                    filter_sequences(self, r['acc'], reason="Empty RING output - Possibly from malformed CIF.")
        self.file_df = self.file_df.merge(results, on='acc', how='left')

    
    def build_atom_df(self):
        # Build a DataFrame containing atom info for ALL structures
        dfs = []
        for i, r in self.file_df.iterrows():
            acc = r['acc']
            chain = r['pdb_chain']
            acc_chn = r['acc_chn'] #query_acc if (acc == query_acc) else r['acc_chn']
            pth = r['struct_pth']
            # TODO: set filter_chain according to if using AlphaFill
            cif_df = process_AFill_cif(acc, pth, chain, self.q_state, filter_chain=True)
            cif_df['acc_chn'] = acc_chn
            dfs.append(cif_df)
        atom_df = pd.concat(dfs, ignore_index=True)
              # Map the seq_id assigned to residues to columns in the MSA
        #col_blast_idx = ['.'] * len(atom_df)
        col_msa_idx = [pd.NA] * len(atom_df)
        last_acc_chn = None
        for i,r in atom_df.iterrows():
            acc = r['acc']
            acc_chn = r['acc_chn']
            if acc_chn != last_acc_chn:
                last_acc_chn = acc_chn
                auth_to_seq_map = self.authId_to_seqId_map[acc_chn] # preprocessed to prevent needing to parse the CIF here
                msa_seq_rec = self.get_seqrecord_by_id(acc_chn) # processed for '_QUERY' special case above
                seq_to_aln_map = build_seqId_to_alnIdx_map(msa_seq_rec)
            auth_seq_id = r['auth_seq_id']
            if r['group_PDB'] == 'ATOM':
                ### CORE LOGIC OF INDEX MAPPING ###
                auth_seq_id = int(auth_seq_id)
                try:
                    contig_seq_id = auth_to_seq_map[auth_seq_id]
                    msa_idx = seq_to_aln_map[contig_seq_id]
                except:
                    print(r)
                    raise Exception(IndexError)
                col_msa_idx[i] = msa_idx
            else:
                continue
        atom_df['msa_col'] = col_msa_idx
        return atom_df    
    
    def build_ring_dfs(self, drop_duplicates = True):
        edge_dfs = []
        node_dfs = []
        # there can be many results that are different chains of the same cif.
        processed_accs = [] 
        for i, r in self.file_df.iterrows():
            acc = r['acc']
            acc_chn = r['acc_chn']
            chain = r['pdb_chain']
            node_pth = r['node_pth']
            edge_pth = r['edge_pth']
            
            # cols of interest to take from the RING output files
            edge_header = ['NodeId1','Interaction','NodeId2', 'Distance', 'Model']

            node_header = ['NodeId', 'Position']

            try:
                edge_cur_df = pd.read_csv(edge_pth, sep='\t', usecols=edge_header)
                node_cur_df = pd.read_csv(node_pth, sep='\t', usecols=node_header)
            except:
                raise Exception(f"Malformed RING output {acc_chn}, {edge_pth}, {node_pth}")

            edge_cur_df['acc'] = acc
            node_cur_df['acc'] = acc
            # acc_chn is used as an identifier for which entry the rows correspond to.
            # in some cases, like ligands, atom chains may differ to what is indicated here.
            edge_cur_df['acc_chn'] = acc_chn
            node_cur_df['acc_chn'] = acc_chn
            
            edge_cur_df['chain1'] = edge_cur_df['NodeId1'].str.split(':').str[0].astype(str)
            edge_cur_df['chain2'] = edge_cur_df['NodeId2'].str.split(':').str[0].astype(str)
            node_cur_df['chain'] = node_cur_df['NodeId'].str.split(':').str[0].astype(str)

            edge_cur_df['struct_id1'] = edge_cur_df['NodeId1'].str.split(':').str[1].astype(int)
            edge_cur_df['struct_id2'] = edge_cur_df['NodeId2'].str.split(':').str[1].astype(int)

            edge_cur_df['inter'] = edge_cur_df['Interaction'].str.split(':').str[0]
            edge_cur_df['inter_orientation'] = edge_cur_df['Interaction'].str.split(':').str[1]

            edge_cur_df['msaCol1'], edge_cur_df['msaCol2'] = pd.NA, pd.NA

            # index map from indicies in continuous AA sequence to Gapped sequence provided by MSA
            # Functionally maps indicies from the structure sequence to MSA cols
            # requires that input sequence be trimmed to only have the same residue in the gapped sequence
            # from the MSA.
            auth_to_seq_map = self.authId_to_seqId_map[acc_chn]
            msa_seq_rec = self.get_seqrecord_by_id(acc_chn)
            seq_to_aln_map = build_seqId_to_alnIdx_map(msa_seq_rec)
            #compose the two maps: auth_seq_id -> contig_seq_id and contig_seq_id -> aln_seq_idx becomes auth_seq_id -> aln_seq_idx
            auth_to_aln_map = {auth_seq_id: seq_to_aln_map[contig_seq_id] for auth_seq_id, contig_seq_id in auth_to_seq_map.items() if contig_seq_id in seq_to_aln_map}
            # Filter edges so that atleast one node belongs to target auth_asym_id
            # To filter LIG_LIG interactions, it is not sufficient to ensure that cain1 or chain2 is target auth_asym_id. peptides can share the same auth_asym_id with ligands
            try:
                edge_sele1 = (edge_cur_df['chain1'] == chain) & (edge_cur_df['inter_orientation'].str.split('_')[0] != 'LIG')
                edge_sele2 = (edge_cur_df['chain2'] == chain) & (edge_cur_df['inter_orientation'].str.split('_')[1] != 'LIG')
            except:
                print(acc_chn, edge_pth)
                raise(Exception("something wrong with edge file"))
            # all residues in target chain should sucessfully map to an MSA col. Just need to make sure that the other node is not a ligand, which still has an auth_seq_id
            edge_cur_df.loc[edge_sele1, 'msaCol1'] = edge_cur_df.loc[edge_sele1, 'struct_id1'].map(auth_to_aln_map).astype('Int64')
            ##edge_sele = edge_sele & ~edge_cur_df['inter_orientation'].str.contains('LIG')
            edge_cur_df.loc[edge_sele2, 'msaCol2'] = edge_cur_df.loc[edge_sele2, 'struct_id2'].map(auth_to_aln_map).astype('Int64')
            # optionally filter edge_cur_df to remvoe irrelevant contacts
            edge_cur_df = edge_cur_df.loc[edge_sele1 | edge_sele2]
            node_cur_df

            # just reordering the cols
            #edge_cur_df = edge_cur_df[['acc'] + edge_header + ['inter', 'inter_orientation', 'chain1', 'chain2', 'struct_id1', 'struct_id2', 'msaCol1', 'msaCol2']]
            #node_cur_df = node_cur_df[['acc','chain'] + node_header]
            edge_dfs.append(edge_cur_df)
            node_dfs.append(node_cur_df)        

        edge_df = pd.concat(edge_dfs, ignore_index=True)
        node_df = pd.concat(node_dfs, ignore_index=True)
        edge_LIG_LIG = edge_df['inter_orientation'] == 'LIG_LIG'
        num_LIG_LIG = sum(edge_LIG_LIG)
        if num_LIG_LIG > 0:
            self.logger.warning(f"Filtering {num_LIG_LIG} Ligand-Ligand interactions from contact_df.")
            edge_df = edge_df[~edge_LIG_LIG]
        if drop_duplicates:
            self.logger.info(f"# Redundant conatacts(Res Level): {edge_df.duplicated().sum()}/{len(edge_df)}.")
            self.logger.info("Dropping duplicates...")
            # this is just to keep the contact with smallest distance, which requires sorting the df, but after the operation reorder to match the rows of the MSA
            id_order_map = {acc_chn : i for i, acc_chn in enumerate([r.id for r in self.msa])}
            edge_unique_features = ['acc_chn', 'struct_id1', 'struct_id2', 'inter']
            if self.multi_state_ring:
                edge_unique_features.append('Model')
            edge_df = edge_df.sort_values(by='Distance').drop_duplicates(subset=edge_unique_features, keep='first').sort_values(by='acc_chn', key = lambda x:x.map(id_order_map))
        edge_df = edge_df.reset_index(drop=True)

        # add col characterizing the participants of an interaction (intra-chain / inter-chain / ligand)
        edge_df['Node1_name'] = edge_df['NodeId1'].str.split(':').str[-1]
        edge_df['Node2_name'] = edge_df['NodeId2'].str.split(':').str[-1]
        # this logic is slow, but needed because there are some PI interactions with ligands that are labeled as sidechains (bug)
        res1_is_LIG = ~edge_df['Node1_name'].isin(standard_AA_three)
        res2_is_LIG = ~edge_df['Node2_name'].isin(standard_AA_three)
        # edge_df['res1_is_LIG'] = res1_is_LIG
        # edge_df['res2_is_LIG'] = res2_is_LIG
        is_with_ligand = (res1_is_LIG | res2_is_LIG)#edge_df['inter_orientation'].str.contains('LIG')
        edge_df['inter_class'] = pd.NA
        edge_df.loc[is_with_ligand,'inter_class'] = 'LIG'
        subject_chain = edge_df['acc_chn'].str.split('_').str[-1]
        edge_df.loc[(edge_df['chain1'] == subject_chain) & (edge_df['chain2'] == subject_chain) & (~is_with_ligand), 'inter_class'] = 'intra'
        edge_df.loc[((edge_df['chain1'] == subject_chain) ^ (edge_df['chain2'] == subject_chain)) & (~is_with_ligand), 'inter_class'] = 'inter'
        edge_df['res1_one'] = pd.NA
        edge_df['res2_one'] = pd.NA
        # res1_is_LIG = edge_df['inter_orientation'].str.split('_').str[0] == 'LIG'
        # res2_is_LIG = edge_df['inter_orientation'].str.split('_').str[1] == 'LIG'
        edge_df.loc[~res1_is_LIG, 'res1_one'] = edge_df.loc[~res1_is_LIG, 'Node1_name'].apply(three_to_one)
        edge_df.loc[~res2_is_LIG, 'res2_one'] = edge_df.loc[~res2_is_LIG, 'Node2_name'].apply(three_to_one)
        return edge_df, node_df
    

    def build_entity_contact_df(self, return_grouped=False, id_nodes_with_acc=True):
        """Enriches the conact_df which stores edge (contact) information for all homologs with information about the entities that each node belongs to.


        Args:
            return_grouped (bool, optional): _description_. Defaults to False.

        Returns:
            Pandas.DataFrame: _description_
        """
        if self.rot_tran_dict is None:
            build_transforms(self) # get rot trans mats for each homolog chain
        chain_identifer = 'label_asym_id' if self.use_label_asym_id else 'auth_asym_id'
        # Parse the _atom_site section of the CIF file to get a map between ['label_entity_id', 'label_asym_id', 'auth_asym_id'] for eitities in structures.
        cfr = CifFileReader()
        dfs = []
        # collect the entities present in each CIF file
        for _,r in self.file_df.drop_duplicates('acc').iterrows():
            ### OVERVIEW
            # for each homolog cif file
            # use _atom_site section to define relation between entity_id and both label_asym_id and auth_asym_id
            # check the _entity section to get identifying information for each entity_id in the file (name)
            # check the _struct_ref section to see if polymer entities are mapped to DB entry (UNP/PDB)
            # goal is to create identification scheme for entities: use UNP/PDB assention if present, else the name (usually done for ligands)

            src_acc = r['acc'] # entity ids are spesific to cif files - record which file information came from
            cif_obj = cfr.read(r['struct_pth']) #read the cif file - path comes from file_df
            acc_chn = r['acc_chn']
            acc, cif_obj = next(iter(cif_obj.items())) # get first entry in cif. Check if this will ever not be the case
            # in practice, the best way to tell which entities are in which chains is just to parse the _atom_site information
            # some entities can be in multiple chains
            atom_df = pd.DataFrame(cif_obj['_atom_site'])
            # auth_asym_ids do not uniquely identify entities. CIF provieds unique auth_seq_id, which is also in ring output.
            # recording this value is useful for joining entity information to contact_df
            #atom_df['hetatm_auth_seq_id'] = atom_df.apply( lambda atom: (atom['auth_seq_id'] if atom['group_PDB'] == 'HETATM' else None), axis=1)
            atom_df['hetatm_auth_seq_id'] = np.where(atom_df['group_PDB'] == 'HETATM', atom_df['auth_seq_id'], None)
            atom_df[['Cartn_x', 'Cartn_y', 'Cartn_z']] = atom_df[['Cartn_x', 'Cartn_y', 'Cartn_z']].astype(float)
            # used to get position information of entities for plotting
            mean_coords = atom_df.groupby(['label_entity_id', 'label_asym_id', 'auth_asym_id'])[['Cartn_x', 'Cartn_y', 'Cartn_z']].mean().reset_index()
            if src_acc != query_acc:
                rot = self.rot_tran_dict[acc_chn]['rot']
                tran = self.rot_tran_dict[acc_chn]['tran']
                mean_coords[['Cartn_x', 'Cartn_y', 'Cartn_z']] = np.dot(mean_coords[['Cartn_x', 'Cartn_y', 'Cartn_z']].values, rot.T) + tran # align the entity so pos is relative to query
            unique_atom_df = atom_df[['label_entity_id', 'label_asym_id', 'auth_asym_id', 'hetatm_auth_seq_id']].drop_duplicates()
            # Merge the mean coordinates back with the non-duplicate DataFrame
            atom_df = pd.merge(unique_atom_df, mean_coords, on=['label_entity_id', 'label_asym_id', 'auth_asym_id'])

            entity_section = cif_obj['_entity']
            for key in entity_section:
                if not isinstance(entity_section[key], list):
                    entity_section[key] = [entity_section[key]]
            # Create the entity_df DataFrame
            entity_df = pd.DataFrame(entity_section)[['id', 'type', 'src_method', 'pdbx_description']]
            entity_df = atom_df.merge(entity_df, left_on='label_entity_id', right_on='id') # issue: right on was previously on 'id' which had NaN values
            # if _struct_ref section is not a #loop, return format is inconsistent. (dict of values instead of dict of lists)
            # force consistency for df creation.
            struct_ref = cif_obj['_struct_ref']
            for key in struct_ref:
                if not isinstance(struct_ref[key], list):
                    struct_ref[key] = [struct_ref[key]]
            # create a dataframe
            ref_df = pd.DataFrame(struct_ref)[['entity_id','db_name', 'db_code','pdbx_db_accession']]
            # ISSUE: an entity with multiple links to different DB enteries will cause entities to be overrepresented in this merge.
            # use drop duplicates later to resolve. Either entry should contain enough information.
            entity_df = entity_df.merge(ref_df, left_on='label_entity_id', right_on='entity_id', how='left') # left_on='label_entity_id'

            #_pdbx_entity_nonpoly
            non_poly_ent = cif_obj.get('_pdbx_entity_nonpoly')
            if non_poly_ent is not None:
                for key in non_poly_ent:
                    if not isinstance(non_poly_ent[key], list):
                        non_poly_ent[key] = [non_poly_ent[key]]
                # create a dataframe
                non_poly_ent_df = pd.DataFrame(non_poly_ent)
                entity_df = entity_df.merge(non_poly_ent_df, on = 'entity_id', how='left') # left_on='label_entity_id', right_on='entity_id'

            # # if _struct_ref section is not a #loop, return format is inconsistent. (dict of values instead of dict of lists)
            # # force consistency for df creation.
            # struct_ref = cif_obj['_struct_ref']
            # for key in struct_ref:
            #     if not isinstance(struct_ref[key], list):
            #         struct_ref[key] = [struct_ref[key]]
            # # create a dataframe
            # ref_df = pd.DataFrame(struct_ref)#[['entity_id','db_name','pdbx_db_accession']]
            # # ISSUE: an entity with multiple links to different DB enteries will cause entities to be overrepresented in this merge.
            # # use drop duplicates later to resolve. Either entry should contain enough information.
            # entity_df = entity_df.merge(ref_df, left_on='label_entity_id', right_on='entity_id', how='left')
            
            # #### Get information about the identifier of non-poly ligands (not strictly needed)
            # #_pdbx_nonpoly_scheme
            # try:
            #     nonpoly_section = cif_obj['_pdbx_nonpoly_scheme']
            #     for key in nonpoly_section:
            #         if not isinstance(nonpoly_section[key], list):
            #             nonpoly_section[key] = [nonpoly_section[key]]
            #     # Create the entity_df DataFrame
            #     nonpoly_df = pd.DataFrame(nonpoly_section)[['entity_id', 'asym_id', 'auth_seq_num', 'pdb_mon_id']]
            #     entity_df = entity_df.merge(nonpoly_df, left_on=['label_entity_id','label_asym_id'], right_on=['entity_id', 'asym_id']) # issue: right on was previously on 'id' which had NaN values
            # except: 
            #     entity_df['pdb_mon_id'] = entity_df['pdbx_description']

            entity_df['src_acc'] = src_acc
            dfs.append(entity_df)
        self.entity_df = pd.concat(dfs)
        # for hRINs with ligands or interchain contacts, we need a consistent system to identify entities that participate in contacts so they can be mapped to a node.
        # Use PDB/UNP acc if the entity has one, else use the name - typically done for ligands
        if id_nodes_with_acc:
            self.entity_df['identifier'] = self.entity_df.apply(lambda row: (row['pdbx_db_accession'] + (f'_{row[chain_identifer]}' if row['db_name'] == 'PDB' else '')) if pd.notna(row['pdbx_db_accession']) else row['pdbx_description'], axis=1) # you should not be using a lambda for this.
        else:
            self.entity_df['identifier'] = self.entity_df['db_code']
        self.EIDN_dict = {identifier:eidn for eidn, identifier in enumerate(self.entity_df['identifier'].unique())}
        # just reverse the map so you can get identifier (name) from an EIDN value
        identifier_dict = {value: key for key, value in self.EIDN_dict.items()}
        self.identifier_dict = identifier_dict
        msa_len = self.msa.get_alignment_length()
        # used for creating axis ticks to identify nodes/entities in hRIN probability plot
        self.node_dict = {i:f'MSA {i}' if i < msa_len else f"EIDN {i - msa_len}; {identifier_dict[i - msa_len]}" for i in range(msa_len + len(self.EIDN_dict))}
        # set the ENTITY ID NUMBER (EIDN) To be used to map entities to nodes.
        self.entity_df['EIDN'] = self.entity_df['identifier'].map(self.EIDN_dict)
        # save the mean pos of entities for plotting
        self.EIDN_pos_dict = self.entity_df.groupby('EIDN')[['Cartn_x', 'Cartn_y', 'Cartn_z']].mean().to_dict(orient='index')
        # ring can optionally use label_asym_ids. Appropiate identifier needs to be used for merge
        right_key = chain_identifer #'label_asym_id' if self.use_label_asym_id else 'auth_asym_id'
        entity1_df = self.entity_df[[
            'label_entity_id', 'label_asym_id', 'auth_asym_id', 'type',
            'src_method', 'pdbx_description', 'db_name', 'db_code',
            'pdbx_db_accession', 'entity_id', 'src_acc', 'identifier', 'EIDN',
            'hetatm_auth_seq_id'
        ]].rename(columns=lambda x: f'{x}_1') .drop_duplicates() # 'id_x' removed
        entity2_df = self.entity_df[[
            'label_entity_id', 'label_asym_id', 'auth_asym_id', 'type',
            'src_method', 'pdbx_description', 'db_name', 'db_code',
            'pdbx_db_accession', 'entity_id', 'src_acc', 'identifier', 'EIDN', 
            'hetatm_auth_seq_id'
        ]].rename(columns=lambda x: f'{x}_2').drop_duplicates()
        # enriched version of contact_df whith information aobut what chains (UNP / ligiand id info) are forming the contact
        if self.use_label_asym_id:
            # for label_asym_ids, entities get their own chain (confirm this is the case), so the join is easy
            entity_contact_df = self.contact_df.merge(entity1_df, how='left', left_on=['chain1', 'acc'], right_on=['label_asym_id_1', 'src_acc_1'])
            entity_contact_df = entity_contact_df.merge(entity2_df, how='left', left_on=['chain2', 'acc'], right_on=['label_asym_id_2', 'src_acc_2'])
        else:
            entity_contact_df = self.contact_df
            entity_contact_df['hetatm_auth_seq_id_1'] = entity_contact_df.apply( lambda contact: (str(contact['struct_id1']) if (contact['inter_class'] == 'LIG') & (pd.isna(contact['msaCol1'])) else None), axis=1)
            entity_contact_df['hetatm_auth_seq_id_2'] = entity_contact_df.apply( lambda contact: (str(contact['struct_id2']) if (contact['inter_class'] == 'LIG') & (pd.isna(contact['msaCol2'])) else None), axis=1)
            entity_contact_df = entity_contact_df.merge(entity1_df, how='left', left_on=['chain1', 'acc', 'hetatm_auth_seq_id_1'], right_on=['auth_asym_id_1', 'src_acc_1', 'hetatm_auth_seq_id_1'])
            entity_contact_df = entity_contact_df.merge(entity2_df, how='left', left_on=['chain2', 'acc', 'hetatm_auth_seq_id_2'], right_on=['auth_asym_id_2', 'src_acc_2', 'hetatm_auth_seq_id_2'])


        bad_edges_mask = entity_contact_df['src_acc_1'].isna() | entity_contact_df['src_acc_2'].isna()
        num_bad_edges = bad_edges_mask.sum()
        if num_bad_edges:
            msg = f"Bad edges found: {num_bad_edges} instances in {list(entity_contact_df[bad_edges_mask]['acc'].unique())}. They have been filtered, but this issue should be resolved."
            self.logger.warning(msg)
            print(msg)
            entity_contact_df = entity_contact_df[~bad_edges_mask].reset_index()

        # calculate node ids for all entities
        entity_contact_df['u_node'] = entity_contact_df.apply(lambda row: int(row['msaCol1']) if pd.notna(row['msaCol1']) else int(msa_len + row['EIDN_1']), axis=1)
        entity_contact_df['v_node'] = entity_contact_df.apply(lambda row: int(row['msaCol2']) if pd.notna(row['msaCol2']) else int(msa_len + row['EIDN_2']), axis=1)

        self.entity_df['node_id'] =  msa_len + self.entity_df['EIDN'].astype(int)

        if self.interchain_edges == 'single':
            entity_contact_df = entity_contact_df.drop_duplicates(['acc_chn', 'inter', 'u_node', 'v_node']).reset_index()

        if not return_grouped: # if this is not being used, remove it.
            self.entity_contact_df = entity_contact_df
        else:
            entity_contact_df = entity_contact_df.groupby(['pdbx_db_accession_1', 'pdbx_db_accession_2', 'inter']).size().reset_index(name='count')
        
        return entity_contact_df

    def build_contact_img_base(self, out_type = 'CLUSTAL', fade = 0.8):
        # Create the background image over which the contact information will be displayed
        acc_msa_row_dict = {seq_rec.id: index for index, seq_rec in enumerate(self.msa)} # maybe useful for debug. delete if not using
        n_col = self.msa.get_alignment_length()
        img = []
        for seq_rec in self.msa: #for acc in self.accs:
            seq = seq_rec.seq
            row = []
            for j in range(n_col):
                res = seq[j]
                if out_type == 'CLUSTAL':
                    color = np.array(make_fainter(hex_to_rgb(clustal_colors[res]),fade)) / 255.0
                elif out_type == 'cmap': # residue coding scheme to use discreate colormap for heatmap in plotly (jankbodge)
                    color = res_color_idx_dict[str.lower(res)]
                else: # binary contact plot
                    color = 1
                row.append(color)
            img.append(row)
        img_base = np.array(img)
        return img_base
    
    def filter_contact_df(self, inter_type = 'all', inter_class = 'all', use_entity_contact_df = False):
        if use_entity_contact_df:
            if self.entity_contact_df is None:
                self.build_entity_contact_df()
            contact_df = self.entity_contact_df
        else: 
            contact_df = self.contact_df
        # filter contact_df according to desired inter_participants
        #subject_chain = self.contact_df['acc_chn'].str.split('_').str[-1]
        # some aromatic compounds like GDP are sometimes reported as sidechains for PI interactions (bug)
        #is_with_ligand = self.contact_df['inter_orientation'].str.contains('LIG')
        accepted_inter_class = ['all', 'intra', 'inter', 'LIG']
        if inter_class not in accepted_inter_class:
            raise Exception(f"Accepted values for interclass are {accepted_inter_class}, got {inter_class}")
        if inter_class == 'all':
            sele = pd.Series([True]*len(contact_df))
        elif inter_class == 'intra':
            # should only return contacts between peptides in the same chain
            # Ligands can have the same auth_asym_id so need to filter
            #subject_chain = self.q_chain # query chain is definately incorrect, one solution would be to use the chain_id from file_df
            # sele = (self.contact_df['chain1'] == subject_chain) & (self.contact_df['chain2'] == subject_chain) & (contact_df['inter_class'] != 'LIG')
            #sele = (contact_df['chain1'] == contact_df['chain2']) & (contact_df['inter_class'] != 'LIG')
            # this was changed to solve an issue with incorrect filtering with the entity contact df, take note if it causes issues
            sele = contact_df['inter_class'] == 'intra'
        elif inter_class == 'inter':
            #sele = ((self.contact_df['chain1'] == subject_chain) ^ (self.contact_df['chain2'] == subject_chain)) & (~is_with_ligand)
            sele = contact_df['inter_class'] == 'inter'
        elif inter_class == 'LIG':
            #sele = is_with_ligand
            sele = contact_df['inter_class'] == 'LIG'
        else: 
            sele = pd.Series([False]*len(contact_df))
        if inter_type != 'all':
            sele = sele & (contact_df['inter'] == inter_type)
        filtered_contact_df = contact_df[sele]
        return filtered_contact_df

    def build_contact_plot(self, inter_class):
        # create the color values for the backgorund of the MSA plot.
        # same for all plots so done once.
        if self.contact_img_base is None:
             self.contact_img_base = self.build_contact_img_base()

        contact_img_dict = {}
        # Theere is one subplot for each interaction type
        # init the dictionary where keys are the interaction type and values image array
        for contact_type in self._interaction_types:
            contact_img_dict[contact_type] = self.contact_img_base.copy()

        # map the structure identifiers to the row index in the image
        acc_msa_row_dict = {seq_rec.id: index for index, seq_rec in enumerate(self.msa)}

        # # filter contact_df according to desired inter_participants
        filtered_contact_df = self.filter_contact_df(inter_class=inter_class)#self.contact_df[sele]

        for i,r in filtered_contact_df.iterrows():
            acc = r['acc']
            acc_chn = r['acc_chn']
            msa_row = acc_msa_row_dict[acc_chn]
            inter = r['inter']
            msaCol1 = r['msaCol1']
            msaCol2 = r['msaCol2']
            alpha = 0.00
            if not pd.isna(msaCol1):
                msaCol1 = int(msaCol1)
                res1 = three_to_one(r['NodeId1'].split(':')[-1])
                contact_img_dict[inter][msa_row, msaCol1] = np.array(make_darker(hex_to_rgb(clustal_colors[res1]), alpha)) / 255.0
                
            if not pd.isna(msaCol2):
                msaCol2 = int(msaCol2)
                res2 = three_to_one(r['NodeId2'].split(':')[-1])
                contact_img_dict[inter][msa_row, msaCol2] = np.array(make_darker(hex_to_rgb(clustal_colors[res2]), alpha)) / 255.0
        return contact_img_dict

    def plot_contacts(self, inter_participants = 'all', display_acc = False, trim = False):
        """
        inter_participants: str or array of str
            accepted values: 'all', 'intra', 'inter', 'LIG'
        """
        if type(inter_participants) == str:
            inter_participants = [inter_participants]
        fig, axs = plt.subplots(len(self._interaction_types),len(inter_participants), figsize=(40, 40), squeeze=False)  
        if display_acc:
            y_lab = self.accs
        else:
            y_lab = np.arange(0,len(self.msa)).astype(str)
        y_lab[0] = 'Query'
        y_lab_pos = np.arange(len(self.msa))

        for col, col_inter_participant in enumerate(inter_participants):
            contact_img_dict = self.build_contact_plot(col_inter_participant)
            for row, inter_type in enumerate(sorted(self._interaction_types, key = lambda x: contact_order[x])):
                img = contact_img_dict[inter_type]
                
                l, r, is_trimmed = trim_msa(self.msa, trim)
                x_labels = np.arange(l, r)
                img_slice = img[:, l:r] if is_trimmed else img
                
                axs[row, col].imshow(img_slice)
                axs[row, col].set_title(inter_type, fontsize=20)
                axs[row, col].set_yticks(y_lab_pos)
                axs[row, col].set_yticklabels(y_lab)

                if is_trimmed:
                    #axs[row, col].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
                    x_tick_positions = np.linspace(0, len(x_labels) - 1, num=10, dtype=int)
                    axs[row, col].set_xticks(x_tick_positions)
                    axs[row, col].set_xticklabels(x_labels[x_tick_positions])
                    #axs[row, col].set_xticks(np.arange(len(x_labels)))
                    #axs[row, col].set_xticklabels(x_labels)
                
        plt.tight_layout()
        plt.show()

    def dist_matrix(self, contact_types, inter_class, trim_region=False, method='graph_weighted'):
        """Used to construct co-evolution plot, but also useful for evaluating similarity of proteins based on the contacts they form.

        Args:
            contact_type (str or list of str): Types of non-covalent contacts to be considered for determining similarity.
            inter_class (str): ['intra', 'inter', 'LIG', 'all']
            trim_region (tuple (int, int)): inclusive bounds for region of considered contacts
            method (string): options 'msa', 'graph' or 'graph_weighted' determines what method is used for calculaitng distances.

        Returns:
            _type_: _description_
        """
        if isinstance(contact_types, str):
            contact_types = [contact_types]
        # map the structure identifiers to the row index in the image
        acc_msa_row_dict = {seq_rec.id: index for index, seq_rec in enumerate(self.msa)}

        n_row, n_col = len(self.msa), self.msa.get_alignment_length()
        if trim_region:
            x_start, x_end = trim_region
            n_col = x_end - x_start + 1
        else:
            x_start = 0

        if method == 'msa':
            filtered_contact_df = self.filter_contact_df(inter_type = 'all', inter_class = inter_class)
            filtered_contact_df = filtered_contact_df[filtered_contact_df['inter'].isin(contact_types)]
            contact_data_dict = {}
            for inter_type in contact_types:
                contact_data_dict[inter_type] = np.zeros((n_row, n_col))
            #contact_data = np.zeros((n_row, n_col))
            for row_i, r in filtered_contact_df.iterrows():
                acc_chn = r['acc_chn']
                i = acc_msa_row_dict[acc_chn]
                j = r['msaCol1']
                inter_type = r['inter']
                if (not pd.isna(j)) and (not trim_region or x_start <= i <= x_end):
                    j = int(j)
                    # contact_data[i,j - x_start] += 1
                    contact_data_dict[inter_type][i,j - x_start] += 1
                j = r['msaCol2']
                if (not pd.isna(j)) and (not trim_region or x_start <= j <= x_end):
                    j = int(j)
                    #contact_data[i,j - x_start] += 1
                    contact_data_dict[inter_type][i,j - x_start] += 1

            #dist_mat = pdist(contact_data, metric='matching')
            dist_mat = np.zeros((n_row, n_row))
            for i in range(n_row):
                for j in range(i):
                    #print(np.dot(contact_data[i,:],contact_data[j,:]))
                    # hamming distance
                    # Idea: weight hamming distance so matched non-contacts are worth less than corresponding contacts
                    # can also weight the match by conservation of the contact in the col.
                    #dist = 1-np.sum(contact_data[i,:] == contact_data[j,:]) / n_col#np.linalg.norm(contact_data[i,:] - contact_data[j,:], 2)#distance.cosine(contact_data[i,:],contact_data[j,:])
                    dist = sum(np.sum(contact_data_dict[inter_type][i,:] != contact_data_dict[inter_type][j,:]) for inter_type in contact_types) / (n_col * len(contact_types))
                    dist_mat[i,j] = dist
                    dist_mat[j,i] = dist
            return dist_mat
        else:
            seq_ids = {rec.id for rec in self.msa}
            dist_matrix_weighted = np.zeros((n_row, n_row), dtype=float)
            dist_matrix_graph = np.zeros_like(dist_matrix_weighted, dtype=int)

            for u,v, data in self.MultiGraph.edges(data=True):

                # query filtering
                if data['inter'] not in contact_types:
                    continue
                if trim_region:
                    region_start, region_end = trim_region[0], trim_region[1]
                    if not ((region_start <= u <= region_end) or (region_start <= v <= region_end)):
                        continue
                u_nodeType, v_nodeType = self.MultiGraph.nodes[u]['node_type'], self.MultiGraph.nodes[v]['node_type']
                if inter_class == 'intra':
                    if not ((u_nodeType == 'RES') and (v_nodeType == 'RES')):
                        continue
                elif inter_class == 'inter':
                    if not (((u_nodeType == 'RES') and (v_nodeType == 'CHAIN')) or ((u_nodeType == 'CHAIN') and (v_nodeType == 'RES'))):
                        continue
                elif inter_class == 'LIG':
                    if not (((u_nodeType == 'RES') and (v_nodeType == 'LIG')) or ((u_nodeType == 'LIG') and (v_nodeType == 'RES'))):
                        continue
                # else:
                    # default behaivor is 'all'
                
                # defining weights
                obs = set(data['observations'])
                prob = data['prob']
                diff = seq_ids.difference(obs)
                for x in obs:
                    x_idx = acc_msa_row_dict[x]
                    for y in diff:
                        y_idx = acc_msa_row_dict[y]
                        dist_matrix_weighted[x_idx,y_idx] += prob
                        dist_matrix_weighted[y_idx,x_idx] += prob
                        dist_matrix_graph[x_idx,y_idx] += 1
                        dist_matrix_graph[y_idx,x_idx] += 1

            # with np.errstate(divide='ignore', invalid='ignore'):
            #     dist_matrix_weighted = np.divide(dist_matrix_weighted, dist_matrix_graph)
            #     dist_matrix_weighted[np.isnan(dist_matrix_weighted)] = 0.0
            if method == 'graph':
                return dist_matrix_graph
            elif method == 'graph_weighted':
                return dist_matrix_weighted
            else:
                raise Exception("Valid chocies for method are: 'msa', 'graph', and 'graph_weighted'")
                return 

    def plot_dist_matrix(self, contact_types, inter_class = 'all', trim_region = False, method='graph_weighted'):
        if isinstance(contact_types, str):
            if contact_types == 'all':
                contact_types = self._interaction_types
            elif contact_types in list(clustal_colors.keys()):
                contact_types = [contact_types]
            else:
                raise Exception(f"Invalid contact_type {contact_types}")
        elif isinstance(contact_types, list):
            for inter in contact_types:
                if not inter in all_interaction_types:
                    raise Exception(f"Invalid interaction type: {inter}")
        else:
            raise Exception(f"Contact_types must be string or list of strings in {all_interaction_types}")


        dist_mat = self.dist_matrix(contact_types, inter_class=inter_class, method=method, trim_region=trim_region)
        data_array = dist_mat
        ytdist = squareform(dist_mat)
        # dist_mat = rh.dist_matrix(contact_types=contact_type,inter_class = inter_class)#np.random.randn(32,32)
        # ytdist = squareform(dist_mat)
        Z = hierarchy.linkage(ytdist, 'average')

        # dendro_permutation = [int(i) for i in dendro_side['layout']['yaxis']['ticktext']]
        labels = [rec.id for rec in self.msa]

        fig = ff.create_dendrogram(np.zeros_like(dist_mat), distfun=lambda X: dist_mat, linkagefun=lambda X: Z, orientation='bottom', labels=labels) # this gives clusters consistent with scipy
        for i in range(len(fig['data'])):
            fig['data'][i]['yaxis'] = 'y2'

        dendro_side = ff.create_dendrogram(np.zeros_like(dist_mat), distfun=lambda X: dist_mat, linkagefun=lambda X: Z, orientation='right') # this gives clusters consistent with scipy
        for i in range(len(dendro_side['data'])):
            dendro_side['data'][i]['xaxis'] = 'x2'

        # Add Side Dendrogram Data to Figure
        for data in dendro_side['data']:
            fig.add_trace(data)

        # Create Heatmap
        dendro_leaves = dendro_side['layout']['yaxis']['ticktext']
        dendro_leaves = list(map(int, dendro_leaves))
        data_dist = pdist(data_array)
        heat_data = squareform(data_dist)
        heat_data = heat_data[dendro_leaves,:]
        heat_data = heat_data[:,dendro_leaves]

        heatmap = [
            go.Heatmap(
                x = dendro_leaves,
                y = dendro_leaves,
                z = heat_data,
                colorscale = 'Blues'
            )
        ]

        heatmap[0]['x'] = fig['layout']['xaxis']['tickvals']
        heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']

        # Add Heatmap Data to Figure
        for data in heatmap:
            fig.add_trace(data)

        # Edit Layout
        fig.update_layout({'width':800, 'height':800,
                                'showlegend':False, 'hovermode': 'closest',
                                })
        fig.update_layout(xaxis={'domain': [.15, 1],
                                        'mirror': False,
                                        'showgrid': False,
                                        'showline': False,
                                        'zeroline': False,
                                        'ticks':""})
        fig.update_layout(xaxis2={'domain': [0, .15],
                                        'mirror': False,
                                        'showgrid': False,
                                        'showline': False,
                                        'zeroline': False,
                                        'showticklabels': False,
                                        'ticks':""})

        fig.update_layout(yaxis={'domain': [0, .85],
                                        'mirror': False,
                                        'showgrid': False,
                                        'showline': False,
                                        'zeroline': False,
                                        'showticklabels': False,
                                        'ticks': ""
                                })
        fig.update_layout(yaxis2={'domain':[.825, .975],
                                        'mirror': False,
                                        'showgrid': False,
                                        'showline': False,
                                        'zeroline': False,
                                        'showticklabels': False,
                                        'ticks':""})

        fig.show()
        return heat_data, dendro_leaves
    
    def build_single_contact_plot(self, inter_class, contact_type):
        # create the color values for the backgorund of the MSA plot.
        # same for all plots so done once.
        if self.contact_img_base is None:
            self.contact_img_base = self.build_contact_img_base()

        contact_img = self.contact_img_base.copy()
        #dn_order_dict = {index: value for index, value in enumerate(list(reversed(leaves_order)))}

        # map the structure identifiers to the row index in the image
        acc_msa_row_dict = {seq_rec.id: index for index, seq_rec in enumerate(self.msa)}
        filtered_contact_df = self.filter_contact_df(inter_class=inter_class, inter_type=contact_type)

        for i,r in filtered_contact_df.iterrows():
            acc = r['acc']
            acc_chn = r['acc_chn']
            msa_row = acc_msa_row_dict[acc_chn]
            msaCol1 = r['msaCol1']
            msaCol2 = r['msaCol2']
            alpha = 0.00
            if not pd.isna(msaCol1):
                msaCol1 = int(msaCol1)
                res1 = three_to_one(r['NodeId1'].split(':')[-1])
                contact_img[msa_row, msaCol1] = np.array(make_darker(hex_to_rgb(clustal_colors[res1]), alpha)) / 255.0
                
            if not pd.isna(msaCol2):
                msaCol2 = int(msaCol2)
                res2 = three_to_one(r['NodeId2'].split(':')[-1])
                contact_img[msa_row, msaCol2] = np.array(make_darker(hex_to_rgb(clustal_colors[res2]), alpha)) / 255.0
        return contact_img

    def plot_contact_conservation(self, contact_type, inter_class, trim = False, calculate_on_trimmed_region=False, method='graph_weighted'):
        n_col = self.msa.get_alignment_length()
        n_row = len(self.msa)
        scl = 5
        l, r, is_trimmed = trim_msa(self.msa, trim)
        x_labels = np.arange(l, r)
        if is_trimmed:
            n_col = r - l + 1
        a = n_col/scl
        b = n_row/scl
        trimmed_info = (l,r) if is_trimmed and calculate_on_trimmed_region else False
        dist_mat = self.dist_matrix( contact_types=contact_type, inter_class=inter_class, trim_region = trimmed_info, method=method)
        ytdist = squareform(dist_mat)
        fig, axes = plt.subplots(1, 2,figsize=(a+5, b), width_ratios=[1,a])
        Z = hierarchy.linkage(ytdist, 'average')
        y_lab = [r.id for r in self.msa]
        dn = hierarchy.dendrogram(Z,orientation='left',ax=axes[0], leaf_label_func=(lambda i: y_lab[i]))
        permutation = list(reversed(dn['leaves']))
        permutated_labels = [y_lab[leaf_idx] for leaf_idx in permutation]
        img = self.build_single_contact_plot( inter_class, contact_type)[permutation,:]
        img_slice = img[:, l:r] if is_trimmed else img
        axes[1].imshow(img_slice)
        y_lab_pos = np.arange(len(self.msa))
        axes[1].set_yticks(y_lab_pos)
        axes[1].set_yticklabels(permutated_labels)
        if is_trimmed:
            #axs[row, col].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            x_tick_positions = np.linspace(0, len(x_labels) - 1, num=10, dtype=int)
            axes[1].set_xticks(x_tick_positions)
            axes[1].set_xticklabels(x_labels[x_tick_positions])

        fig.subplots_adjust(wspace=0)
        plt.show()
        return permutation, permutated_labels

    def build_contact_prob_df(self): # Optimized version
        """Canidate for deprication, replaced with create_hRIN()

        Returns:
            _type_: _description_
        """
        n_sequences = len(self.msa)
        contact_info = {}
        for _, row in self.contact_df.iterrows():
            i, j, ct = row['msaCol1'], row['msaCol2'], row['inter']
            pair = (i, j)
            if pd.isna(i) or pd.isna(j):
                 continue # contact with ligand
            if pair not in contact_info:
                max_count = sum(1 for r in range(n_sequences) if self.msa[r, i] != '-' and self.msa[r, j] != '-')
                contact_info[pair] = {'max_count': max_count, 'contact_counts': {ct: 0 for ct in self._interaction_types}}
            contact_info[pair]['contact_counts'][ct] += 1

        contact_probs = []
        for (i, j), info in contact_info.items():
            max_count = info['max_count']
            for ct, count in info['contact_counts'].items():
                prob = count / max_count if max_count > 0 else 0
                if prob > 0:  # Only include non-zero probabilities
                    contact_probs.append({
                        'i': i,
                        'j': j,
                        'contact_type': ct,
                        'probability': prob,
                        'count': count
                    })
        self.prob_df = pd.DataFrame(contact_probs)
        return self.prob_df

    def create_hRIN(self, inter_class = 'all', set_attr = False, normalize = 'strong'):
        """
        Constructs one hRIN for each contact type by calculating the probability that node pairs have given contact type in homolog list.
        cronstructs One graph for each hRIN and combines them all for optional viewing as multiGraph
        Uses the positions from QUERY file to assign positions to nodes.

        Args:
            inter_class (str, optional): _description_. Defaults to 'all'.
            set_attr (bool, optional): Save class attributes for Graphs, leaving false allows for user queries without overwriting. Defaults to False.
            normalize (str, bool): method used to nomalize interaction counts to probabilities, options are 'weak' - uses num. homologs, 'possible' - considers gaps, and False - returns counts
        Returns:
            inter_prob_smat_dict: (dict:np.array()) dictionary with values counts or probabilities that an edge given by pair of nodes of corresponding interaction type
            MG (nx.Multigraph): Multigraph with multiple edge types corresponding to the different interaction types
            G_inter_dict: (dict: nx.MultiGraph) Graph spesific to each interaction type
        """
        if self.entity_contact_df is None:
            self.build_entity_contact_df()

        msa_len = self.msa.get_alignment_length()
        if inter_class != 'intra':
            # make room to accomidate inter chain contacts
            # TODO: if class == INTER, there does not need to be room for INTRA chain contacts
            n_cols = msa_len + len(self.EIDN_dict)
            use_entity_contact_df = True
            u_col, v_col = 'u_node', 'v_node'
        else:
            n_cols = msa_len
            use_entity_contact_df = False
            u_col, v_col = 'msaCol1', 'msaCol2'
        edge_df = self.filter_contact_df(inter_class = inter_class, use_entity_contact_df = use_entity_contact_df).copy()
        # remove edges that does not have atleast one node in the query entity (peptide)
        # some analysis may want this but it is messing with the filter logic for now.
        edge_df = edge_df[~(edge_df['msaCol1'].isna() & edge_df['msaCol2'].isna())]
        n_homologs = len(self.file_df)#edge_df['acc'].nunique()
        inters = edge_df['inter'].unique()
        
        # create a dictionary will map the non-res nodes present in a given structure
        # keys: structure acc (PDB/UNP), # values: list of node ids
        EIDN_struct_presence_dict = self.entity_df.groupby('src_acc')['node_id'].apply(lambda x: list(x.unique())).to_dict()
        
        G_inter_dict = {} # dictionary of NX graphs for each interaction (hRINs)
        inter_prob_smat_dict = {} # dictionary where each entry contains a sparse matrix with enteries the probability node pairs form contacts in homologs
        for inter in inters:
            inter_prob_smat_dict[inter] = dok_matrix((n_cols, n_cols), dtype = float if normalize else int)
            G_inter = nx.Graph()
            f_edges_df = edge_df[edge_df['inter'] == inter]
            nodes_has_inter = []
            for _, r in f_edges_df.iterrows():
                u,v = int(r[u_col]), int(r[v_col])
                edge_inter_class = r['inter_class']
                # After block know nodes u,v exist in graph
                if G_inter.has_edge(u,v):
                    e = G_inter[u][v]
                    e['count'] += 1
                    #e['prob'] = e['count'] / n_homologs
                    e['observations'].append(r['acc_chn'])
                else:
                    G_inter.add_edge(u,v, inter=inter, prob=1 / n_homologs, observations = [r['acc_chn']], count=1, color=contact_colors[inter], inter_class=edge_inter_class)
                    e = G_inter[u][v]
                    nodes_has_inter.append(u)
                    nodes_has_inter.append(v)

                # add information to nodes to signifiy what they represent (MSA col, chain, LIGAND)
                u_node, v_node = G_inter.nodes[u], G_inter.nodes[v]
                if ('node_type' not in u_node) or ('node_type' not in v_node):
                    if edge_inter_class != 'intra':
                        # Most of the time the first residue in inter-chain contact will be the one in the query chain, but not always.
                        # contact_df is filtered, so one node will always be residue. The other may be res in same chain, other chain, or ligand
                        if r['u_node'] < msa_len: # zero-indexed so inequality is strong
                            
                            (query_node, other_node) = (u_node, v_node)
                            # u node is in the query chain and v node is OTHER
                            # other information corresponds to node2 in edge_df
                            other_identifier = r['identifier_2']
                            other_EIDN = r['EIDN_2']
                            other_db_code = r['db_code_2']
                            other_desc = r['pdbx_description_2']
                        else:
                            (query_node, other_node) = (v_node, u_node)
                            other_identifier = r['identifier_1']
                            other_EIDN = r['EIDN_1']
                            other_db_code = r['db_code_1']
                            other_desc = r['pdbx_description_1']
                        query_node['node_type'] = 'RES'
                        query_node['color'] = colors.to_hex('tab:blue')
                        other_type = 'LIG' if edge_inter_class == 'LIG' else 'CHAIN'
                        other_node['consensus'] = 'LIG' if edge_inter_class == 'LIG' else 'CHN'
                        other_node['node_type'] = other_type
                        other_node['identifier'] = other_identifier
                        other_node['EIDN'] = other_EIDN
                        other_node['db_code'] = other_db_code
                        other_node['pdbx_description'] = other_desc
                        other_node['x'] = self.EIDN_pos_dict[other_EIDN]['Cartn_x']
                        other_node['y'] = self.EIDN_pos_dict[other_EIDN]['Cartn_y']
                        other_node['z'] = self.EIDN_pos_dict[other_EIDN]['Cartn_z']
                        other_color = colors.to_hex('tab:green') if other_type == 'CHAIN' else colors.to_hex('tab:red')
                        other_node['color'] = other_color
                    else:
                        # both nodes are residues for intra-chain contacts (residues in the MSA)
                        u_node['node_type'] = 'RES'
                        u_node['color'] = colors.to_hex('tab:blue')
                        if ('node_type' not in v_node):
                            v_node['node_type'] = 'RES'
                            v_node['color'] = colors.to_hex('tab:blue')
                
                # Neither MSA column numbers or EIDN values are 1 indexed, so should not offset the index here.
                inter_prob_smat_dict[inter][u, v] += 1.0
                inter_prob_smat_dict[inter][v, u] += 1.0

            # Adding query position information to nodes for plotting
            q_atoms_df = self.atom_df[(self.atom_df['acc'] == self.query_acc) & (self.atom_df['label_atom_id'] == 'CA')]
            q_atoms_df = q_atoms_df[['acc', 'msa_col', 'Cartn_x', 'Cartn_y', 'Cartn_z']]
            # get only atoms that have atleast one of the current interaction type
            #q_atoms_df['auth_seq_id'] = q_atoms_df['auth_seq_id'].astype(int)
            q_atoms_df = q_atoms_df[q_atoms_df['msa_col'].isin(nodes_has_inter)]
            # Set pos for nodes corresponding to MSA residues
            for i, r in q_atoms_df.iterrows():
                u_id = int(r['msa_col'])
                node_u = G_inter.nodes[u_id]
                node_u['x'], node_u['y'], node_u['z'] = [float(coord) for coord in r[['Cartn_x', 'Cartn_y', 'Cartn_z']]]
            # returns count if not weak or strong
            if normalize == 'strong':
                inter_prob_smat_dict[inter] /= n_homologs # probabably better to get the number of non-gaps in a col
            elif normalize == 'possible':
                msa_n_cols = self.msa.get_alignment_length()
                msa_n_rows = len(self.msa)
                for u,v in inter_prob_smat_dict[inter].keys():
                    # this is somewhat inefficient but the easiest way to traverse sparse mat
                    # if node is non-residue will not normalize based on chain / lig
                    # TODO: instead of counting TRUE if node is not of type RES, create a means of checking if EIDN is present in structure corresponding to row
                    u_nongap = [res != '-' for res in self.msa[:,u]] if u < msa_n_cols else [u in EIDN_struct_presence_dict[rec.id.split('_')[0]] for rec in self.msa] # [True] * msa_n_rows
                    v_nongap = [res != '-' for res in self.msa[:,v]] if v < msa_n_cols else [v in EIDN_struct_presence_dict[rec.id.split('_')[0]] for rec in self.msa] # [True] * msa_n_rows
                    z = sum(r and s for r, s in zip(u_nongap, v_nongap))
                    z = z if z else 1 # avoid divide by zero
                    inter_prob_smat_dict[inter][u,v] /= z
            for u, v, data in G_inter.edges(data=True): # update the graph with edge weight information
                G_inter[u][v]['prob'] = inter_prob_smat_dict[inter][u,v]
            # the above will only add nodes if they have a contact. 
            # the following will create the missing RES nodes that don't have a contact.
            for i, res in enumerate(self.consensus_seq):
                if i not in G_inter.nodes:
                    G_inter.add_node(i, node_type = 'RES', consensus = one_to_three(res))
                if G_inter.nodes[i].get('node_type') is None:
                    G_inter.nodes[i]['node_type'] = 'RES'
                G_inter.nodes[i]['consensus'] = one_to_three(res)

            G_inter_dict[inter] = G_inter


        MG = nx.MultiGraph()
        for inter, G in G_inter_dict.items():
            for node, data in G.nodes(data=True):
                if node not in MG:
                    MG.add_node(node)
                MG.nodes[node].update(data)
            # Add edges along with their data attributes to the MultiGraph
            for u, v, data in G.edges(data=True):
                try:
                    existant_multi_edges = len(MG[u][v])
                except:
                    existant_multi_edges = 0
                MG.add_edge(u, v, **data, _crv_idx = existant_multi_edges)
                for k in MG[u][v]:
                    MG[u][v][k]['_crv_total'] = existant_multi_edges + 1

        for u, v, k, d in MG.edges(keys=True, data=True):
            MG[u][v][k]['_crv_offset'] = d['_crv_idx'] - 0.5 * d['_crv_total'] + 0.5

        if set_attr: # user may want to create a hRIN for a query. In which case, do not set class attributes.
            self.edge_colors = [mcolors.to_rgba(data['color'])[:3] for u, v, key, data in MG.edges(keys=True, data=True)]
            self.node_pos = {i: [data.get('x', 0.0), data.get('y', 0.0)] for i, data in MG.nodes(data=True)}
            self.hRIN_dict = inter_prob_smat_dict # save dictionary of sparse matricies containing probabilities of contacts across homologs
            self.MultiGraph = MG # save multi-graph with all interactions
            self.Graph_inter_dict = G_inter_dict # save dictionary for interaction unique graphs
        return inter_prob_smat_dict, MG, G_inter_dict

    def plot_hRIN_prob(self, hRIN_dict=None, inter='all', sparse_plot=True):
        # Create a figure and axis with adjusted width for the colorbar
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_axes([0.15, 0.1, 0.7, 0.8])  # [left, bottom, width, height]
        
        if hRIN_dict is None:
            # allows for user to optinally pass a dict that is the result of a custom query.
            # if none is provided, will use the one from parent.
            hRIN_dict = self.hRIN_dict

        if sparse_plot:
            ax2 = ax1.twinx()
            rows = set()
            cols = set()
            for k, p in hRIN_dict[inter].items():
                i, j = k[0], k[1]
                rows.add(i)
                cols.add(j)
            row_slice = sorted(list(rows))
            col_slice = sorted(list(cols))
            idx_key = {idx: nid for idx, nid in enumerate(row_slice)}

            img = hRIN_dict[inter].A[row_slice, :][:, col_slice]
            im = ax1.imshow(img, cmap='hot', aspect='auto')
            
            if img.shape[0] <= 40: # ticktext gets too close if there's too many nodes
                # y axis left residue/node number
                ax1.set_yticks(np.arange(img.shape[0]))
                ax1.set_yticklabels([str(row) for row in row_slice])
                
                # y axis right - residue/node identifier (name)
                ax2_tick_locs = np.arange(img.shape[0])
                ax2.set_yticks(ax2_tick_locs)
                ax2.set_ylim(ax1.get_ylim())
                ax2.set_yticklabels([self.node_dict[i] for i in row_slice])
                ax2.set_ylabel('Node Entity')
                # X- axis
                ax1.set_xticks(np.arange(img.shape[1]))
                ax1.set_xticklabels([str(col) for col in col_slice], rotation=90)
        else:
            img = hRIN_dict[inter].A
            im = ax1.imshow(img, cmap='hot', aspect='auto')

        # Create an axes for the colorbar on the left side, adjusting its position
        cbar_ax = fig.add_axes([0.05, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
        
        ax1.set_title(inter)
        ax1.set_ylabel('Node ID')
        ax1.set_xlabel('Node ID')
        plt.show()
        return img, idx_key

    def plot_contact_prob(self):
        n_cols = self.msa.get_alignment_length()
        if self.prob_df is None:
            self.build_contact_prob_df()
        # Initialize an empty probability matrix for each contact type
        probs_dict = {ct: np.zeros((n_cols, n_cols)) for ct in self._interaction_types}
        # if you want the diagonal and below to be empty (np.nan) instead of 0 
        # probs_dict = {ct: np.full((n_cols, n_cols), np.nan) for ct in self._interaction_types}
        # for ct in self._interaction_types:
        #     upper_tri_indices = np.triu_indices(n_cols, k=1)
        #     probs_dict[ct][upper_tri_indices] = 0

        # Populate the probability matrices using the contact_prob_df
        for _, row in self.prob_df.iterrows():
            i, j = row['i'], row['j']
            ct = row['contact_type']
            prob = row['probability']
            probs_dict[ct][i, j] = prob
            probs_dict[ct][j, i] = prob  # For symmetric plots

        # Plotting
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs_flat = axs.flatten()
        for ax, (key, array) in zip(axs_flat, probs_dict.items()):
            print(f"Max prob for {key}: {np.max(probs_dict[key]):.3f}")
            img = ax.imshow(array, vmin=0, vmax=1, cmap='hot', aspect='auto')
            ax.set_title(key) 
            plt.colorbar(img, ax=ax)
            ax.invert_yaxis()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle("Probability of Contact", fontsize=20)
        plt.show()
        return probs_dict

    def build_network(self):
        """Canidate for deprication, replaced with create_hRIN
        """
        self.G = nx.MultiGraph()
        # the contact df needs to be filtered to ensure that msaCol1 and 2 are not NA.
        # this is possible because of the addition of AlphaFill and the contact df was reimplement to keep residues outside of the MSA range
        edge_df = self.contact_df[~(self.contact_df['msaCol1'].isna() | self.contact_df['msaCol2'].isna())]
        # infer the num cols in the MSA from indicies
        # add node for each MSA col
        for i in range(self.msa.get_alignment_length()):#range(np.max(self.contact_df[['msaCol1','msaCol2']])):
            self.G.add_node(i, )
        for i, r in edge_df.iterrows():
            self.G.add_edge(r['msaCol1'], r['msaCol2'], acc=r['acc'], interaction=r['inter'])
        return
    
    
    
    def plot_hRIN_network(self, inter_types='all', inter_class='all', trim=False, layout='kamada_kawai', arc_multi_edges=True, weight_edges=True, spring_k=0.2, prob_weight=1, min_weight=0.5, crv_rad = 0.3, res_lables = True, het_lables = 'gene_name', node_size = 15):
        """_summary_

        Args:
            inter_types (str, optional): _description_. Defaults to 'all'.
            inter_class (str, optional): _description_. Defaults to 'all'.
            trim (bool, optional): _description_. Defaults to False.
            layout (str, optional): Method used to assign node positions. Defaults to 'kamada_kawai' other options 'spring' or str of coords to use from query structure e.g. 'xy' or 'zx'.
            arc_multi_edges (bool, optional): _description_. Defaults to True.
            weight_edges (bool, optional): _description_. Defaults to True.
            spring_k (float, optional): _description_. Defaults to 0.2.
            prob_weight (int, optional): _description_. Defaults to 1.
            min_weight (float, optional): _description_. Defaults to 0.5.
            crv_rad (float, optional): Amount multiEdges will curve. Defaults to 0.3.

        """
        out_graph = nx.MultiGraph()
        # node_size = 15
        #plt.figure(dpi=300) # for higher quality output

        # Ensure valid interaction types
        valid_inters = list(contact_colors.keys())
        if isinstance(inter_types, str):
            if inter_types == 'all':
                inter_types = valid_inters
            else:
                inter_types = [inter_types]
        elif not (isinstance(inter_types, list) and all(isinstance(t, str) and t in valid_inters for t in inter_types)):
            raise Exception(f"inter_types must be str or list of str. Valid interactions are {valid_inters}.")

        # Filter graph (edges and nodes) based on query parameters
        l, r, _ = trim_msa(self.msa, trim)# x_min, x_max = ()
        for u, v_dict in self.MultiGraph.adj.items():
            for v, data in v_dict.items():
                if u > v: # adj will iterate over each edge twice, consider only one instance
                    continue
                if not ((l <= u <= r) or (l <= v <= r)): # Skip in node is not in the trim region
                    continue
                # print(u,v)
                u_node = self.MultiGraph.nodes[u]
                v_node = self.MultiGraph.nodes[v]
                # enforce edge is of the correct intra_class
                if (inter_class == 'intra'):
                    if (u_node['node_type'] != 'RES') or (v_node['node_type'] != 'RES'):
                        continue
                elif (inter_class == 'inter'):
                    if not (((u_node['node_type'] == 'CHAIN') and (v_node['node_type'] == 'RES')) or ((u_node['node_type'] == 'RES') and (v_node['node_type'] == 'CHAIN'))):
                        continue
                elif (inter_class == 'LIG'):
                    if not (((u_node['node_type'] == 'LIG') and (v_node['node_type'] == 'RES')) or ((u_node['node_type'] == 'RES') and (v_node['node_type'] == 'LIG'))):
                        continue
                num_selected_edges = 0
                selected_edges = []
                for edge in data.values():
                    # enforce edge is of inter_type of interest
                    edge_type = edge['inter']
                    if edge_type not in inter_types:
                        continue
                    new_edge = edge.copy()
                    new_edge['crv_idx'] = num_selected_edges
                    selected_edges.append(new_edge)
                    num_selected_edges += 1
                for multi_id, edge in enumerate(selected_edges):
                    edge['_crv_total'] = num_selected_edges
                    edge['_crv_offset'] = 0.0 if num_selected_edges == 1 else (multi_id - (num_selected_edges-1)/2) / ((num_selected_edges-1)/2) * crv_rad
                    out_graph.add_edge(u,v, key=multi_id, **edge)
        if layout == 'spring':
            node_pos = nx.spring_layout(out_graph, k=spring_k)
        elif layout == 'kamada_kawai':
            node_pos = nx.kamada_kawai_layout(out_graph, scale=100)
        else:
            if (len(layout) != 2) or (sum(coord in str.lower(layout) for coord in ['x', 'y', 'z']) < 2):
                raise Exception()
            coord_x, coord_y = str.lower(layout)[0], str.lower(layout)[1]
            node_pos = {}
            for nid in out_graph.nodes():
                interp_node_pos(self, nid)
                data = self.MultiGraph.nodes[nid]
                node_pos.update({nid: (data[coord_x], data[coord_y])})
        node_colors = []
        node_type_color_map = {'RES': 'tab:gray', 'CHAIN':'tab:green', 'LIG':'tab:red'}
        for nid, data in out_graph.nodes(data=True):
            out_graph.nodes[nid].update(self.MultiGraph.nodes[nid])
            node_colors.append(node_type_color_map[data['node_type']])
        if arc_multi_edges:
            nx.draw_networkx_nodes(out_graph, node_size=node_size, pos=node_pos, node_color=node_colors)
            for u, v, data in out_graph.edges(data=True):
                rad = data['_crv_offset']
                edge_color = data['color']
                prob = data['prob']
                u_pos, v_pos = node_pos[u], node_pos[v]
                patch = patches.FancyArrowPatch(
                    posA=u_pos, posB=v_pos,
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle='-', color=edge_color, lw=prob_weight * prob + min_weight,
                    mutation_scale=10.0, zorder=0
                )
                plt.gca().add_patch(patch)
        else:
            nx.draw(out_graph, node_size=node_size, pos=node_pos, node_color=node_colors)


        label_dict = {}
        for i, data in out_graph.nodes(data=True):
            if data['node_type'] == 'RES':
                cur_node_label = str(i) if res_lables else ''
            elif het_lables == 'acc':
                cur_node_label = data.get('identifier', '')
            elif het_lables == 'gene_name':
                cur_node_label = data.get('db_code', '')
            else:
                cur_node_label = ''
            label_dict[i] = cur_node_label
        nx.draw_networkx_labels(out_graph, node_pos, labels=label_dict, font_size=12, font_color='darkred', font_family='sans-serif', font_weight='bold')
        # if het_lables == 'acc':
        #     nx.draw_networkx_labels(out_graph, node_pos, labels={i: str(i) if data['node_type'] == 'RES' else data.get('identifier', '') for i, data in out_graph.nodes(data=True)}, font_size=12, font_color='darkred', font_family='sans-serif', font_weight='bold')
        # elif het_lables == 'gene_names':
        #     nx.draw_networkx_labels(out_graph, node_pos, labels={i: str(i) if data['node_type'] == 'RES' else data.get('db_code', '') for i, data in out_graph.nodes(data=True)}, font_size=12, font_color='darkred', font_family='sans-serif', font_weight='bold')
        return out_graph

    def export_hRIN(self, output_name, net=None, out_dir = None):
        if net is None:
            net = self.MultiGraph
        elif not (isinstance(net, nx.MultiGraph) or isinstance(net, nx.Graph)):
            raise Exception("net must be a hRIN in the form of  nx.MultiGraph or nx.Graph")
        edges = []
        for u, v, data in net.edges(data=True):
            row = {'u': u, 'v': v}
            row.update(data)
            edges.append(row)
        edges = pd.DataFrame(edges)

        nodes = []
        for u, data in net.nodes(data=True):
            row = {'u': u}
            row.update(data)
            nodes.append(row)
        nodes = pd.DataFrame(nodes)

        node_pth = f'{output_name}_nodes.tsv' if out_dir is None else os.paht.join(out_dir, f'{output_name}_nodes.tsv')
        edge_pth = f'{output_name}_edges.tsv' if out_dir is None else os.paht.join(out_dir, f'{output_name}_edges.tsv')
        nodes.to_csv(node_pth, sep='\t', index=False)
        edges.to_csv(edge_pth, sep='\t', index=False)
        print(f"Wrote files {[node_pth, edge_pth]}")

    def plot_contact_net(self, use_coords='xz', arced=True, prob_weight = 1.3, min_weight = 0.3):
        """
        Needs to be adapeded for use with create_hRIN - code in NB

        Args:
            use_coords (str, optional): _description_. Defaults to 'xz'.
            arced (bool, optional): _description_. Defaults to True.
            prob_weight (float, optional): _description_. Defaults to 1.3.
            min_weight (float, optional): _description_. Defaults to 0.3.

        Raises:
            Exception: _description_
        """
        use_coords = str.lower(use_coords)
        cord1_mmCIF_colname = f'Cartn_{use_coords[0]}'
        cord2_mmCIF_colname = f'Cartn_{use_coords[1]}'
        #cord_to_idx = {'x': 0, 'y': 1, 'z': 2}
        #atom_pos_contig = self.q_atom_pos[:, [cord_to_idx[use_coords[0]], cord_to_idx[use_coords[1]]]]
        q_atom_pos_contig_df = self.atom_df[(self.atom_df['acc'] == query_acc)] # filter out just the query struct from atom_df
        q_atom_pos_contig_df = q_atom_pos_contig_df[(q_atom_pos_contig_df['group_PDB'] == 'ATOM')] # Some Calcium Ligands can pass the previous query
        q_atom_pos_contig_df = q_atom_pos_contig_df[(q_atom_pos_contig_df['label_atom_id'] == 'CA')] # get just the carbon-alpha locations
        q_atom_pos_contig_df = q_atom_pos_contig_df[q_atom_pos_contig_df['label_alt_id'].isin(['.','A'])] #for some residues alternate locations are given. Take the first (or only) location.
        if q_atom_pos_contig_df['label_seq_id'].duplicated().any() or q_atom_pos_contig_df['auth_seq_id'].duplicated().any():
            raise Exception('Residue with multiple CA postions per residue')
        q_atom_pos_contig_pos = q_atom_pos_contig_df[[cord1_mmCIF_colname,cord2_mmCIF_colname]].to_numpy(dtype=float)
        q_nonGap_msaCol = list(q_atom_pos_contig_df['msa_col'])
        
        nongap_idx = 0
        atom_pos = []
        for msa_col in range(self.msa.get_alignment_length()):
            cur_res = self.msa[0,msa_col]
            if cur_res != '-':
                atom_pos.append(q_atom_pos_contig_pos[nongap_idx,:])
                #print('nongap', atom_pos[-1].shape)
                nongap_idx += 1
            else:
                atom_pos.append(np.zeros((2)))
                #print('gap', atom_pos[-1].shape)
        #print(len(atom_pos))
        #print(sum([res != '-' for res in str(self.msa[0].seq)]))
        atom_pos = np.array(atom_pos)
        atom_pos = impute_zeros(atom_pos)

        atom_pos_scaled = np.array(atom_pos).astype(float)
        lb = np.min(atom_pos_scaled, axis=0)
        ub = np.max(atom_pos_scaled, axis=0)
        atom_pos_scaled = -1 + ((atom_pos_scaled - lb) / (ub - lb)) * (2)
        pos = {i: atom_pos_scaled[i] for i in range(len(atom_pos_scaled))}

        if self.prob_df is None:
            self.build_contact_prob_df()
        if self.G is None:
            self.build_network()

        # nx.draw_networkx_nodes(self.G, pos, node_size=10)
        #plt.figure(dpi=400)

        if arced:
            # Drawing arcs for edges with curvature adjustments
            edge_curvature = {}
            for _, row in self.prob_df.iterrows():
                pair = (row['i'], row['j'])
                if pair not in edge_curvature:
                    edge_curvature[pair] = 0
                edge_curvature[pair] += 1

            for pair, count in edge_curvature.items():
                if count == 1:
                    edge_curvature[pair] = np.array([0])
                else:
                    edge_curvature[pair] = np.linspace(-0.15, 0.15, count)

            edge_index = {}
            for _, row in self.prob_df.iterrows():
                u, v = row['i'], row['j']
                ct = row['contact_type']
                color = contact_colors.get(ct, 'black')
                pair = (u, v)
                prob = row['probability']

                if pair not in edge_index:
                    edge_index[pair] = 0
                
                rad = edge_curvature[pair][edge_index[pair]]
                edge_index[pair] += 1

                start_pos = np.array(pos[u])
                end_pos = np.array(pos[v])

                patch = patches.FancyArrowPatch(posA=start_pos, posB=end_pos,
                                                connectionstyle=f"arc3,rad={rad}",
                                                arrowstyle='-', color=color, lw= prob_weight*prob + min_weight,
                                                mutation_scale=10.0, zorder=0)
                plt.gca().add_patch(patch)
        else:
            # Drawing straight edges or no edges if prob_df is not provided
            # for (u, v, data) in self.G.edges(data=True):
            #     color = contact_colors.get(data.get('interaction'), 'black')
            for _, r in self.prob_df.iterrows():
                u, v = r['i'], r['j']
                color = 'black'#contact_colors.get(r['contact_type'])
                prob = r['probability']
                start_pos = np.array(pos[u])
                end_pos = np.array(pos[v])

                plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], color=color, lw=prob_weight*prob + min_weight)

        nx.draw_networkx_nodes(self.G, pos, node_size=10)

        plt.axis('equal')
        plt.axis('off')
        plt.show()
    
    def plot_contact_net_3D(self, prob_weight=1.3, min_weight=0.3, elev=30, azim=45):
        
        if self.prob_df is None:
            self.build_contact_prob_df()
        # Order contact prob df so that more interesting interactions are plotted on top
        self.prob_df['contact_type'] = pd.Categorical(self.prob_df['contact_type'], categories=['VDW', 'HBOND', 'IONIC', 'PIPISTACK', 'PICATION', 'PIHBOND'], ordered=True)

        # Assuming atom positions are correctly assigned to self.q_atom_pos
        atom_pos = self.q_atom_pos

        # Ensure atom positions are correctly scaled
        atom_pos_scaled = np.array(atom_pos).astype(float)
        lb = np.min(atom_pos_scaled, axis=0)
        ub = np.max(atom_pos_scaled, axis=0)
        atom_pos_scaled = -1 + ((atom_pos_scaled - lb) / (ub - lb)) * (2)
        pos = {i: atom_pos_scaled[i] for i in range(len(atom_pos_scaled))}

        # Create the 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=elev, azim=azim)

        # Plot nodes
        node_xyz = np.array([pos[node] for node in sorted(self.G.nodes)])
        ax.scatter(*node_xyz.T, s=20)

        # Plot edges with dynamic line width based on probability
        for _, row in self.prob_df.sort_values('contact_type').iterrows():
            u = row['i']
            v = row['j']
            pos_u = pos[u]
            pos_v = pos[v]
            probability = row['probability']
            line_width = min_weight + (prob_weight * probability)  # Adjust line width based on probability
            ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], [pos_u[2], pos_v[2]], color=contact_colors[row['contact_type']], linewidth=line_width)

        def _format_axes(ax):
            """Visualization options for the 3D axes."""
            ax.grid(False)  # Turn gridlines off
            ax.xaxis.set_ticks([])  # Suppress tick labels
            ax.yaxis.set_ticks([])
            ax.zaxis.set_ticks([])
            ax.set_xlabel("x")  # Set axes labels
            ax.set_ylabel("y")
            ax.set_zlabel("z")

        _format_axes(ax)
        plt.tight_layout()
        plt.show()

    def _trim_cif(self, pth, sstart, send, chain_id = 'A', state_id = '1', seq=None):
        #f'{os.path.splitext(pth)[0]}_{chain_id}{state_id}.cif'
        name = os.path.basename(pth).split('.')[0]
        out_pth =  f'cif/trimmed/{name}.cif'
        ATOM_header = []
        atom_pos = []
        AA_seq = ''
        with open(pth,'r') as src, open(out_pth, 'w') as dst:
            # Write beginning of CIF file
            dst.writelines([f"data_{name}\n","#\n",f"_entry.id   {name}\n","#\n","loop_\n"])
            for l in src:
                if l.startswith('_atom_site.'): # ATOM header
                    ATOM_header.append(l.split('.')[1].strip())
                    dst.write(l)
                if l.startswith('ATOM'): # ATOM data
                    l_tab = re.sub(r"\s+(?=\S)", "\t", l) # CIF uses inconsitent whitespace delims, replace with one tab
                    ATOM_arr = l_tab.split('\t')

                    chain = ATOM_arr[ATOM_header.index('label_asym_id')].strip()
                    state = ATOM_arr[ATOM_header.index('pdbx_PDB_model_num')].strip()
                    seq_idx = ATOM_arr[ATOM_header.index('label_seq_id')].strip()
                    atom_lbl = ATOM_arr[ATOM_header.index('label_atom_id')].strip() # needed to get the pos for only carbon alpha
                    x = ATOM_arr[ATOM_header.index('Cartn_x')].strip()
                    y = ATOM_arr[ATOM_header.index('Cartn_y')].strip()
                    z = ATOM_arr[ATOM_header.index('Cartn_z')].strip()
                    res = three_to_one(ATOM_arr[ATOM_header.index('label_comp_id')].strip())
                    
                    # Make sure that retained atoms are in desired state and chain
                    if (chain == chain_id) and (state == state_id) and (int(seq_idx) >= int(sstart)) and (int(seq_idx) <= int(send)):
                        dst.write(l)
                        if atom_lbl == 'CA':
                            atom_pos.append((x,y,z))
                            AA_seq += res
        if (seq != None) and (seq != AA_seq):
            self.logger.error(f"BLAST/CIF mismatch for {name}.")
            self.logger.error(f"BLAST:\t\t{seq}")
            self.logger.error(f"ALPHAFOLD:\t{AA_seq}")
        return np.array(atom_pos).astype(float)
    
    def get_pos(self, pth, start = None, end = None, chain_id = 'A', state_id = '1', atoms = 'CA', seq=None, return_seq = False):
        #f'{os.path.splitext(pth)[0]}_{chain_id}{state_id}.cif'
        name = os.path.basename(pth).split('.')[0]
        #out_pth =  f'cif/trimmed/{name}.cif'
        ATOM_header = []
        atom_pos = []
        AA_seq = ''
        with open(pth,'r') as src:#, open(out_pth, 'w') as dst:
            # Write beginning of CIF file
            #dst.writelines([f"data_{name}\n","#\n",f"_entry.id   {name}\n","#\n","loop_\n"])
            for l in src:
                if l.startswith('_atom_site.'): # ATOM header
                    ATOM_header.append(l.split('.')[1].strip())
                    #dst.write(l)
                if l.startswith('ATOM'): # ATOM data
                    l_tab = re.sub(r"\s+(?=\S)", "\t", l) # CIF uses inconsitent whitespace delims, replace with one tab
                    ATOM_arr = l_tab.split('\t')

                    chain = ATOM_arr[ATOM_header.index('label_asym_id')].strip()
                    state = ATOM_arr[ATOM_header.index('pdbx_PDB_model_num')].strip()
                    seq_idx = ATOM_arr[ATOM_header.index('label_seq_id')].strip()
                    atom_lbl = ATOM_arr[ATOM_header.index('label_atom_id')].strip() # needed to get the pos for only carbon alpha
                    x = ATOM_arr[ATOM_header.index('Cartn_x')].strip()
                    y = ATOM_arr[ATOM_header.index('Cartn_y')].strip()
                    z = ATOM_arr[ATOM_header.index('Cartn_z')].strip()
                    res = three_to_one(ATOM_arr[ATOM_header.index('label_comp_id')].strip())
                    
                    # Make sure that retained atoms are in desired state and chain
                    # allows for start or end = None which takes from start or until end respectively
                    if (chain == chain_id) and (state == state_id) and \
                    ((start == None) or (int(seq_idx) >= int(start))) and \
                    ((end == None) or (int(seq_idx) <= int(end))):
                        #dst.write(l)
                        if atom_lbl == 'CA' and atoms == 'CA':
                            atom_pos.append((x,y,z))
                            AA_seq += res
                        elif atoms == 'all':
                            atom_pos.append((x,y,z))
                        elif atoms not in ['CA', 'all']:
                            print(atoms)
                            raise Exception("atoms must be CA for alpha carbons or all.")
        if (seq != None) and (seq != AA_seq):
            self.logger.error(f"BLAST/CIF mismatch for {name}.")
            self.logger.error(f"BLAST:\t\t{seq}")
            self.logger.error(f"ALPHAFOLD:\t{AA_seq}")
        out = np.array(atom_pos).astype(float)
        if return_seq:
            return out, AA_seq
        else:
            return out

    def _append_alignment(self, subject_pth, ens_path, rot_tran, state, seperate_states):
        ATOM_header_ens = []
        ATOM_header_sbj = []
        col_idx = []
        inHeader = False
        with open(ens_path, 'r') as dst:
            for l in dst:
                if l.startswith('_atom_site.'): # ATOM header
                    ATOM_header_ens.append(l.split('.')[1].strip())
        with open(subject_pth,'r') as src, open(ens_path, 'a') as dst:
            for l in src:
                if l.startswith('_atom_site.'): # ATOM header
                    inHeader = True
                    ATOM_header_sbj.append(l.split('.')[1].strip())
                elif inHeader:
                    inHeader = False
                    col_idx = [ATOM_header_ens.index(col) for col in ATOM_header_sbj if col in ATOM_header_ens]
                if l.startswith('ATOM'): # ATOM data
                    l_tab = re.sub(r"\s+(?=\S)", "\t", l) # CIF uses inconsitent whitespace delims, replace with one tab
                    ATOM_arr = l_tab.split('\t')
                    state_idx = ATOM_header_sbj.index('pdbx_PDB_model_num')
                    x_idx = ATOM_header_sbj.index('Cartn_x')
                    y_idx = ATOM_header_sbj.index('Cartn_y')
                    z_idx = ATOM_header_sbj.index('Cartn_z')
                    chain_idx = ATOM_header_sbj.index('label_asym_id')
                    chain_auth_idx = ATOM_header_sbj.index('auth_asym_id')

                    # TODO: deal with multi chain structures.
                    chain = ATOM_arr[chain_idx]
                    xyz = np.array((
                        ATOM_arr[x_idx].strip(),
                        ATOM_arr[y_idx].strip(),
                        ATOM_arr[z_idx].strip()
                    ), dtype=np.float64)
                    rot, tran = rot_tran
                    xyz_aln = np.dot(xyz, rot) + tran
                    
                    ATOM_arr[x_idx] = xyz_aln[0]
                    ATOM_arr[y_idx] = xyz_aln[1]
                    ATOM_arr[z_idx] = xyz_aln[2]
                    if seperate_states:
                        ATOM_arr[state_idx] = state
                    else: # seperate by chain
                        ATOM_arr[chain_idx] = int_to_letters(state - 1)
                        ATOM_arr[chain_auth_idx] = int_to_letters(state - 1)
                        ATOM_arr[state_idx] = 1
                    l_arr = [ATOM_arr[col_idx[i]] for i in range(len(col_idx))]
                    line = ' '.join([str(val) for val in l_arr]) + '\n'
                    dst.write(line)

    def buildEnsemble(self, out_pth, seperate_states = False):
        shutil.copyfile(self.trimmed_pth, out_pth)
        self.file_df['ens_id'] = None
        if seperate_states:
            self.file_df.loc[query_acc,'ens_id'] = 1
        else: # seperate by chain
            self.file_df.loc[query_acc,'ens_id'] = 'A'
        for i, r in self.blast_df.iterrows():
            acc = r['acc']
            pos_q1 = get_pos1(self.atom_df, query_acc, r['qstart'], r['qend'])
            pos_s1 = get_pos1(self.atom_df, acc, r['sstart'], r['send'])
            
            pos_q1 = pos_q1[~pos_q1['msa_col'].isna()]
            pos_s1 = pos_s1[~pos_s1['msa_col'].isna()]
            pos_q1['msa_col'] = pos_q1['msa_col'].astype(int)
            pos_s1['msa_col'] = pos_s1['msa_col'].astype(int)
            pos_df = pd.merge(pos_q1, pos_s1, on='msa_col', how='inner')
            pos_q_match = pos_df[['Cartn_x_x', 'Cartn_y_x', 'Cartn_z_x']].to_numpy(dtype=float)
            pos_s_match = pos_df[['Cartn_x_y', 'Cartn_y_y', 'Cartn_z_y']].to_numpy(dtype=float)

            sup = SVDSuperimposer()
            sup.set(pos_q_match, pos_s_match)
            sup.run()
            rot_tran = sup.get_rotran()
            # TODO: Modify _append_alignment to use the infromation from atom_df instead of the trimmed files
            self._append_alignment(f'cif/trimmed/{acc}.cif', out_pth, rot_tran, i+2, seperate_states)
            if seperate_states:
                self.file_df.loc[acc,'ens_id'] = i+2
            else: # seperate by chain
                self.file_df.loc[acc,'ens_id'] = int_to_letters(i + 1)
    # def buildEnsemble(self, out_pth, seperate_states = False):
    #     n_row = len(self.msa)
    #     n_col = self.msa.get_alignment_length()
    #     #query_seq_aln = str(RingHomology.get_seqrecord_by_id(msa,query_acc).seq)
    #     #shutil.copyfile('query/2h9r_A1.cif','cif/ens/2h9r_ens.cif')
    #     shutil.copyfile(self.trimmed_pth, out_pth)
    #     # This will be a problem if there is ever a gap in the query
    #     #query_nongap = [res != '-' for res in subject_seq_aln.strip('-')]
    #     for i, r in self.blast_df.iterrows():
    #         acc = r['acc']
    #         subject_seq_aln = str(self.get_seqrecord_by_id(acc).seq)
    #         sbj_nongap = [res != '-' for res in subject_seq_aln.strip('-')]
    #         sbj = r['sseq'].replace('-','')
    #         qry = r['qseq'].replace('-','')

    #         pos_q = self.get_pos(self.query_file, r['qstart'], r['qend'], seq=qry)#r['qseq'].strip('-'))
    #         pos_s = self._trim_cif(f'{self.cifDir}{acc}.cif', r['sstart'], r['send'], seq=sbj)#r['sseq'].strip('-'))
    #         #print(acc)
    #         # print(r['qseq'])
    #         # print(r['sseq'])
    #         # print(r['qstart'], r['qend'])
    #         # print(r['sstart'], r['send'])
    #         # print('sbj_nongap', len(sbj_nongap))
    #         # print('x,y:',len(pos_q), len(pos_s))
    #         # print('#non-gap sbj:', sum(sbj_nongap))
    #         # print('===============')
    #         # print(len(pos_q))
    #         # print('#non-gap sbj:', sum(sbj_nongap))
    #         # #print(len(pos_q[sbj_nongap,:])) # issue: arrays are different len
    #         # print(len(pos_s))
    #         # print('====================')
    #         # # want to ensure that the query and subject we are comparing are the same length
    #         # # we take the 
    #         # # problem is caused by gaps in the query
    #         # print(r['qseq'])
    #         # print(subject_seq_aln)
    #         idx_q = 0
    #         idx_s = 0
    #         pos_q_match = []
    #         pos_s_match = []
    #         while (idx_q <  len(qry)) and (idx_s < len(sbj)):
    #             res_q = r['qseq'][idx_q]
    #             res_s = r['sseq'][idx_s]
    #             if (res_q != '-') and (res_s != '-'):
    #                 pos_q_match.append(pos_q[idx_q,:])
    #                 pos_s_match.append(pos_s[idx_s,:])
    #             idx_q += 1
    #             idx_s += 1
    #         #print(len(pos_q_match))
    #         #print(len(pos_s_match))
    #         pos_q_match = np.array(pos_q_match)
    #         pos_s_match = np.array(pos_s_match)

    #         sup = SVDSuperimposer()
    #         sup.set(pos_q_match, pos_s_match)#(pos_q[sbj_nongap,:], pos_s)
    #         sup.run()
    #         rot_tran = sup.get_rotran()
    #         self._append_alignment(f'cif/trimmed/{acc}.cif', out_pth, rot_tran, i+2, seperate_states)

    def InteractionSummary(self, u, v, verbose = False): 
        if self.prob_df is None:
            self.build_contact_prob_df()
        if self.pos_dict is None:
            self._build_pos_dict()
        inter_type_df = self.prob_df[(self.prob_df['i'] == u) & (self.prob_df['j'] == v)].copy()
        # it does not make sense (nor is it computationally efficient) to calculat prob of NO_INTERACTION for each pair of indicies
        # should be done here.
        accs_no_interaction = self.accs
        for _, inter_type_r in inter_type_df.iterrows():
            contact_type_selected = inter_type_r['contact_type']
            print(f"Interaction: {contact_type_selected} Probability: {inter_type_r['probability']:.3f}, Count: {inter_type_r['count']}")
            accs = list(self.contact_df[(self.contact_df['Interaction'] == contact_type_selected) & \
                                            ((self.contact_df['msaCol1'] == u) & (self.contact_df['msaCol2'] == v) | \
                                            (self.contact_df['msaCol1'] == v) & (self.contact_df['msaCol2'] == u))]['acc'])
            accs_no_interaction = [acc for acc in accs_no_interaction if acc not in accs] # remove acc_ids that have interaction of this contact type
            #print(accs_no_interaction)
            print('AA pair probability: ' + ', '.join([f"{one_to_three(inter[0])}-{one_to_three(inter[1])}: {count / inter_type_r['count']:.3f}" for inter, count in interaction_dict.items()]))
            interaction_dict = {}
            for acc in accs:
                seq_aln_record = self.get_seqrecord_by_id(acc)
                res_u = seq_aln_record[u]
                res_v = seq_aln_record[v]
                AA_inter_pair = res_u + res_v
                interaction_dict[AA_inter_pair] = interaction_dict.get(AA_inter_pair, 0) + 1
                if verbose:
                    if acc != query_acc:
                        blast_r = self.blast_df[self.blast_df['acc'] == acc].iloc[0]
                        sbj_offset_start = blast_r['sstart'] - 1 # subtract one to convert residue number to index
                        #sbj_offset_end = blast_r['send'] - 1 # One indexed to zero indexed
                        #full_seq = blast_r['full_seq']
                    else:
                        sbj_offset_start = 0
                        #full_seq = str(seq_rec_acc.seq).strip('-')
                    u_trimmed_idx = aln_to_seq_idx(str(seq_aln_record.seq), u)
                    v_trimmed_idx = aln_to_seq_idx(str(seq_aln_record.seq), v)
                    idx_full_u, idx_full_v =  sbj_offset_start + u_trimmed_idx, sbj_offset_start + v_trimmed_idx
                    print(f"\t{acc}: {one_to_three(res_u)}-{one_to_three(res_v)}\tRes# (Uniprot): {idx_full_u + 1}, {idx_full_v + 1}, Dist(CA): {np.linalg.norm(self.pos_dict[acc][u_trimmed_idx,:] - self.pos_dict[acc][v_trimmed_idx,:]):.3f}")
                    # Atom position debug
                    # print(f"ATOM POS({u_trimmed_idx},{v_trimmed_idx}) : {self.pos_dict[acc][u_trimmed_idx,:]} {self.pos_dict[acc][v_trimmed_idx,:]} Dist(CA): {np.linalg.norm(self.pos_dict[acc][u_trimmed_idx,:] - self.pos_dict[acc][v_trimmed_idx,:])}")
                    ## INDEX MAPPING DEBUGGING
                    # print(f"{aln_to_seq_idx(str(seq_rec_acc.seq), u) + sbj_offset_start}, {aln_to_seq_idx(str(seq_rec_acc.seq), v) + sbj_offset_end}")
                    #print(str(seq_rec_acc.seq).strip('-'))
                    #print(full_seq[sbj_offset_start-1:sbj_offset_end])
                    #print(res_u, res_v)
                    #print(_one_to_three(res_u), _one_to_three(res_v))
                    #print(u,v, '->', aln_to_seq_idx(str(seq_rec_acc.seq), u), aln_to_seq_idx(str(seq_rec_acc.seq), v))
                    # if acc == query_acc:
                    #     continue
                    #print(len(full_seq), sbj_offset_start, aln_to_seq_idx(str(seq_rec_acc.seq), u))

            
        # calulate prob and count of no contact for pair (u,v)
        count_nc = len(accs_no_interaction)
        prob_nc = count_nc / len(self.accs)
        if len(accs_no_interaction) > 0:
            print(f"Interaction: NO_INTER Probability: {prob_nc:.3f}, Count: {count_nc}")
        inter_type_df = pd.concat([inter_type_df, pd.DataFrame({'i':[u],'j':[v],'contact_type':["NO_INTER"],'probability':[prob_nc],'count':[count_nc]})], ignore_index=True)
        no_interaction_dict = {}
        for acc in accs_no_interaction:
            seq_aln_record = self.get_seqrecord_by_id(acc)
            res_u = seq_aln_record[u]
            res_v = seq_aln_record[v]
            AA_inter_pair = res_u + res_v
            no_interaction_dict[AA_inter_pair] = no_interaction_dict.get(AA_inter_pair, 0) + 1
            if verbose:
                if acc != query_acc:
                    blast_r = self.blast_df[self.blast_df['acc'] == acc].iloc[0]
                    sbj_offset_start = blast_r['sstart'] - 1 # subtract one to convert residue number to index
                    #sbj_offset_end = blast_r['send'] - 1 # One indexed to zero indexed
                    #full_seq = blast_r['full_seq']
                else:
                    sbj_offset_start = 0
                    #full_seq = str(seq_rec_acc.seq).strip('-')
                u_trimmed_idx = aln_to_seq_idx(str(seq_aln_record.seq), u)
                v_trimmed_idx = aln_to_seq_idx(str(seq_aln_record.seq), v)
                idx_full_u, idx_full_v =  sbj_offset_start + u_trimmed_idx, sbj_offset_start + v_trimmed_idx
                print(f"\t{acc}: {one_to_three(res_u)}-{one_to_three(res_v)}\tRes# (Uniprot): {idx_full_u + 1}, {idx_full_v + 1}, Dist(CA): {np.linalg.norm(self.pos_dict[acc][u_trimmed_idx,:] - self.pos_dict[acc][v_trimmed_idx,:]):.3f}")
        print('\t' + ', '.join([f"{one_to_three(inter[0])}-{one_to_three(inter[1])}: {count / len(accs_no_interaction):.3f}" for inter, count in no_interaction_dict.items()]))
        return inter_type_df
    
    def _remove_AF_UNP_desc(self):
        # if a sequence of an Alphafold structures differs from Uniprot sequence, remove it
        # this is a simple solution. Could instead recalculate sstart and send in blast_df and keep sequences that differ.
        # TO BE CALLED BEFORE CONSTRUCTING MSA
        diff_accs = list(self.blast_df[self.blast_df['full_seq'] != self.blast_df['struct_seq']]['acc'])
        if len(diff_accs) > 0:
            self.logger.warning(f"Found Structures with sequence deviating from uniprot {diff_accs}! Removing...")
        for acc in diff_accs:
            self.file_df = self.file_df.loc[self.file_df.index != acc]
            self.blast_df = self.blast_df[self.blast_df['acc'] != acc]
            self.accs = list(self.file_df.index)

### RUNNING FROM COMMANDLINE ###
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        add_help=True,
        description="This package creates homology enriched Residue Interaction Networks for a protein family.\n"+
                    "Outputs network in form of tsv files caputring network nodes and edges. Which may be analyzed here in python, or imported to other programs like cytoscape.",
        epilog="Examples of usage:\n"
               "  python HomologyRing.py -q query.cif -c A -n 64\n"
               "  python HomologyRing.py --help\n\n"
               "For more detailed instructions, refer to the README, documentation, or theis text in /docs."
    )
    arggrp_query = parser.add_argument_group("Query Options")
    arggrp_BLAST = parser.add_argument_group("BLAST Arguments")
    arggrp_struct = parser.add_argument_group("Structure Options")


    # Query information
    arggrp_query.add_argument('-q', '--query_file', type=str, required=True, help="Path of MMCIF file that contains the query chain.")
    arggrp_query.add_argument('-c', '--query_chain', type=str, required=True, help="Chain Identifer of the query used as the basis of the BLAST homology search.")
    arggrp_query.add_argument('--use_label_asym_id', action='store_true', default=False)
    # TODO: state_id is broken
    
    arggrp_query.add_argument('-F', '--family_dir', type=str, help='Path to directory or text file containing paths of cif files. Allows for creation of hRIN for user-defined protein family.\nSpesifying will ignore BLAST argumentes.')

    # BLAST SEARCH
    arggrp_BLAST.add_argument('-E', '--E_value', type=float, help='E-value cutoff for BLAST search.')
    arggrp_BLAST.add_argument('-n', '--max_results', type=int, default=100, help='Maximum number of results to be considered in a faimily. Takes the n results with the best E-value. Value of -1 takes all results below E-value threshold.')
    arggrp_BLAST.add_argument('-r', '--remote', action='store_true')
    # BLAST DB options
    arggrp_BLAST.add_argument('-db' '--blast_db', type=str, help="Path of compiled BLAST peptide database. See https://ftp.ncbi.nlm.nih.gov/blast/documents/blastdb.html for details.")
    arggrp_BLAST.add_argument('-p', '--use_pdb_ids', action='store_true', help='Indicates if accentions in provided BLAST DB are PDB IDs. Default is UNP accs.')
    
    # Structure Options
    arggrp_struct.add_argument('-s', '--struct_source', choices=['AlphaFold', 'PDB', 'AlphaFill'], default='AlphaFold', help="The name of the source from which matched homologs should be downloaded.")
    arggrp_struct.add_argument('-f', '--force_download', action='store_true', help="Re-download existing MMCIF structure files that already exist locally.")
    arggrp_struct.add_argument('-R', '--force_ring', '--force_RING', action='store_true', help="Re-run RING on structures, even if the RING output files already exist locally.")
    parser.add_argument('-N', '--normalization_method', type=str)
    #parser.add_argument('-m', '--edge_multiplicity')
    args = parser.parse_args()

    print(args.family_dir)

    rh = HomologyRing(
        query_file=args.query_file, 
        blast_DB=args.blast_db, 
        struct_source=args.struct_source, 
        uses_PDB_ids=args.use_pdb_ids
    )

    if args.family_dir is None:
        ## Normal hRIN construction with BLAST search
        rh.build(
            chain_id=args.query_chain,
            max_results=args.max_results,
            eval=args.E_value,
            remote_blast=args.remote,
            force_download=args.force_download,
            force_RING=args.force_ring,
            use_label_asym_id=args.use_label_asym_id,
            prob_normalization=args.normalization_method
        )
    else: 
        ## Build hRIN for a user-defined family
        rh.build_fromFamily(
            family=args.family_dir,
            chain_id=args.query_chain,
            force_RING=args.force_ring,
            use_label_asym_id=args.use_label_asym_id,
            prob_normalization=args.normalization_method
        )
