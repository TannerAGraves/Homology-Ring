import os
import re
import requests
from collections import Counter
import numpy as np
import pandas as pd
from matplotlib import colors
import networkx as nx
import subprocess
from Bio import SeqIO
from Bio.SeqUtils import IUPACData
from Bio.PDB.Polypeptide import three_to_one
from Bio import AlignIO
from Bio.Align import AlignInfo
from Bio.Align import MultipleSeqAlignment
from pdbecif.mmcif_io import CifFileReader

query_acc = 'QUERY' # handle to identify the the query structure and corresponding sequence

contact_colors = {
            'HBOND': 'tab:blue',
            'VDW': 'tab:gray',
            'IONIC': 'tab:red',
            'PIPISTACK': 'tab:orange',
            'PICATION': 'tab:green',
            'PIHBOND': 'yellow',
            'SSBOND': 'tab:purple',
            'METAL_ION': 'tab:pink',
            'HALOGEN': 'tab:cyan',
            'IAC': 'tab:brown'
        }

all_interaction_types = list(contact_colors.keys())

contact_order = {inter_type: i for i, inter_type in enumerate(contact_colors.keys())} # used for consistent sorting by interaction type

# reference: https://www.jalview.org/help/html/colourSchemes/clustal.html
clustal_colors = {
        'G': colors.to_hex("tab:orange"), # '#ff0000', # Glycine - red
        'P': '#ffff00', # '#ff0000', # Proline - red
        'S': colors.to_hex("tab:green"),  # '#ff00ff', # Serine - pink
        'T': colors.to_hex("tab:green"),  # '#ff00ff', # Threonine - pink
        'H': colors.to_hex("tab:cyan"),  # '#ff00ff', # Histidine - pink
        'A': colors.to_hex("tab:blue"),  # '#ff9900', # Alanine - orange
        'V': colors.to_hex("tab:blue"),  # '#ff9900', # Valine - orange
        'M': colors.to_hex("tab:blue"),  # '#ff9900', # Methionine - orange
        'C': colors.to_hex("tab:pink"), # Cysteine - yellow
        'F': colors.to_hex("tab:green"),  # '#00ff00', # Phenylalanine - green
        'Y': colors.to_hex("tab:cyan"),  # '#00ff00', # Tyrosine - green
        'W': colors.to_hex("tab:blue"),  # '#00ff00', # Tryptophan - green
        'L': colors.to_hex("tab:blue"),  # '#00ff00', # Leucine - green
        'I': colors.to_hex("tab:blue"),  # '#00ff00', # Isoleucine - green
        'N': colors.to_hex("tab:green"),  # '#00ffff', # Asparagine - cyan
        'Q': colors.to_hex("tab:green"),  # '#00ffff', # Glutamine - cyan
        'K': colors.to_hex("tab:red"),  # '#0000ff', # Lysine - blue
        'R': colors.to_hex("tab:red"),  # '#0000ff', # Arginine - blue
        'D': colors.to_hex("tab:purple"),  # '#ff00ff', # Aspartic Acid - magenta
        'E': colors.to_hex("tab:purple"),  # '#ff00ff', # Glutamic Acid - magenta
        '-': '#ffffff', # Gap - white,
        'X': '#000000' # missing - Black
    }

def init_file_structure(logger=None):
    output_pths = [
        "cif",
        "ring_log",
        "ring_out",
        "ring_out/AlphaFill",
        "ring_out/AlphaFill/label_asym_id",
        "ring_out/AlphaFold",
        "ring_out/AlphaFold/label_asym_id",
        "ring_out/PDB",
        "ring_out/PDB/label_asym_id"
    ]
    for pth in output_pths:
        if not os.path.exists(pth):
            os.makedirs(pth)
            if logger is not None:
                logger.info(f"Created non-existant path: {pth}")

def check_ring_version(use_label_asym_id):
    """
    Check that ring is installed
    """
    command = ['ring', '--version']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    version, stderr = process.communicate()
    version = version.decode()
    match = re.search(r'v(\d+)\.(\d+)-(\d+)', version)
    if match:
        major, minor, patch = map(int, match.groups())
        is_sufficient = (major, minor, patch) >= (4, 0, 7)
        if (not is_sufficient) and use_label_asym_id:
            raise Exception('Filtering by label_asym_id only supported by ring 4.0-7 and higher.')
        return 1
    else: # version number not found
        raise Exception("Pipeline requires local installation of RING.")
        #print('Warning: ring has been updated. Stability may suffer.')


def get_full_sequences(accs, blast_db, acc_idx, sep = '|'):
    # not sure this is even used
    # accs = list(self.blast_df['acc'])
    accs_lower = [str.lower(acc) for acc in accs]
    acc_seq_dict = {}
    for record in SeqIO.parse(blast_db, "fasta"):
        r_acc = str.lower(record.id.split(sep)[acc_idx])
        if r_acc in accs_lower:
            acc_seq_dict[r_acc] = str(record.seq)
    # self.blast_df['full_seq'] = 
    return [acc_seq_dict[acc] for acc in accs_lower]

def filter_sequences(self, to_remove, reason = None):
    """Removes entire structures (all chains) from list of homologs

    Args:
        to_remove (_type_): _description_
        reason (_type_, optional): Reason for removal - used for logging. Defaults to None.
    """
    self.logger.warning(f"Filtering {to_remove}: {reason}")
    if self.verbose:
        print(f'Filtered {to_remove}. Reason: {reason}')
    self.file_df = self.file_df[self.file_df['acc'] != to_remove].reset_index(drop=True)
    self.blast_df = self.blast_df[self.blast_df['acc'] != to_remove].reset_index(drop=True)
    if self.accs is not None:
        self.accs = [acc for acc in self.accs if acc != to_remove]
    if self.msa is not None:
        records = [rec for rec in self.msa if to_remove not in rec.id]
        new_msa = MultipleSeqAlignment(records)
        if len(new_msa) != self.msa:
            self.msa = new_msa
            AlignIO.write(new_msa, "msa.fasta", "fasta")

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def make_fainter(rgb, alpha):
    return tuple(c + (255.0 - c) * alpha for c in rgb)

def make_darker(rgb, alpha):
    return tuple(c * (1 - alpha) for c in rgb)

## for heatmap discrete colorscale CLUSTAL collors in app
res_color_dict = {str.lower(res):rgb_to_hex(make_fainter(hex_to_rgb(hex), 0.8)) for res, hex in clustal_colors.items()}
res_color_dict.update(clustal_colors)
def _discrete_colorscale(bvals, colors):
    """
    From: https://community.plotly.com/t/colors-for-discrete-ranges-in-heatmaps/7780/3
    This is used as part of a workaround for px.imshow not having the nice zoom methods that other plots have. Using a heatmap avoids this but need to define a custom discrete color scale to keep CLUSTAL colors
    bvals - list of values bounding intervals/ranges of interest
    colors - list of rgb or hex colorcodes for values in [bvals[k], bvals[k+1]],0<=k < len(bvals)-1
    returns the plotly  discrete colorscale
    """
    if len(bvals) != len(colors)+1:
        raise ValueError('len(boundary values) should be equal to  len(colors)+1')
    bvals = sorted(bvals)     
    nvals = [(v-bvals[0])/(bvals[-1]-bvals[0]) for v in bvals]
    dcolorscale = []
    for k in range(len(colors)):
        dcolorscale.extend([[nvals[k], colors[k]], [nvals[k+1], colors[k]]])
    return dcolorscale  
res_color_idx_dict = {res:i for i, res in enumerate(res_color_dict.keys())} # get the index (colorscale code) from res (with or without contact)
plotly_heatmap_colorscale = _discrete_colorscale(list(np.arange(0,len(res_color_idx_dict)+1)), list(res_color_dict.values()))

def impute_zeros(arr): # used for giving nodes in network positions that correspond to gaps in the query sequence
    nrows, ncols = arr.shape
    for col in range(ncols):
        non_zero_indices = np.nonzero(arr[:, col])[0]
        if len(non_zero_indices) == 0:
            continue
        extended_indices = np.concatenate(([-1], non_zero_indices, [nrows]))
        extended_values = np.concatenate(([arr[non_zero_indices[0], col]], arr[non_zero_indices, col], [arr[non_zero_indices[-1], col]]))
        for i in range(len(extended_indices) - 1):
            start_idx = extended_indices[i] + 1
            end_idx = extended_indices[i + 1]
            num_zeros = end_idx - start_idx
            if num_zeros > 0:
                start_val = extended_values[i]
                end_val = extended_values[i + 1]
                imputed_values = np.linspace(start_val, end_val, num_zeros + 2)[1:-1]
                arr[start_idx:end_idx, col] = imputed_values   
    return arr

def int_to_letters(i):
    if i < 0:
        return None
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if i < 26:
        return letters[i]
    else:
        return int_to_letters((i // 26) - 1) + letters[i % 26]
    
def one_to_three(letter):
    return IUPACData.protein_letters_1to3.get(letter.upper(), 'UNK')

def dumber_consensus(msa):
    consensus = ''
    for col_num in range(msa.get_alignment_length()):
        residues = re.sub(r'-', '', AlignInfo.SummaryInfo(msa).get_column(col_num))
        res_count = Counter(residues)
        max_count = max(res_count.values())
        for res in residues:
            if res_count[res] == max_count:
                consensus += res
                break
    return consensus

standard_AA_one = list('ACDEFGHIKLMNPQRSTVWY')
standard_AA_three = [str.upper(one_to_three(res)) for res in standard_AA_one]

def aln_to_seq_idx(seq_aln, idx_aln):
    return sum([res != '-' for res in seq_aln[:idx_aln]])

# Map for converting sequence positions to MSA cols
def build_index_mapping(aligned_seq): # DEPRICATED -- RENAMED REMOVE LATER
    mapping = {}
    orig_index = 0
    for msa_index, char in enumerate(aligned_seq):
        if char != '-':
            mapping[orig_index] = int(msa_index)
            orig_index += 1
    return mapping

def build_seqId_to_alnIdx_map(aligned_seq):
    """
    Returns a dictionary that serves as a map between residue ids (1-indexed position)
    in contigous AA sequence to its corresponding index (0-indexed) in an aligned sequence.
    For example:
    Contigous: 'ABC'
    Gapped: '--A-BC'
    id 2 corresponds to 'B' in contig seq and maps to idx 4 in gapped sequence.
    Assumes that contig and gapped sequences is the same AA sequence.
    """
    mapping = {}
    orig_index = 1
    for msa_index, char in enumerate(aligned_seq):
        if char != '-':
            mapping[orig_index] = int(msa_index)
            orig_index += 1
    return mapping

def build_index_maps(self):
    map_dict = {}
    for i, r, in self.blast_df.iterrows():
        try:
            acc = r['acc_chn']
        except:
            acc = r['acc']
        s_record = self.get_seqrecord_by_id(acc)
        try:
            s_aln = s_record.seq
        except:
            print(acc)
        map_dict[acc] = build_index_mapping(s_aln)
    return map_dict

def trim_msa(msa, trim):
    n_row, n_col = len(msa), msa.get_alignment_length()
    use_custom_xtick = False
    if trim == False:
        # img_slice = img
        l, r = 0, n_col - 1
    elif isinstance(trim, tuple) and (len(trim) == 2) and all(isinstance(x, int) for x in trim):
        l, r = trim
        r = min(n_col - 1, r + 1) # right bound is inclusive as passed
        #img_slice = img[:, x_start:x_end]
        use_custom_xtick = True
    elif isinstance(trim, float) and 0.0 < trim <= 1.0:
        # trim tails by col occupancy
        found_left = False
        found_right = False
        
        l, r = 0, n_col - 1
        while not (found_left and found_right):
            if not found_left:
                l_col = msa[:,l]
                l_occupancy = sum(res != '-' for res in l_col) / n_row
                found_left = l_occupancy >= trim
                l += not found_left
            if not found_left:
                r_col = msa[:,r]
                r_occupancy = sum(res != '-' for res in r_col) / n_row
                found_right = r_occupancy >= trim
                r -= not found_right
            if l >= r:
                raise Exception(f"Failed to trim ends to occupancy {trim}.")
        use_custom_xtick = True
    else: 
        l,r = None, None
        raise Exception("trim may be False for full alignmentm, float in (0,1] to trim tails by occupancy, or tuple of ints.")
    return l, r, use_custom_xtick


def contains_non_standard_aa(sequence):
    """
    Check if the amino acid sequence contains non-standard residues.
    Standard residues are A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y, and '-' for gaps.
    Non-standard residues will be any character not in the above list.
    """
    # Regex to find non-standard amino acid residues
    if re.search(r'[^ACDEFGHIKLMNPQRSTVWY-]', sequence):
        return True
    return False

def get_alphafold(self, accs, force=False, out_dir='cif/AlphaFold/'):
    """Download AlphaFold predictions for given accessions."""
    # old api: https://alphafold.com/api/prediction/
    return general_download(self, accs, source_url='https://alphafold.ebi.ac.uk/files/', file_suffix='.cif', out_dir=out_dir, force=force, AF_API_URL=True)

def get_alphafill(self, accs, force=False, out_dir='cif/AlphaFill/'):
    """Download AlphaFill structures for given accessions."""
    # note that the dest_url for alphafill structure (cif) files does not end in '.cif' so nothing is passed here.
    return general_download(self, accs, source_url='https://alphafill.eu/v1/aff/', file_suffix='', out_dir=out_dir, force=force)

def get_pdb(self, accs, force=False, out_dir = 'cif/PDB/'):
    """Download PDB structures using the general download method."""
    return general_download(self, accs, source_url='https://files.rcsb.org/download/', file_suffix='.cif', out_dir=out_dir, force=force)


def general_download(self, accs, source_url, file_suffix='.cif', out_dir='cif/', force=False, timeout=4, AF_API_URL=False):
    status = {}
    for acc in accs:
        file_path = f'{out_dir}{acc}{file_suffix}'
        if not force and os.path.exists(file_path):
            status[acc] = "Exists"
            self.file_df.loc[self.file_df['acc'] == acc, 'struct_pth'] = file_path
            continue
        try:
            request_url = f'{source_url}AF-{acc}-F1-model_v4{file_suffix}' if AF_API_URL else f'{source_url}{acc}{file_suffix}'
            response = requests.get(request_url, timeout=timeout)  # Using a common timeout for all requests
            if response.status_code == 200:
                with open(file_path, 'wb') as file:
                    file.write(response.content)
                #self.file_df.loc[acc,'struct_pth'] = file_path
                self.file_df.loc[self.file_df['acc'] == acc, 'struct_pth'] = file_path
                status[acc] = "Success"
            else:
                status[acc] = f"Failed: HTTP {response.status_code}"
                self.logger.warning(f"Failed to download file for {acc}. HTTP status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            status[acc] = f"Failed: {str(e)}"
            self.logger.error(f"Error processing download for {acc}: {str(e)}")
    return status


def seq_from_cif_new(mmcif_path, target_chain_id='A', seq_id_source = 'auth', asym_id_source = 'auth', state_id = '1'):
    """
    Responsible for extracting the polypeptide sequence of given chain from an mmCIF file.
    Returns AA sequcne and indexMap dictionary.
    The chain sequece can be gotten from other parts of of the mmCIF file, but _atom_site section traversal is needed for creating the index map.
    indexMap_dict: dictionary where the keys are a seq_id from the structure file
        this can be label_seq_id or auth_seq_id depending on the setting of seq_id.
        Values are 1-indexed ids in the contiguous AA sequence
    target_chain_id: The identifier of the desired chain, or 'asym_id'. Example, 'B' or 'AC'.
    seq_id: Which 
    """
    # with open(mmcif_path, 'r') as file:
    #     content = file.readlines()
    cfr = CifFileReader()
    cif_obj = cfr.read(mmcif_path)
    acc, cif_obj = next(iter(cif_obj.items())) # cif can (technically) contain many structures, get the first
    poly_seq_df = pd.DataFrame(cif_obj['_pdbx_poly_seq_scheme'])
    # pdb_strand_id is auth_asym_id (?) Needs to be use reguardless of asym_id_source, because blast_df file lists auth_asym_id
    poly_chain_id_col = 'pdb_strand_id' if asym_id_source == 'auth' else 'asym_id'
    poly_seq_df = poly_seq_df[poly_seq_df[poly_chain_id_col] == target_chain_id] # filter target chain
    # pdb includes residues from uniprot gene not represented in structure (I think that is whats happening)
    # removes trailing residues not represented in structure
    poly_seq_df = poly_seq_df[poly_seq_df['auth_seq_num'] != '?'] 
    seq_keys = list(poly_seq_df['seq_id'].astype(int))
    seq_slice = [idx - 1 for idx in seq_keys]

    target_entity_id = poly_seq_df['entity_id'].unique()
    if len(target_entity_id) != 1:
        raise Exception("Given chain mapped to incorrect number of entities. {target_entity_id}")
    target_entity_id = target_entity_id[0]
    
    # ent = cif_obj['_entity']
    # for key in ent: # make sure it's a dataframe if there is only one polymer 
    #     if not isinstance(ent[key], list):
    #         ent[key] = [ent[key]]
    # ent_df = pd.DataFrame(ent)

    ent_poly = cif_obj['_entity_poly']
    for key in ent_poly: # make sure it's a dataframe if there is only one polymer 
        if not isinstance(ent_poly[key], list):
            ent_poly[key] = [ent_poly[key]]
    ent_poly_df = pd.DataFrame(ent_poly)
    # use the _entity_poly section to get the seqence with one letter codes with modified residues already mapped to base residues
    # to find the correspinding entry, only need the entity id
    


    #ent_df = ent_df.merge(ent_poly_df,how='left', left_on='id', right_on='entity_id')# need the entity section because ent_poly does not have information about auth_asym_id
    seq_ent_poly = (ent_poly_df.loc[ent_poly_df['entity_id'] == target_entity_id].iloc[0])['pdbx_seq_one_letter_code_can']
    seq_ent_poly = re.sub(r'\s+', '', seq_ent_poly)
    seq = ''.join([seq_ent_poly[i] for i in seq_slice])
    #seq_poly = ent_poly_df['pdbx_seq_one_letter_code_can']
    #seq_series = ent_poly_df['pdbx_seq_one_letter_code_can'].reset_index()
    #seq_seq_poly = ''.join(seq_series)
    #seq_series.index = seq_series.index + 1 # for some reason I am using one index for the map, even though node_ids are zero-indexed
    #idx_dict = seq_series.to_dict()
    idx_dict = dict(zip(seq_keys, list(range(1, len(seq_keys) + 1))))#{idx: res for idx, res in zip(seq_keys, seq)}
    has_unk = 'X' in seq_ent_poly
    if len(seq) != len(poly_seq_df):
        raise Exception(f"sequence length mismatch! _pdbx_poly_seq_scheme: {len(poly_seq_df)}, _entity_poly: {len(seq)}")
    return seq_ent_poly, idx_dict, has_unk

def seq_from_cif(mmcif_path, target_chain_id='A', seq_id_source = 'auth', asym_id_source = 'auth', state_id = '1'):
    """
    Responsible for extracting the polypeptide sequence of given chain from an mmCIF file.
    Returns AA sequcne and indexMap dictionary.
    The chain sequece can be gotten from other parts of of the mmCIF file, but _atom_site section traversal is needed for creating the index map.
    indexMap_dict: dictionary where the keys are a seq_id from the structure file
        this can be label_seq_id or auth_seq_id depending on the setting of seq_id.
        Values are 1-indexed ids in the contiguous AA sequence
    target_chain_id: The identifier of the desired chain, or 'asym_id'. Example, 'B' or 'AC'.
    seq_id: Which 
    """
    with open(mmcif_path, 'r') as file:
        content = file.readlines()
    
    found_atom_header = False
    header_idx = 0
    header_dict = {}
    seq = ''
    seq_id_list = [] # collect the sequence of seq_ids in the structure
    prev_seq_id = None

    #idx_dict = {}
    has_unk = False # accs that have missing/unknown residue

    for line in content:
        if line.startswith('_atom_site.'):
            found_atom_header = True
            col_name = line.split('.')[1]
            header_dict[str.strip(col_name)] = header_idx
            header_idx += 1
        elif found_atom_header:
            if ('ATOM' in line.split()[0]):
                line_list = line.split()
                line_dict = dict(zip(header_dict.keys(), line.split()))
                # in the case that the CIF has multiple states, discard all information but the desired state.
                if line_dict['pdbx_PDB_model_num'] != state_id:
                    continue
                if line_dict[f'auth_asym_id'] == target_chain_id: # use auth because enteries in blast_db refer to auth_chain ids
                    seq_id = int(line_dict[f'{seq_id_source}_seq_id'])
                    if seq_id != prev_seq_id:
                        res = line_dict[f'{seq_id_source}_comp_id']
                        if res =='UNK': # code for unknown residue
                            seq += 'X'
                            has_unk = True
                        elif res not in standard_AA_three:
                            1 # debug
                            raise Exception(f"Unknown redidue code in {mmcif_path} auth chain {line_dict[f'{asym_id_source}_asym_id']} at auth_seq_id {line_dict['auth_seq_id']}: {res}")
                        else:
                            seq += three_to_one(line_dict[f'{seq_id_source}_comp_id'])
                            seq_id_list.append(seq_id)
                    prev_seq_id = seq_id
            if 'loop_' in line:
                break # in the next section, done parcing ATOM info
        else:
            continue
    # convert the list of seq_ids to a dict to be used as a map
    # map: struct_seq_id (auth/label) -> continuous AA seq id starting with 1
    idx_dict = dict(zip(seq_id_list, list(range(1, len(seq_id_list) + 1))))
    return seq, idx_dict, has_unk

def get_struct_seqs(self, seq_id_source='auth', asym_id_source='auth'):
    """
    Adds the PP sequence as it exists in the mmCIF file to the blast_df
    """
    self.blast_df['struct_seq'] = None
    accs = []
    struct_seqs = []
    idx_dict = {}
    col_has_unk = []
    for i,r in self.file_df.iterrows():
        acc = r['acc']
        acc_chn = r['acc_chn'] #query_acc if (acc == query_acc) else r['acc_chn']
        chain = r['pdb_chain']
        seq, label_to_seq_id_map, has_unk = seq_from_cif(r['struct_pth'], target_chain_id=chain, seq_id_source=seq_id_source, asym_id_source=asym_id_source)
        col_has_unk.append(has_unk)
        idx_dict[acc_chn] = label_to_seq_id_map
        accs.append(acc)
        struct_seqs.append(seq)
        self.blast_df.loc[(self.blast_df['acc']==acc) & (self.blast_df['pdb_chain']==chain), 'struct_seq'] = seq
    self.file_df['has_unk_res'] = col_has_unk # record which structures have unknown residues
    return idx_dict
        

def process_AFill_cif(acc, pth, q_chain, q_state, asym_id_source = 'auth', filter_chain = True):
    ATOM_header = ['acc']
    rows = []
    with open(pth,'r') as src:
        for l in src:
            if l.startswith('_atom_site.'): # ATOM header
                ATOM_header.append(l.split('.')[1].strip())
            if l.startswith('ATOM') or l.startswith('HETATM'): # ATOM data
                l_arr = [acc] + l.split()
                rows.append(l_arr)
    cur_df = pd.DataFrame(rows, columns=ATOM_header)
    if filter_chain:#acc == '_QUERY': 
        # this wastes time but is neater than creating handels for col values while parsing
        cur_df = cur_df[cur_df['pdbx_PDB_model_num'] == str(q_state)]
        cur_df = cur_df[(cur_df['group_PDB'] == 'HETATM') | (cur_df[f'{asym_id_source}_asym_id'] == q_chain)]
    return(cur_df)

def get_pos1(atom_df, acc, start = None, end = None, only_CA = True):
    pos = atom_df[atom_df['acc'] == acc]
    if only_CA:
        pos = pos[pos['label_atom_id'] == 'CA']
        # CA atoms cannot belong to ligands, so will have sequence IDs
        # We wanyt to filter by seq_id, but missing values in ligands makes
        pos['label_seq_id'] = pos['label_seq_id'].astype(str).astype(int)
        pos = pos[(start is None) | (pos['label_seq_id'] >= start) &\
              (end is None) | (pos['label_seq_id'] <= end)]
    return pos

def build_transforms(self):
    from Bio.SVDSuperimposer import SVDSuperimposer
    q_df = self.atom_df[(self.atom_df['acc'] == query_acc) & (self.atom_df[f'{self.asym_id_source}_asym_id'] == self.q_chain) & (self.atom_df['msa_col'].notna())][['Cartn_x', 'Cartn_y', 'Cartn_z', 'msa_col']]
    self.rot_tran_dict = {}
    #mats = []
    for i, r in self.file_df.iterrows():
        if r['acc'] == query_acc:
            continue
        sbj_chn = r['pdb_chain']
        sbj_acc = r['acc']
        sbj_df = self.atom_df[(self.atom_df['acc'] == sbj_acc) & (self.atom_df['auth_asym_id'] == sbj_chn) & (self.atom_df['msa_col'].notna())][['Cartn_x', 'Cartn_y', 'Cartn_z', 'msa_col']]
        merged_df = q_df.merge(sbj_df, on='msa_col', how='inner')
        q_pos = merged_df[['Cartn_x_x', 'Cartn_y_x', 'Cartn_z_x']].to_numpy(dtype = float)
        sbj_pos = merged_df[['Cartn_x_y', 'Cartn_y_y', 'Cartn_z_y']].to_numpy(dtype = float)
        if not len(merged_df):
            rot, tran = np.eye(3), np.zeros(3)
        else:
            sup = SVDSuperimposer()
            sup.set(q_pos, sbj_pos)
            sup.run()
            rot, tran = sup.get_rotran()
        self.rot_tran_dict[r['acc_chn']] = {
            'rot': rot,
            'tran': tran
        }
        # mats.append({
        #     'acc':sbj_acc,
        #     'pdb_chain': sbj_chn, # file_df always uses auth_asym_id
        #     'acc_chn': r['acc_chn'], # useful for merging on one key
        #     'rot': rot,
        #     'tran': tran
        # })
    #self.transform_df = pd.DataFrame(mats) # don't think I need info in df form

def check_ring_msa_parody(rh, edge_df):
    raise_ex = False
    bad_idx = []
    for i, r in edge_df.iterrows(): 
        acc = r['acc']
        msa_idx1, msa_idx2 = r['msaCol1'], r['msaCol2']
        if msa_idx1 and msa_idx2:
            msa_idx1, msa_idx2 = int(msa_idx1), int(msa_idx2)
            edg_res1 = three_to_one(r['NodeId1'].split(':')[-1])
            edg_res2 = three_to_one(r['NodeId2'].split(':')[-1])
            rec = rh.get_seqrecord_by_id(acc)
            edg_res1 = rec[msa_idx1]
            msa_res2 = rec[msa_idx2]
            if (edg_res1 != edg_res1) or (edg_res2 != msa_res2):
                raise_ex = True
                bad_idx.append({'acc': acc, 'u': msa_idx1, 'v': msa_idx2, 'res_u': edg_res1, 'res_v': edg_res2})
    if raise_ex:
        return pd.DataFrame(bad_idx)
        #raise Exception("Residue mapping mismatch", bad_idx)
    else:
        print("Passed.")
    return

def check_blast_struct_parody(rh, atom_df):
    raise_ex = False
    for i, r in rh.blast_df.iterrows():
        acc = r['acc']
        uni_seq = r['full_seq']
        ## Hack solution, but groupby is giving unexpected results
        seq_dict = {}
        for i,r in atom_df[(atom_df['acc'] == acc) & (atom_df['group_PDB'] == 'ATOM')].iterrows():
            seq_dict[r['label_seq_id']] = three_to_one(r['label_comp_id'])
            if r['label_seq_id'] != r['pdbx_sifts_xref_db_num']:
                print("AF / uniprot index disparity")
                print(acc, i, r['label_seq_id'], r['pdbx_sifts_xref_db_num'])
                raise_ex = True
        str_seq = ''.join(list(seq_dict.values()))
        #str_seq = ''.join(list(atom_df[(atom_df['acc'] == acc) & (atom_df['group_PDB'] == 'ATOM')].sort_values('label_seq_id').groupby('label_seq_id').first()['pdbx_sifts_xref_db_res']))
        if uni_seq != str_seq:
            raise_ex = True
            print(acc)
            print('UNI',uni_seq, len(uni_seq))
            print('STR', str_seq, len(str_seq))
    if raise_ex:
        raise Exception('Found differences between sequence in blastDB/structure')
    else:
        print('Passed.')
    return

def interp_node_pos(self, q_id, MG = None, G_dict = None):
    """Used to interpolate positions of nodes (MSA cols) that may not be present in the query sequence.
    The resulting positions are used for plotting.

    Args:
        self (RingHomology instance)
        q_id (int): id of query node
        MG (NetworkX.MultiGraph, optional): _description_. Defaults to None, which uses 
        G_dict (_type_, optional): _description_. Defaults to None.

    Raises:
        Exception: _description_
    """
    # use the class attributes for graph/dict of graphs if none given
    MG = self.MultiGraph if MG is None else MG
    G_dict = self.Graph_inter_dict if G_dict is None else G_dict
    inters = list(G_dict.keys())

    u_id = q_id
    v_id = q_id
    found_left = False
    found_right = False
    while not found_left:
        u_id = u_id - 1
        u_node = MG.nodes.get(u_id, {})
        if u_node.get('x') is not None:
            found_left = True
            u_x, u_y, u_z = u_node.get('x'), u_node.get('y'), u_node.get('z')
        elif u_id <= 0:
            break
    while not found_right:
        v_id = v_id + 1
        v_node = MG.nodes.get(v_id, {})
        if v_node.get('x') is not None:
            found_right = True
            v_x, v_y, v_z = v_node.get('x'), v_node.get('y'), v_node.get('z')
        elif v_id >= self.msa.get_alignment_length():
            break
    if found_left and found_right:
        interp_nodes = np.arange(u_id, v_id + 1) # add one for including endpoint
        interp_x = np.linspace(u_x, v_x, len(interp_nodes))
        interp_y = np.linspace(u_y, v_y, len(interp_nodes))
        interp_z = np.linspace(u_z, v_z, len(interp_nodes))
        pos_info = {node: {'x': x, 'y': y, 'z': z} for node, x, y, z in zip(interp_nodes, interp_x, interp_y, interp_z)}
        #for n_id in interp_nodes:
        nx.set_node_attributes(MG, pos_info)
        for inter in inters: # update the interaction spesific graphs
            nx.set_node_attributes(G_dict[inter], pos_info)
    elif found_left ^ found_right:
        # keep searching left to get direction and spacing
        additional_steps = 0
        found_next = False
        if found_left: # found left, but not right pos info
            id_inc = -1
            last_x, last_y, last_z = u_x, u_y, u_z
            last_id = u_id
        else: # foung right, but not left pos info
            id_inc = 1 # keep searching right to get direction
            last_x, last_y, last_z = v_x, v_y, v_z
            last_id = v_id
        next_id = last_id
        while not found_next:
            next_id += id_inc
            additional_steps += 1
            next_node = MG.nodes.get(next_id, {})
            if next_node.get('x') is not None:
                found_next = True
                next_x, next_y, next_z = next_node.get('x'), next_node.get('y'), next_node.get('z')
                del_x, del_y, del_z = next_x - last_x, next_y - last_y, next_z - last_z
                del_x /= additional_steps
                del_y /= additional_steps
                del_z /= additional_steps
        cur_id = last_id
        step = 0
        while (cur_id > 0) and (cur_id < self.msa.get_alignment_length() - 1):
            cur_id -= id_inc # not searching but filling in missing position info at tail
            step += 1
            interp_x, interp_y, interp_z = last_x + step * del_x, last_y + step * del_y, last_z + step * del_z
            MG.nodes[cur_id]['x'] = interp_x
            MG.nodes[cur_id]['y'] = interp_y
            MG.nodes[cur_id]['z'] = interp_z
            for inter in inters:
                self.Graph_inter_dict[inter].nodes[cur_id].update(
                    {
                        'x':interp_x,
                        'y':interp_y,
                        'z':interp_z
                    }
                )
    else:
        raise Exception('No node position information found.')