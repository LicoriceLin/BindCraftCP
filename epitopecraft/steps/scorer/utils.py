from Bio.PDB import PDBParser
from Bio.Data import PDBData
import numpy as np

def pdb2seq(pdbfile:str,chain:str='B'):
    pdb=PDBParser(QUIET=True).get_structure('tmp',pdbfile)[0][chain]
    seq_=[]
    for residue in pdb.get_residues():
        seq_+=PDBData.protein_letters_3to1.get(residue.get_resname(),'X')
    return ''.join(seq_)

def ca_bfactor(pdbfile:str,chain:str='B'):
    pdb=PDBParser(QUIET=True).get_structure('tmp',pdbfile)[0][chain]
    plddt=[]
    for residue in pdb.get_residues():
        plddt.append(residue['CA'].bfactor)
    return np.array(plddt)
