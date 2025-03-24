import json
from typing import Literal
from ast import literal_eval
import glob
import os
# from pathlib import Path
from pathlib import PosixPath as Path
from tqdm import tqdm
from subprocess import run

from typing import Iterable,Union,Callable,Generator,List,Dict,Tuple,Any
from tempfile import TemporaryDirectory
import pickle as pkl
import json
from ast import literal_eval
from collections.abc import Iterable as collections_Iterable
# from itertools import tee

import math
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from statannotations.Annotator import Annotator

import Bio.PDB as BP
from Bio.PDB import PDBParser
from Bio.PDB.Entity import Entity
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.SASA import ShrakeRupley
from Bio.Data import PDBData

from pymol import cmd