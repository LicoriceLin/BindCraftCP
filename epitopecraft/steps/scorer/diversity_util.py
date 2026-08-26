import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path
import shutil
import subprocess
import tempfile
from collections.abc import Sequence

from sklearn.cluster import SpectralClustering

def _simple_identity(x:str,y:str):
    m=0
    for i,j in zip(x,y):
        if i==j:
            m+=1
    return m/len(x)
    
def simple_diversity(df:pd.DataFrame):
    '''
    same-length simple diversity
    '''
    odf=pd.DataFrame(columns=df.index,index=df.index)
    for i, j in combinations(df['sequence'].index, 2):
        odf.at[i, j] = odf.at[j, i] = 1/(_simple_identity(df['sequence'][i], df['sequence'][j])+ 1e-3)
    odf=odf.fillna(1.).infer_objects(copy=False)
    return odf

def cluster_and_get_medoids(distance_matrix, num_clusters=5):
    spectral = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', random_state=42)
    labels = spectral.fit_predict(distance_matrix)

    medoids = []
    for cluster_id in range(num_clusters):
        cluster_points = np.where(labels == cluster_id)[0]  
        medoids.append(cluster_points[0])

    return medoids, labels

def mmseqs2_diversity(
    df: pd.DataFrame,
    min_seq_id: float = 0.3,
    *,
    coverage: float = 0.8,
    cov_mode: int = 0,
    cluster_mode: int = 0,
    sensitivity: float = 4.0,
    mmseqs_executable: str = "mmseqs",
    threads: int = 1,
    temp_dir: str | None = None,
    extra_args: Sequence[str] = (),
) -> pd.DataFrame:
    """Cluster protein sequences with MMseqs2 ``easy-cluster``.

    The input follows :func:`simple_diversity`: rows are designs and the
    ``sequence`` column contains protein sequences. The number of clusters is
    determined by MMseqs2 from ``min_seq_id``, coverage, and clustering mode.

    Returns a copy of ``df`` with four additional columns:
    ``mmseqs_cluster_id``, ``mmseqs_cluster_size``,
    ``mmseqs_representative_id``, and ``mmseqs_is_representative``.
    ``extra_args`` is passed directly to MMseqs2 after the standard options.
    """
    _validate_mmseqs2_input(
        df, min_seq_id, coverage, cov_mode, cluster_mode, sensitivity, threads
    )
    executable = shutil.which(mmseqs_executable)
    if executable is None:
        raise FileNotFoundError(
            f"MMseqs2 executable not found: {mmseqs_executable!r}"
        )

    sequences = df["sequence"].tolist()
    with tempfile.TemporaryDirectory(dir=temp_dir) as work_dir:
        work_path = Path(work_dir)
        fasta_path = work_path / "sequences.fasta"
        result_prefix = work_path / "clusters"
        result_path = work_path / "clusters_cluster.tsv"
        mmseqs_tmp = work_path / "mmseqs_tmp"
        sequence_ids = [f"sequence_{i:08d}" for i in range(len(df))]
        fasta_path.write_text(
            "".join(
                f">{sequence_id}\n{sequence}\n"
                for sequence_id, sequence in zip(sequence_ids, sequences)
            )
        )

        command = [
            executable,
            "easy-cluster",
            str(fasta_path),
            str(result_prefix),
            str(mmseqs_tmp),
            "--min-seq-id",
            str(min_seq_id),
            "-c",
            str(coverage),
            "--cov-mode",
            str(cov_mode),
            "--cluster-mode",
            str(cluster_mode),
            "-s",
            str(sensitivity),
            "--shuffle",
            "0",
            "--threads",
            str(threads),
            "-v",
            "0",
            *map(str, extra_args),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            message = error.stderr.strip() or error.stdout.strip()
            raise RuntimeError(f"MMseqs2 failed: {message}") from error

        representative_positions = _read_mmseqs2_clusters(result_path, sequence_ids)

    unique_representatives = sorted(set(representative_positions))
    readable_ids = {
        representative: cluster_id
        for cluster_id, representative in enumerate(unique_representatives, start=1)
    }
    cluster_ids = np.array(
        [readable_ids[representative] for representative in representative_positions]
    )

    result = df.copy()
    result["mmseqs_cluster_id"] = cluster_ids
    cluster_sizes = np.bincount(cluster_ids)
    result["mmseqs_cluster_size"] = cluster_sizes[cluster_ids]
    result["mmseqs_representative_id"] = [
        df.index[position] for position in representative_positions
    ]
    result["mmseqs_is_representative"] = (
        np.arange(len(df)) == representative_positions
    )
    return result


def _validate_mmseqs2_input(
    df: pd.DataFrame,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
    cluster_mode: int,
    sensitivity: float,
    threads: int,
) -> None:
    if "sequence" not in df.columns:
        raise KeyError("df must contain a 'sequence' column")
    if df.empty:
        raise ValueError("df must contain at least one sequence")
    if not df.index.is_unique:
        raise ValueError("df index must be unique")
    if not 0 <= min_seq_id <= 1:
        raise ValueError("min_seq_id must be between 0 and 1")
    if not 0 <= coverage <= 1:
        raise ValueError("coverage must be between 0 and 1")
    if cov_mode not in range(6):
        raise ValueError("cov_mode must be between 0 and 5")
    if cluster_mode not in range(4):
        raise ValueError("cluster_mode must be between 0 and 3")
    if sensitivity <= 0:
        raise ValueError("sensitivity must be greater than 0")
    if threads < 1:
        raise ValueError("threads must be at least 1")
    for design_id, sequence in df["sequence"].items():
        if not isinstance(sequence, str) or not sequence:
            raise ValueError(f"invalid sequence for design {design_id!r}")
        if ">" in sequence or any(character.isspace() for character in sequence):
            raise ValueError(f"invalid FASTA sequence for design {design_id!r}")


def _read_mmseqs2_clusters(
    result_path: Path,
    sequence_ids: Sequence[str],
) -> np.ndarray:
    positions = {sequence_id: i for i, sequence_id in enumerate(sequence_ids)}
    representative_positions = np.full(len(sequence_ids), -1, dtype=int)
    with result_path.open() as result_file:
        for line in result_file:
            representative, member = line.rstrip().split("\t")
            representative_positions[positions[member]] = positions[representative]

    if np.any(representative_positions < 0):
        missing = [
            sequence_ids[position]
            for position in np.flatnonzero(representative_positions < 0)
        ]
        raise RuntimeError(
            f"MMseqs2 cluster output omitted {len(missing)} sequences: {missing[:3]}"
        )
    return representative_positions

def foldseek_diversity():
    pass
    
