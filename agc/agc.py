#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""OTU clustering"""

import argparse
import sys
import os
import gzip
import statistics
import textwrap
from pathlib import Path
from collections import Counter
from typing import Iterator, Dict, List
# https://github.com/briney/nwalign3
# ftp://ftp.ncbi.nih.gov/blast/matrices/
# Régler incompatibilité numpy / nwalign3
import numpy as _np
if not hasattr(_np, "int"):
    _np.int = int
import nwalign3 as nw

__author__ = "Romain Gerard"
__copyright__ = "Universite Paris Cité"
__credits__ = ["Romain Gerard"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Romain Gerard"
__email__ = "romain.ge@proton.me"
__status__ = "Developpement"


def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments(): # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, usage=
                                     "{0} -h"
                                     .format(sys.argv[0]))
    parser.add_argument('-i', '-amplicon_file', dest='amplicon_file', type=isfile, required=True, 
                        help="Amplicon is a compressed fasta file (.fasta.gz)")
    parser.add_argument('-s', '-minseqlen', dest='minseqlen', type=int, default = 400,
                        help="Minimum sequence length for dereplication (default 400)")
    parser.add_argument('-m', '-mincount', dest='mincount', type=int, default = 10,
                        help="Minimum count for dereplication  (default 10)")
    parser.add_argument('-o', '-output_file', dest='output_file', type=Path,
                        default=Path("OTU.fasta"), help="Output file")
    return parser.parse_args()


def read_fasta(amplicon_file: Path, minseqlen: int) -> Iterator[str]:
    """Read a compressed fasta and extract all fasta sequences.

    :param amplicon_file: (Path) Path to the amplicon file in FASTA.gz format.
    :param minseqlen: (int) Minimum amplicon sequence length
    :return: A generator object that provides the Fasta sequences (str).
    """
    seq_parts = []
    with gzip.open(amplicon_file, mode="rt") as file:
        for line in file:
            if not line:
                continue
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if seq_parts:
                    seq = "".join(seq_parts)
                    if len(seq) >= minseqlen:
                        yield seq
                    seq_parts = []
            else:
                seq_parts.append(line)

        if seq_parts:
            seq = "".join(seq_parts)
            if len(seq) >= minseqlen:
                yield seq


def dereplication_fulllength(amplicon_file: Path, minseqlen: int, mincount: int) -> Iterator[List]:
    """Dereplicate the set of sequence

    :param amplicon_file: (Path) Path to the amplicon file in FASTA.gz format.
    :param minseqlen: (int) Minimum amplicon sequence length
    :param mincount: (int) Minimum amplicon count
    :return: A generator object that provides a (list)[sequences, count] of sequence with a count >= mincount and a length >= minseqlen.
    """
    counts = Counter()

    for seq in read_fasta(amplicon_file, minseqlen):
        counts[seq] += 1

    items = ((s, c) for s, c in counts.items() if c >= mincount)
    for seq, cnt in sorted(items, key=lambda x: (-x[1], x[0])):
        yield [seq, cnt]


def get_identity(alignment_list: List[str]) -> float:
    """Compute the identity rate between two sequences

    :param alignment_list:  (list) A list of aligned sequences in the format ["SE-QUENCE1", "SE-QUENCE2"]
    :return: (float) The rate of identity between the two sequences.
    """
    if not alignment_list or len(alignment_list) != 2:
        return 0.0
    a, b = alignment_list
    if not a:
        return 0.0
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return 100.0 * matches / len(a)


def abundance_greedy_clustering(amplicon_file: Path, minseqlen: int, mincount: int, chunk_size: int, kmer_size: int) -> List:
    """Compute an abundance greedy clustering regarding sequence count and identity.
    Identify OTU sequences.

    :param amplicon_file: (Path) Path to the amplicon file in FASTA.gz format.
    :param minseqlen: (int) Minimum amplicon sequence length.
    :param mincount: (int) Minimum amplicon count.
    :param chunk_size: (int) A fournir mais non utilise cette annee
    :param kmer_size: (int) A fournir mais non utilise cette annee
    :return: (list) A list of all the [OTU (str), count (int)] .
    """
    _ = (chunk_size, kmer_size)

    otus: List[List] = []
    matrix_path = str(Path(__file__).parent / "MATCH")

    for seq, count in dereplication_fulllength(amplicon_file, minseqlen, mincount):
        similar = False
        for otu_seq, _otu_count in otus:
            align = nw.global_align(seq, otu_seq, gap_open=-1, gap_extend=-1, matrix=matrix_path)
            if get_identity(align) >= 97.0:  # seuil 97%
                similar = True
                break
        if not similar:
            otus.append([seq, count])

    return otus


def write_OTU(OTU_list: List, output_file: Path) -> None:
    """Write the OTU sequence in fasta format.

    :param OTU_list: (list) A list of OTU sequences
    :param output_file: (Path) Path to the output file
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)  # sécurité

    with output_file.open("w") as out:
        for i, (seq, count) in enumerate(OTU_list, start=1):
            out.write(f">OTU_{i} occurrence:{count}\n")
            out.write(textwrap.fill(seq, width=80) + "\n")


#==============================================================
# Main program
#==============================================================
def main(): # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()

    # Votre programme ici
    chunk_size = getattr(args, "chunk_size", 100)
    kmer_size = getattr(args, "kmer_size", 8)

    otu_list = abundance_greedy_clustering(
        amplicon_file=args.amplicon_file,
        minseqlen=args.minseqlen,
        mincount=args.mincount,
        chunk_size=chunk_size,
        kmer_size=kmer_size,
    )

    write_OTU(otu_list, args.output_file)


if __name__ == '__main__':
    main()
