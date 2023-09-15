# -*- coding: utf-8 -*-
"""
Filename:    utils
Author:      chenxin
Date:        2023/8/1
Description: 
"""
import json
import os.path
import random
import time
import subprocess

import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

from itertools import product

def load_json(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as reader:
        for line in reader.readlines():
            content = line.strip("\n")
            sample = json.loads(content)
            data.append(sample)
    return data


def select_topK(samples, k=2, sampling="top"):
    if sampling == "top":
        return samples[:k]
    elif sampling == "random":
        random.shuffle(samples)
        return samples[:k]
    else:
        raise ValueError


def construct_icl_pool(df, k=2, sampling="top"):
    icl_pool = []

    if sampling in ["top", "random"]:
        icl_df = df.groupby("target").agg(lambda x: select_topK(x, k=k, sampling=sampling)).reset_index()
        for row_id, row in icl_df.iterrows():
            target = row["target"]
            sequences = row["sequence"]
            for sequence in sequences:
                item = {
                    "sequence": sequence,
                    "target": target
                }
                icl_pool.append(item)

    random.shuffle(icl_pool)
    return icl_pool


def save_fasta(df, filepath):
    records = []
    for sample_id, sample in df.iterrows():
        sequence_str = sample["sequence"]
        sequence_id = f"seq{sample_id}"
        record = SeqRecord(Seq(
            sequence_str),
            id=sequence_id,
        )
        records.append(record)

    SeqIO.write(records, filepath, "fasta")



class SequenceComparator(object):

    def __init__(self, sequences, cache_dir):
        super(SequenceComparator, self).__init__()
        self.sequences = sequences
        self.cache_dir = cache_dir
        self.fasta_filepath = os.path.join(self.cache_dir, "database.fasta")
        self.db_filepath = os.path.join(self.cache_dir, "database.db")
        self.query_filepath = os.path.join(self.cache_dir, "query.fasta")
        self.result_filepath = os.path.join(self.cache_dir, "compare.csv")

        # 建库
        self.construct_fasta_db()
        self.seq_id2sequence = self.db_to_dict()

    def retrieve(self, query_sequence, topK=4, identity_threshold=0, cover_ratio_threshold=0):
        # 检索，序列比对
        self.construct_query(query_sequence)
        command = f'blastp -query {self.query_filepath} -db {self.db_filepath} -outfmt "6 qseqid sseqid qlen slen pident length  mismatch gapopen qstart qend sstart send qcovs evalue" -out {self.result_filepath}'
        os.system(command)
        result_df = pd.read_csv(self.result_filepath, sep="\t", index_col=False,
                    names=["query_id", "subject_id", "query_length", "subject_length",
                       "identity_percentage", "alignment_length", "mismatches",
                       "gap_opens", "q_start", "q_end", "s_start", "s_end", "cover_ratio", "p_value"])
#         result_df = result_df[
#             (result_df["identity_percentage"] >= identity_threshold) &
#             (result_df["cover_ratio"] >= cover_ratio_threshold)]

        if len(result_df) > 0:
            result_df = result_df.iloc[:topK]
            results = [
                {
                    "sequence": self.seq_id2sequence[sample["subject_id"]],
                    "identity_percentage": sample["identity_percentage"],
                    "cover_ratio": sample["cover_ratio"],
                    "p_value": sample["p_value"],
                }
                for sample_idx, sample in result_df.iterrows()
            ]
            return results
        else:
            return None

    def construct_fasta_db(self):
        if os.path.exists(self.fasta_filepath):
            return

        records = []
        for sample_id, sequence_str in enumerate(self.sequences):
            sequence_id = f"seq{sample_id}"
            record = SeqRecord(
                Seq(sequence_str),
                id=sequence_id,
            )
            records.append(record)
        SeqIO.write(records, self.fasta_filepath, "fasta")

        command = f"makeblastdb -in {self.fasta_filepath} -dbtype prot -out {self.db_filepath}"
        # p = subprocess.Popen(command, shell=True)
        # p.wait()
        os.system(command)
        print("Building protein database ...")
        db_filepaths = [
            self.db_filepath + ".phr",
            self.db_filepath + ".pin",
            self.db_filepath + ".psq",
        ]
        while not os.path.exists(db_filepaths[0]) or not os.path.exists(db_filepaths[1]) or not os.path.exists(db_filepaths[2]):
            time.sleep(5)
        # assert os.path.exists(self.fasta_filepath) and os.path.exists(self.db_filepath)

    def construct_query(self, query_sequence):
        sequence_str = query_sequence
        sequence_id = f"seq{len(self.sequences)}"
        record = SeqRecord(
            Seq(sequence_str),
            id=sequence_id)
        SeqIO.write(record, self.query_filepath, "fasta")

    def db_to_dict(self):
        seq_no2sequence = SeqIO.to_dict(SeqIO.parse(self.fasta_filepath, "fasta"))
        seq_no2sequence = {seq_no : sequence_obj.seq._data for seq_no, sequence_obj in seq_no2sequence.items()}
        return seq_no2sequence


from langchain.prompts.example_selector.base import BaseExampleSelector
from typing import Dict, List
import numpy as np


class ProteinSimilarityExampleSelector(BaseExampleSelector):
    """基于蛋白质序列的相似度选择样例"""

    def __init__(self, examples: List[Dict[str, str]], data_dir=None, k=4):
        self.examples = examples
#         print(1111111111111111111111111111111111111111111111111111111111111111111)
#         print(self.examples)
        
        self.sequence2target = {example["sequence"]: example["target"] for example in self.examples}
        self.k = k
#         print(self.sequence2target)
        self.sequences = [example["sequence"] for example in self.examples]
#         print(self.sequences)
        self.new_sequences=[]
        for se in self.sequences:
            s1,s2=se.split('__')
            self.new_sequences.append(s1)
            self.new_sequences.append(s2)
        
        self.comparator = SequenceComparator(sequences=self.new_sequences, cache_dir=data_dir)
        #

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to store for a key."""
        self.examples.append(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the inputs."""
        
        sequence1,sequence2=input_variables["sequence"].split('__')
        print("sequence1")
        print(sequence1)
        selected1 = self.comparator.retrieve(sequence1, topK=self.k)
        print(self.k)                                     
        print("selected1....................")
        print(len(selected1))
       
        selected2 = self.comparator.retrieve(sequence2, topK=self.k)
        selected_sequences1 = [result_dict["sequence"] for result_dict in selected1]
#         print(11111111111111111111111111111111)
#         print(selected_sequences1)
        selected_sequences2=[result_dict["sequence"] for result_dict in selected2]
        results = []
        selected_sequences1 = [seq.decode() for seq in selected_sequences1]
        selected_sequences2 = [seq.decode() for seq in selected_sequences2]

        for combo in product(selected_sequences1, selected_sequences2):
            results.append('__'.join(combo))

        for combo in product(selected_sequences2, selected_sequences1):
            results.append('__'.join(combo))
        
#         print(results)
        
        selected_sequences = list(set(results).intersection(set(self.sequences)))

        print(1111111111111111111111111111111111)
        print(len(selected_sequences))
        if len(selected_sequences) < 4:
            randomized_sequences = np.random.choice(self.sequences, 4 - len(selected_sequences), replace=False).tolist()
            selected_sequences = randomized_sequences + selected_sequences
        
        selected_examples = [{"sequence": sequence, "target": self.sequence2target[sequence]} for sequence in selected_sequences]
        return selected_examples
