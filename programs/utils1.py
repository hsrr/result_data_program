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
from typing import Dict, List

from langchain.prompts.example_selector.base import BaseExampleSelector
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


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
        names = [
                "query_id", "subject_id", "query_length", "subject_length",
                "identity_percentage", "alignment_length", "mismatches",
                "gap_opens", "q_start", "q_end", "s_start", "s_end", "cover_ratio", "p_value"
        ]
        result_df = pd.read_csv(
                        self.result_filepath, 
                        sep="\t", 
                        index_col=False,
                        names=names)
        #print("this is the length of compare.csv!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #print(len(result_df))
        result_df = result_df[
            (result_df["identity_percentage"] >= identity_threshold) &
            (result_df["cover_ratio"] >= cover_ratio_threshold)]

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



class ProteinSimilarityExampleSelector(BaseExampleSelector):
    """基于蛋白质序列的相似度选择样例"""

    def __init__(self, examples: List[Dict[str, str]], data_dir=None, k=4):
        self.examples = examples
	#print(len(examples))
        # tmp = [(list(item.keys()), list(item.values())) for item in examples]
        # print(type(tmp[0][0][0]))
        # print(type(tmp[0][1][0]))
        tmp = [(item["sequence"], item["target"]) for item in examples]
        # print(type(tmp[0][0]))
        # print(type(tmp[0][1]))
        
        
        self.sequence2target = {example["sequence"]: example["target"] for example in self.examples}
        self.k = k
        self.sequences = [example["sequence"] for example in self.examples]
#         print(self.sequences)
#         exit(-1)
        self.comparator = SequenceComparator(sequences=self.sequences, cache_dir=data_dir)

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to store for a key."""
        self.examples.append(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the inputs."""
        # print('in select examples')
        selected = self.comparator.retrieve(input_variables["sequence"], topK=self.k)
        if selected is not None:
            selected_sequences = [result_dict["sequence"] for result_dict in selected]
        else:selected_sequences=[]
        # print('#' * 100)
        print(len(selected_sequences))
        if len(selected_sequences) < self.k:
            l=self.k - len(selected_sequences)
#             print(np.random.choice(self.sequences, l, replace=False).tolist())
            
#             randomized_sequences =self.sequences[:l] 
            randomized_sequences = np.random.choice(self.sequences, l, replace=False).tolist()
            selected_sequences = randomized_sequences + selected_sequences
        
        #if isinstance(type(selected_sequences[0]), str):
        #    print(selected_sequences[0], 'str')
        #    exit(-1)
        #else:
        #    print((selected_sequences[0], 'byte'))
        
        # exit(-1)
        # print(type(self.sequence2target[(selected_sequences[0].decode())]))
        # print('#' * 100)
        # exit(-1)
#         print(selected_sequences)
        selected_examples = []
        for item in selected_sequences:
            key = item
            if type(key)==bytes:
                key = key.decode()
#             print("selected_sequence!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#             print(selected_sequences)
            val = self.sequence2target[key]
            tmp = {"sequence": key, "target": val} 
            selected_examples.append(tmp)
        return selected_examples


