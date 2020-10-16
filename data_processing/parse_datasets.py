import argparse
import sys
import csv
import io
import os
import numpy as np
import random
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split



#TODO: don't do seq.upper(), seq with lowescases chars are different seqs, don't include for now.
def shuffle_txtfile(filepath):
    filename = filepath.split('/')[-1]
    new_filename = filename + '_shuffled'
    shuffled_filepath = os.path.join('.', new_filename)

    with open(filepath, 'r') as source:
       data = [(random.random(), line) for line in source]
    data.sort()
    with open(shuffled_filepath, 'w') as target:
        for _, line in data:
            target.write(line)
    return shuffled_filepath


def shuffle_csvfile(filepath):
    filename = filepath.split('/')[-1]
    new_filename = filename + '_shuffled'
    shuffled_filepath = os.path.join('.', new_filename)

    with open(filepath, 'r') as source:
        source_reader = csv.DictReader(source, delimiter=',')
        headers = source_reader.fieldnames
        source_rows = list(source_reader)

        data = [(random.random(), row) for row in source_rows]
        data.sort()
        with open(shuffled_filepath, 'w') as target:
            target_writer = csv.DictWriter(target, fieldnames=headers)
            target_writer.writeheader()
            for _, row in data:
                target_writer.writerow(row)
    return shuffled_filepath


def parse_ampep(ipf1='M_model_train_AMP_sequence.txt',
               ipf2 = 'M_model_train_nonAMP_sequence.txt',
               num_negseqs=5000):
    opf = os.path.join(os.path.dirname(ipf1), 'ampep-parsed.csv')
    #AMPEP has no ids assigned - so giving our own ids
    #opening commen target csv file
    with open(opf, 'w') as target:
        target_writer = csv.DictWriter(target, fieldnames=['Id', 'Sequence', 'Seqlen', 'Seqlabel'])
        target_writer.writeheader()

        with open(ipf1, 'r') as source_pos:
            i = 0
            for line in source_pos.readlines():
                if line.startswith('>'):
                    continue
                else:
                    i += 1
                    id = 'ampep-pos' + str(i)
                    seqlen = len(line)-1
                    if 10 < seqlen < 80:
                        seq = line[:-1]
                        seqlabel = 1
                        print('line NOW:: ', line, len(line))
                        print('seq NOW:: ', seq, len(seq))
                        target_writer.writerow({'Id': id , 'Sequence': seq, 'Seqlen': seqlen, 'Seqlabel' : seqlabel})

        with open(ipf2, 'r') as source_neg:
            i = 0
            n = 0
            data = [(random.random(), line) for line in source_neg.readlines()]
            data.sort()
            for _,line in data:
                if line.startswith('>'):
                    continue
                else:
                    if n != num_negseqs+1:
                        seqlen = len(line)-1
                        if 10 < seqlen < 80:
                            n += 1
                            i += 1
                            id = 'ampep-neg' + str(i)
                            seq = line[:-1]
                            seqlabel = 0
                            target_writer.writerow({'Id': id , 'Sequence': seq, 'Seqlen': seqlen, 'Seqlabel' : seqlabel})

    return opf


def parse_baamp(ipf='BAAMPs_data.csv', seqlabel=1):
    opf = os.path.join(os.path.dirname(ipf), ipf.split('/')[-1] + '_parsed.csv')
    id_seen = []

    with open(ipf, 'r', encoding='utf-16') as source:
        source_reader = csv.DictReader(source, delimiter=',')

        with open(opf, 'w') as target:
            target_writer = csv.DictWriter(target, fieldnames=['Id', 'Sequence', 'Seqlen', 'Seqlabel'])
            target_writer.writeheader()

            for row in source_reader:
                id = row['PeptideID']
                seqlen = row['PeptideSize']
                seq = (row['PeptideSequence']).upper()

                if seqlen != '':
                    if int(seqlen) > 0 and seq.isalpha():
                        if 10 < int(seqlen) < 80:
                            if id not in id_seen:
                                target_writer.writerow({'Id': id, 'Seqlen': seqlen,
                                                        'Sequence': seq, 'Seqlabel': seqlabel})
                                id_seen.append(id)

    return opf


def parse_milkamp(ipf='milkamp.csv', seqlabel=1):
    opf = os.path.join(os.path.dirname(ipf), ipf.split('/')[-1] + '_parsed.csv')
    id_seen = []

    with open(ipf) as source:
        source_reader = csv.DictReader(source, delimiter=',')

        with open(opf, 'w') as target:
            target_writer = csv.DictWriter(target, fieldnames=['Id', 'Sequence', 'Seqlen', 'Seqlabel'])
            target_writer.writeheader()
            for row in source_reader:
                id = row['id']
                seqlen = row['SeqLen']
                seq = row['Sequence'].upper()

                if seqlen != '':
                    if 10 < int(seqlen) < 80:
                        if id not in id_seen:
                            # clean sequence
                            seq_cleaned = seq.split(';')[0]
                            print('seq before :: ', seq)
                            print('seq after :: ', seq_cleaned)
                            target_writer.writerow({'Id': str(id), 'Seqlen' : seqlen,
                                                    'Sequence' : seq_cleaned, 'Seqlabel': seqlabel})
                            id_seen.append(id)

    return opf


def parse_dramp(ipf='DRAMP.tab', seqlabel=1):
    opf = os.path.join(os.path.dirname(ipf), ipf.split('/')[-1] + '_parsed.csv')
    id_seen = []

    with open(ipf, 'rU') as source:
        source_reader = csv.DictReader(source, dialect = 'excel-tab', delimiter = '\t')

        with open(opf, 'w') as target:
            target_writer = csv.DictWriter(target, fieldnames=['Id', 'Sequence', 'Seqlen', 'Seqlabel'])
            target_writer.writeheader()

            for row in source_reader:
                id = row['DRAMP.ID']
                seqlen = row['Sequence.Length']
                seq = (row['Sequence']).upper()

                if seqlen != None or seqlen !='':
                    if id not in id_seen:
                        if unicode(seqlen).isnumeric():
                            if 10 < int(seqlen) < 80:
                                if int(seqlen) > 0 and seq.isalpha():
                                    target_writer.writerow({'Id': str(id), 'Seqlen': seqlen,
                                                                'Sequence': seq, 'Seqlabel': seqlabel})
                                    id_seen.append(id)

    return opf

#TODO: several filtering conditions to be included
def parse_satpdb(ipf='satpdb.csv', seqlabel=1):
    opf = os.path.join(os.path.dirname(ipf), ipf.split('/')[-1] + '_parsed.csv')
    id_seen = []

    with open(ipf) as source:
        source_reader = csv.DictReader(source, delimiter=',')
        headers = source_reader.fieldnames
        print('headers :: ', headers)
        unique_vals = defaultdict(list)

        with open(opf, 'w') as target:
            target_writer = csv.DictWriter(target, fieldnames=['Id', 'Sequence', 'Seqlen', 'Seqlabel'])
            target_writer.writeheader()
            for i,row in enumerate(source_reader):
                for col in headers:
                    print(col, row[col])
                    unique_vals[col].append(row[col])
                #antimicrobial, anti - bacterial, anti - cancer, anti - parasitic, anti - HIV
                # if 'antimicrobial', 'antibacterial', 'anticancer', 'antiparasitic', 'antiHIV'
                #if seq is lowercase don’t consider,
                if (row['Sequence'].isupper() and 'antimicrobial' in row['Major.Functions'] and row['Peptide.Type'] == 'Linear'
                    and row['C.terminal.modification'] == 'Free' and row['N.terminal.modification'] == 'Free'):

                    id = row['Peptide.ID']
                    seq = row['Sequence']
                    seqlen = len(seq)

                    if seqlen != None or seqlen != '':
                        if (id not in id_seen) and (unicode(seqlen).isnumeric()) and (
                                    10 < int(seqlen) < 80) and seq.isalpha():
                            # clean sequence
                            #seq_cleaned = seq.split(';')[0]
                            target_writer.writerow({'Id': str(id), 'Seqlen' : seqlen,
                                                'Sequence' : seq, 'Seqlabel': seqlabel})
                            id_seen.append(id)

            for key,val in unique_vals.items():
               unique_vals[key]=list(set(val))
                #if key == 'Major.Functions':
                #    val = [v for v in val_lst.split(' ') for val_lst in val if v not in (',', '(', ')') and not v.isalpha()]
                #    unique_vals[key] = val
            #print(unique_vals['Major.Functions'])


    return opf


def parse_dbaasp(ipf, seqlabel=1):
    #TODO: write a json parser here, along with filtering conditions
    pass

def parse_uniprot(ipf='uniprot.tab', seqlabel=2):
    print('*****Parsing Uniprot************')
    opf = os.path.join(os.path.dirname(ipf), ipf.split('/')[-1].split('.')[0] + '_parsed.csv')
    id_seen = []

    with open(ipf, 'rU') as source:
        source_reader = csv.reader(source, dialect = 'excel-tab', delimiter = '\t')

        with open(opf, 'w+') as target:
            target_writer = csv.DictWriter(target, fieldnames=['Id', 'Sequence', 'Seqlen', 'Seqlabel'])
            target_writer.writeheader()

            for row in source_reader:
                id = row[0]
                seqlen = row[6]
                seq = (row[7]).upper()

                if seqlen != None or seqlen !='':
                    if (id not in id_seen) and (unicode(seqlen).isnumeric()) and (
                                    10 < int(seqlen) < 80) and seq.isalpha():
                        target_writer.writerow({'Id': str(id), 'Seqlen': seqlen,
                                                'Sequence': seq, 'Seqlabel': seqlabel})
                        id_seen.append(id)

    return opf


def parse_trembl(ipf='trembl.tab', seqlabel=2):
    print('*****Parsing Trembl************')
    opf = os.path.join(os.path.dirname(ipf), ipf.split('/')[-1].split('.')[0] + '_parsed.csv')
    id_seen = []

    with open(ipf, 'rU') as source:
        source_reader = csv.DictReader(source, dialect='excel-tab', delimiter='\t')

        with open(opf, 'w+') as target:
            target_writer = csv.DictWriter(target, fieldnames=['Id', 'Sequence', 'Seqlen', 'Seqlabel'])
            target_writer.writeheader()

            for row in source_reader:
                id = row['Entry']
                seqlen = row['Length']
                seq = (row['Sequence']).upper()

                if seqlen != None or seqlen != '':
                    if id not in id_seen:
                        if unicode(seqlen).isnumeric():
                            if 10 < int(seqlen) < 80 and seq.isalpha():
                                target_writer.writerow({'Id': str(id), 'Seqlen': seqlen,
                                                        'Sequence': seq, 'Seqlabel': seqlabel})
                                id_seen.append(id)

    return opf


def write_nparr_to_csv(x, y, file_handle):
    for seq, seqlabel in zip(x, y):
        seq_str = ''
        for char in seq:
            seq_str += char + ' '
        # Todo: if needed add this later
        # seq_str = seq_str[:-1]  #removing extra space at the end of seq
        file_handle.writerow((seq_str, seqlabel))


def split_dataset(ipf, ratio=[0.2, 0.2]):
    valid_ratio = ratio[0]
    test_ratio = ratio[1] * 2
    data = pd.read_csv(ipf)
    print(data.head())
    X = data.Sequence
    y = data.Seqlabel

    # valid
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_ratio)

    # test
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_ratio)

    # writing to files
    filesn = ipf.split('/')[-1].split('.')[0]
    print('name of the file :: ', filesn)
    print('inside split dataset - path dirname :: ', os.path.dirname(ipf))
    train_file = open(os.path.join(os.path.dirname(ipf), filesn + '_train.csv'), 'w')
    valid_file = open(os.path.join(os.path.dirname(ipf), filesn + '_valid.csv'), 'w')
    test_file = open(os.path.join(os.path.dirname(ipf), filesn + '_test.csv'), 'w')

    try:
        train_writer = csv.writer(train_file)
        valid_writer = csv.writer(valid_file)
        test_writer = csv.writer(test_file)

        write_nparr_to_csv(X_train, y_train, train_writer)
        write_nparr_to_csv(X_test, y_test, test_writer)
        write_nparr_to_csv(X_valid, y_valid, valid_writer)

    finally:
        train_file.close()
        valid_file.close()
        test_file.close()



def create_labeled_data():
    #Combining AMPEP + BAAMP + MILKAMP + DRAAMP , AMPEP - non-AMPs/-ve/ form 0 labeled seqs , AMPEP - AMPs/+ve/, BAAMP, MILKAMP, DRAAMP form 1 labeled seqs
    # TODO: include dbaasp and satpdb
    ampep_filename = parse_ampep(num_negseqs=5000)
    baamp_filename = parse_baamp()
    milkamp_filename = parse_milkamp()
    dramp_filename = parse_dramp()

    file_lst = [baamp_filename, milkamp_filename, dramp_filename, ampep_filename]
    merged_fn = os.path.join('amp_lab.csv')

    full_df = pd.concat([pd.read_csv(file, skiprows=0, usecols=[0, 1, 2, 3]) for file in file_lst])
    unique_df = full_df.drop_duplicates(subset=['Sequence'], keep='last')
    unique_df.to_csv(merged_fn)
    # delete _parsed csv files after creating a merged file - save up space
    for file in file_lst:
        os.remove(file)

    split_dataset(merged_fn)


def create_unlabeled_data():
    #Combining Uniprot + Trembl
    uniprot_fn = parse_uniprot()
    trembl_fn = parse_trembl()
    file_lst = [uniprot_fn, trembl_fn]
    merged_fn = os.path.join('amp_unlab.csv')

    full_df = pd.concat([pd.read_csv(file, skiprows=0, usecols=[0, 1, 2, 3]) for file in file_lst])
    unique_df = full_df.drop_duplicates(subset=['Sequence'], keep='last')
    unique_df.to_csv(merged_fn)
    # delete _parsed csv files after creating a merged file - save up space
    for file in file_lst:
        os.remove(file)

    split_dataset(merged_fn)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--ipfile', type=str)
    #TODO: should extract on its own from ip_filename, no need to take from user
    argparser.add_argument('--opfile', type=str)
    args = argparser.parse_args()
    #create_labeled_data()
    create_unlabeled_data()
    #parse_trembl()