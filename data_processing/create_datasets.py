# coding: utf-8
import json
import pandas as pd
import numpy as np
import glob
import ast
from modlamp.descriptors import *


def not_in_range(seq):
    if len(seq) < 1 or len(seq) > 80:
        return True
    return False


def bad_terminus(peptide):
    if peptide.nTerminus[0] != "#" or peptide.cTerminus[0] != "#":
        return True
    return False


def is_valid(peptide):
    seq = peptide.seq[0]
    if not seq.isupper():
        return False
    if bad_terminus(peptide):
        return False
    if not_in_range(seq):
        return False
    if seq.find("X") != -1:
        return False
    return True


def get_valid_sequences():
    peptides = pd.DataFrame()
    all_file_names = []
    for j_file in glob.glob("./data/dbaasp/*.json"):
        filename = j_file[j_file.rfind("/") + 1:]
        with open(j_file) as train_file:
            dict_train = json.load(train_file)
        if dict_train["peptideCard"].get("unusualAminoAcids") is not None:
            continue
        peptide = pd.DataFrame.from_dict(dict_train, orient='index')
        if is_valid(peptide):
            peptides = pd.concat([peptides, peptide])
            all_file_names.append(filename)
    peptides["filename"] = all_file_names
    peptides.to_csv("valid_sequences.csv")
    return peptides


def add_activity_list(peptides):
    activity_list_all = []
    for targets in peptides.targetActivities:  # one seq has a list of targets
        try:
            activity_list = []
            for target in targets:
                if target['unit'] == 'µM':  # µg/ml
                    try:
                        activity_list.append(target['concentration'])
                    except:
                        continue
            activity_list_all.append(activity_list)
        except:
            activity_list_all.append([])
            continue
    peptides["activity_list"] = activity_list_all
    return peptides


def add_toxic_list(peptides):
    toxic_list_all = []
    for targets in peptides.hemoliticCytotoxicActivities:  # one seq has a list of targets
        try:
            toxic_list = []
            for target in targets:
                if target['unit'] == 'µM':  # µg/ml
                    try:
                        toxic_list.append(target['concentration'])
                    except:
                        continue
            toxic_list_all.append(toxic_list)
        except:
            toxic_list_all.append([])
            continue
    peptides["toxic_list"] = toxic_list_all
    return peptides


def add_molecular_weights(peptides):
    seqs = [doc for doc in peptides["seq"]]
    mws = []
    for seq in seqs:
        try:
            desc = GlobalDescriptor(seq.strip())
            desc.calculate_MW(amide=True)
            mw = desc.descriptor[0][0]
            mws.append(mw)
        except:
            mws.append(None)

    peptides["molecular_weight"] = mws
    return peptides


def convert_units(peptides):
    converted_activity_all = []
    converted_toxic_all = []
    for activity_list, toxic_list, molecular_weight in zip(peptides.activity_list,
                                                           peptides.toxic_list,
                                                           peptides.molecular_weight):
        converted_activity_list = []
        converted_toxic_list = []

        for item in activity_list:
            item = item.replace(">", "")  # '>10' => 10
            item = item.replace("<", "")  # '<1.25' => 1.25
            item = item.replace("=", "")  # '=2' => 2
            if item == "NA":
                continue
            if item.find("±") != -1:
                item = item[:item.find("±")]  # 10.7±4.6 => 10.7
            if item.find("-") != -1:
                item = item[:item.find("-")]  # 12.5-25.0 => 12.5
            item = item.strip()
            try:
                converted_activity_list.append(float(item) * molecular_weight / 1000)
            except:
                pass

        for item in toxic_list:
            item = item.replace(">", "")  # '>10' => 10
            item = item.replace("<", "")  # '<1.25' => 1.25
            item = item.replace("=", "")  # '=2' => 2
            if item == "NA":
                continue
            if item.find("±") != -1:
                item = item[:item.find("±")]  # 10.7±4.6 => 10.7
            if item.find("-") != -1:
                item = item[:item.find("-")]  # 12.5-25.0 => 12.5
            item = item.strip()
            try:
                converted_toxic_list.append(float(item) * molecular_weight / 1000)
            except:
                pass

        converted_activity_all.append(converted_activity_list)
        converted_toxic_all.append(converted_toxic_list)
    peptides["converted_activity"] = converted_activity_all
    peptides["converted_toxic"] = converted_toxic_all
    print('--> Writing valid sequences with molecular weights converted to valid_sequences_with_mw_converted.csv')
    peptides.to_csv("valid_sequences_with_mw_converted.csv")
    return peptides


# Starting process
print('Dataset Creation process begins ... ')
# AMP data
print('**** Creating AMP datasets ****')
# Get Valid Sequences
peptide_all = get_valid_sequences()
print ('1. Getting all valid peptide sequences from DBAASP, number of seqs extracted = ', len(peptide_all))
print('--> Sequences stored in valid_sequences.csv')

# Add molecular weights
print('2. Converting Molecular weights')
peptide_all_with_mw = add_molecular_weights(peptide_all)

# Extract list of anti-microbial activities and list of toxicities
peptide_all_with_activity = add_activity_list(peptide_all)
peptide_all_with_activity_toxicity = add_toxic_list(peptide_all_with_activity)

# Add the converted units to activity list and toxicity list
peptide_all_converted = convert_units(peptide_all_with_activity_toxicity)


# Statistics
def get_stats():
    peptides = pd.DataFrame()
    all_file_names = []
    total = 0
    unusual_amino_acids = 0
    for j_file in glob.glob("./data/dbaasp/*.json"):
        total += 1
        filename = j_file[j_file.rfind("/") + 1:]
        with open(j_file) as train_file:
            dict_train = json.load(train_file)
        if dict_train["peptideCard"].get("unusualAminoAcids") is not None:
            unusual_amino_acids += 1
            continue
        peptide = pd.DataFrame.from_dict(dict_train, orient='index')
        peptides = pd.concat([peptides, peptide])
        all_file_names.append(filename)
    peptides["filename"] = all_file_names
    print ("--> For DBAASP:")
    print ("Total number of sequences:", total)
    print ("Total number of unusual AminoAcids:", unusual_amino_acids)
    return peptides


print('3. Some Statistics of collected valid sequences')
peptide_all = get_stats()
not_valid_count = len([seq for seq in peptide_all.seq if not_in_range(seq)])
print ("--> Number of not in range sequences:", not_valid_count)
print ("--> Number of valid sequences:", len(peptide_all_converted))
has_activity = [item for item in peptide_all_converted.activity_list if item != []]
print ("--> Number of valid sequences with antimicrobial activity:", len(has_activity))
has_toxicity = [item for item in peptide_all_converted.toxic_list if item != []]
print ("--> Number of valid sequences with toxicity:", len(has_toxicity))

################################################################
df = pd.read_csv("valid_sequences_with_mw_converted.csv")
print (len(df))
# df.head()  # default df: is dbaasp


def add_min_max_mean(df_in):
    min_col = [min(ast.literal_eval(li_str)) if li_str != '[]' else '' for li_str in list(df_in.converted_activity)]
    max_col = [max(ast.literal_eval(li_str)) if li_str != '[]' else '' for li_str in list(df_in.converted_activity)]
    mean_col = [np.mean(ast.literal_eval(li_str)) if li_str != '[]' else '' for li_str in list(df_in.converted_activity)]

    df_in["min_activity"] = min_col
    df_in["max_activity"] = max_col
    df_in["avg_activity"] = mean_col
    return df_in


def all_activity_more_than_30(x_str):
    x = ast.literal_eval(x_str)
    for i in range(len(x)):
        if x[i] < 30:
            return False   # all of them
        # just for negative (pos: any item < 10, neg: all of them > 30)
    return True


def all_activity_more_than_str(x_str, num):
    x = ast.literal_eval(x_str)
    if len(x) == 0:
        return False
    for i in range(len(x)):
        if x[i] < num:
            return False
    return True


def all_activity_more_than(df, num):
    return df[df['converted_activity'].apply(lambda x: all_activity_more_than_str(x, num))]


def all_toxic_more_than(df, num):
    return df[df['converted_toxic'].apply(lambda x: all_activity_more_than_str(x, num))]


def all_activity_less_than_str(x_str, num):
    x = ast.literal_eval(x_str)
    if len(x) == 0:
        return False
    for i in range(len(x)):
        if x[i] > num:
            return False
    return True


def all_toxic_less_than(df, num):
    return df[df['converted_toxic'].apply(lambda x: all_activity_less_than_str(x, num))]


def has_activity_less_than_10(x_str):
    x = ast.literal_eval(x_str)
    for i in range(len(x)):
        if x[i] < 10:
            return True
    return False


def has_activity_less_than_str(x_str, num):
    x = ast.literal_eval(x_str)
    for i in range(len(x)):
        if x[i] < num:
            return True
    return False


def has_activity_less_than(df, num):
    return df[df['converted_activity'].apply(lambda x: has_activity_less_than_str(x, num))]


def get_seq_len_less_than(df, seq_length):
    df_short = df[df['seq'].apply(lambda x: len(x) <= seq_length)]
    return df_short


def remove_df(df1, df2):
    return pd.concat([df1, df2, df2]).drop_duplicates(keep=False)


# add min, max, mean to all dbaasp
df = add_min_max_mean(df)
df_dbaasp = df[["seq", "activity_list", "converted_activity",
                "min_activity", "max_activity", "avg_activity"]]
df_dbaasp.to_csv("all_valid_dbaasp.csv")

# 3) Overlapping sequences between DBAASP and Satpdb  with AMP activity <10 ug/ml
print('4. Finding overlapping sequences between DBAASP and Satpdb with AMP activity <10 ug/ml ...')
df_satpdb = pd.read_csv("./data/satpdb/satpdb.csv")
df_satpdb = df_satpdb.rename(index=str, columns={"Sequence": "seq",
                                                 "C.terminal.modification": "cterminal",
                                                 "N.terminal.modification": "nterminal",
                                                 "Peptide.Type": "Peptide_Type",
                                                 "Type.of.Modification": "modi"})

valid_df_satpdb = df_satpdb[(df_satpdb.cterminal == "Free") &
                            (df_satpdb.nterminal == "Free") &
                            (df_satpdb.Peptide_Type == "Linear") &
                            (df_satpdb.modi == "None")]
print ("--> Number of valid satpdb = ", len(valid_df_satpdb))

df_overlap = pd.merge(df, valid_df_satpdb, on='seq', how='inner')
print ("--> Number of overlap sequences = ", len(df_overlap))

min_col = [min(ast.literal_eval(li_str)) if li_str != '[]' else '' for li_str in list(df_overlap.converted_activity)]
max_col = [max(ast.literal_eval(li_str)) if li_str != '[]' else '' for li_str in list(df_overlap.converted_activity)]
mean_col = [np.mean(ast.literal_eval(li_str)) if li_str != '[]' else '' for li_str in list(df_overlap.converted_activity)]
df_overlap["min_activity"] = min_col
df_overlap["max_activity"] = max_col
df_overlap["avg_activity"] = mean_col
df_overlap_all = df_overlap[["seq", "activity_list", "converted_activity",
                             "min_activity", "max_activity", "avg_activity"]]
print('5. Writing the overlap sequences to all_overlap.csv')
df_overlap_all.to_csv("all_overlap.csv")

# length for all <=50
#
# overlap_neg: satpdb all activity greater than 100 : negative
# ** satpdb_pos: satpdb (the same as uniprot1) - overlap_neg
# dbaasp < 25 -> pos anything
# ** amp_pos = dbassp < 25 + satpdb_pos

# select sequences dbaasp, satpdb, and overlap(dbaasp, satpdb) of len <=50
print('6. Selecting sequences dbaasp, satpdb, and overlap(dbaasp, satpdb) of len <=50')
df = get_seq_len_less_than(df, 50)
df_overlap = get_seq_len_less_than(df_overlap, 50)
valid_df_satpdb = get_seq_len_less_than(valid_df_satpdb, 50)

print('7. Selecting negative and positive sequences for AMP activity')
overlap_neg = all_activity_more_than(df_overlap, 100)
print ("--> Number of negative seq in satpdb", len(overlap_neg))
print ("--> Number of unique seq in satpdb", len(valid_df_satpdb["seq"].drop_duplicates()))
satpdb_pos = remove_df(valid_df_satpdb["seq"].drop_duplicates(), overlap_neg["seq"])
satpdb_pos1 = pd.DataFrame({'seq': satpdb_pos.values})  # amp_pos[["seq"]]
satpdb_pos1["source"] = ["satpdb_pos"] * len(satpdb_pos1)
satpdb_pos1 = satpdb_pos1[["seq", "source"]]
print ("--> Number of positive seq in satpdb", len(satpdb_pos))
satpdb_pos1.seq = satpdb_pos1.seq.apply(lambda x: "".join(x.split()))  # remove the space from the seq
satpdb_pos1 = satpdb_pos1.drop_duplicates('seq')
print('--> Writing to satpdb_pos.csv')
satpdb_pos1.to_csv("satpdb_pos.csv", index=False, header=False)


# combine all positive sequences
print('8. Combining all positive sequences for AMP activity')
col_Names = ["seq", "label"]
print('--> Parsing ampep sequences')
ampep_pos = pd.read_csv("./data/ampep/pos_ampep_l1-80.csv", names=col_Names)
ampep_pos = ampep_pos.drop(columns=['label'])
ampep_pos.seq = ampep_pos.seq.apply(lambda x: "".join(x.split()))  # remove the space from the seq
ampep_pos = get_seq_len_less_than(ampep_pos, 50)
ampep_pos["source"] = ["ampep_pos"]*len(ampep_pos)
ampep_pos = ampep_pos[["seq", "source"]]
print('--> Writing to ampep_pos.csv')
print ("--> Number of ampep_pos", len(ampep_pos))
ampep_pos.to_csv("ampep_pos.csv", index=False, header=False)

print('--> Writing dbaasp sequences')
print ("--> Number of all seqs dbaasp", len(df))
dbaasp_pos = has_activity_less_than(df, 25)["seq"]
dbaasp_pos1 = pd.DataFrame({'seq': dbaasp_pos.values})
dbaasp_pos1["source"] = ["dbaasp_pos"] * len(dbaasp_pos1)
dbaasp_pos1 = dbaasp_pos1[["seq", "source"]]

print ("--> Number of dbaasp_less_than_25:", len(dbaasp_pos), "number of satpdb_pos:", len(satpdb_pos))

amp_pos = pd.concat([dbaasp_pos1, satpdb_pos1, ampep_pos]).drop_duplicates('seq')
print ("--> Number of amp_pos", len(amp_pos))
amp_pos.columns = ['seq', 'source']
amp_pos['source2'] = amp_pos['source']
amp_pos['source'] = amp_pos['source'].map({'dbaasp_pos': 'amp_pos', 'ampep_pos': 'amp_pos', 'satpdb_pos': 'amp_pos'})
amp_pos = amp_pos[amp_pos['seq'].str.contains('^[A-Z]+')]
amp_pos = amp_pos[~amp_pos.seq.str.contains("B")]
amp_pos = amp_pos[~amp_pos.seq.str.contains("J")]
amp_pos = amp_pos[~amp_pos.seq.str.contains("O")]
amp_pos = amp_pos[~amp_pos.seq.str.contains("U")]
amp_pos = amp_pos[~amp_pos.seq.str.contains("X")]
amp_pos = amp_pos[~amp_pos.seq.str.contains("Z")]
amp_pos = amp_pos[~amp_pos.seq.str.contains('[a-z]')]
amp_pos = amp_pos[~amp_pos.seq.str.contains("-")]
amp_pos = amp_pos[~amp_pos.seq.str.contains(r'[0-9]')]
#amp_pos.seq = amp_pos.seq.apply(lambda x: " ".join(x)) # remove the space from the seq
print('--> Writing amp_pos.csv combined from dbaasp, ampep, satpdb positive sequences')
amp_pos.to_csv("amp_pos.csv", index=False, header=False)


dbaasp_more_than_100 = pd.DataFrame()
dbaasp_more_than_100["seq"] = all_activity_more_than(df, 100)["seq"]
#print ("dbaasp_more_than_100", len(dbaasp_more_than_100))
#print(all_activity_more_than(df, 100).head())


# ampep  negative and uniprot sequences
print('9. Collecting uniprot sequences as unknown label')
col_Names = ["seq"]
uniprot_unk1 = pd.read_csv("./data/uniprot/uniprot_reviewed_yes_l1-80.txt", names=col_Names)
col_Names = ["seq"]
uniprot_unk2 = pd.read_csv("./data/uniprot/uniprot_reviewed_no_l1-80.txt", names=col_Names)
uniprot_unk = pd.concat([uniprot_unk1, uniprot_unk2]).drop_duplicates()
uniprot_unk = get_seq_len_less_than(uniprot_unk, 50)
print ("--> uniprot_unk", len(uniprot_unk))
uniprot_unk["source"] = ["uniprot"] * len(uniprot_unk)
uniprot_unk["source2"] = uniprot_unk["source"]
uniprot_unk['source'] = uniprot_unk['source'].map({'uniprot': 'unk'})
print('--> Writing uniprot_unk.csv ')
uniprot_unk.to_csv("uniprot_unk.csv", index=False, header=False)

print('10. Collecting negative sequences for AMP activity ...')
col_Names = ["seq", "label"]
ampep_neg = pd.read_csv("./data/ampep/neg_ampep_l1-80.csv", names=col_Names)
ampep_neg.seq = ampep_neg.seq.apply(lambda x: "".join(x.split()))  # remove the space from the seq
#ampep_neg.columns = ['']
ampep_neg = ampep_neg.drop(columns=['label'])
ampep_neg = get_seq_len_less_than(ampep_neg, 50)
#print ("----------")
print ("--> Parsing ampep negative sequences, number of ampep_neg = ", len(ampep_neg))

# dbaasp_neg = dbaasp > 100 -> neg (how many you loose)
# Combined_NEG: 10*(dbaasp > 100) + UNIPROT_0
# Combined_POS = Satpdb_pos + ampep_pos + dbaasp_pos

dbaasp_more_than_100["source"] = ["dbaasp_neg"] * len(dbaasp_more_than_100)

# remove duplicates between ampep negative and dbaasp negative
ampep_neg["source"] = ["ampep_neg"] * len(ampep_neg)
ampep_neg = ampep_neg[["seq", "source"]]

print ("--> dbaasp_more_than_100:", len(dbaasp_more_than_100), "ampep_neg:", len(ampep_neg))
# combined_neg = remove_df(pd.concat([dbaasp_more_than_100, uniprot_neg]).drop_duplicates, amp_pos1)
combined_neg = pd.concat([dbaasp_more_than_100, ampep_neg]).drop_duplicates('seq')
# satpdb_pos = remove_df(valid_df_satpdb["seq"].drop_duplicates(), overlap_neg["seq"])
print ("--> combined_neg number = ", len(combined_neg))

combined_neg.to_csv("dbaasp_more_than100_combined_ampep_neg.csv", index=False, header=False)  # not multiplied the samples.

common = amp_pos.merge(combined_neg, on=['seq'])
# print(common.head())
combined_neg1 = pd.concat([combined_neg, common]).drop_duplicates('seq')
# print(combined_neg1.head())
combined_neg1['source2'] = combined_neg1['source']
combined_neg1['source'] = combined_neg1['source'].map({'dbaasp_neg': 'amp_negc', 'ampep_neg': 'amp_negnc'})
combined_neg1 = combined_neg1.drop(columns=['source_x', 'source_y'])
# print(combined_neg1.head())
combined_neg1 = combined_neg1[combined_neg1['seq'].str.contains('^[A-Z]+')]
combined_neg1 = combined_neg1[~combined_neg1.seq.str.contains("B")]
combined_neg1 = combined_neg1[~combined_neg1.seq.str.contains("J")]
combined_neg1 = combined_neg1[~combined_neg1.seq.str.contains("O")]
combined_neg1 = combined_neg1[~combined_neg1.seq.str.contains("U")]
combined_neg1 = combined_neg1[~combined_neg1.seq.str.contains("X")]
combined_neg1 = combined_neg1[~combined_neg1.seq.str.contains("Z")]
combine_neg1 = combined_neg1[~combined_neg1.seq.str.contains("-")]
combine_neg1 = combined_neg1[~combined_neg1.seq.str.contains('[a-z]')]
#combined_neg1=combined_neg1[~combined_neg1.seq.str.contains("*")]
combined_neg1 = combined_neg1[~combined_neg1.seq.str.contains(r'[0-9]')]
print('--> Writing combined negative sequences collected from DBAASP and AMPEP to amp_neg.csv')
combined_neg1.to_csv("amp_neg.csv", index=False, header=False)    # not multiplied the samples.

# Toxicity data
print('**** Creating Toxicity datasets ****')

# don't need toxinpred_pos as satpdb takes care of it
# toxinpred is already len <=35.
col_Names = ["seq"]
print('1. Collecting Toxicity negative samples')
toxinpred_neg1 = pd.read_csv("./data/toxicity/nontoxic_trembl_toxinnpred.txt", names=col_Names)
print ("--> toxinpred_neg1 number = ", len(toxinpred_neg1))
toxinpred_neg1["source2"] = ["toxinpred_neg_tr"] * len(toxinpred_neg1)
toxinpred_neg1 = toxinpred_neg1[["seq", "source2"]]
toxinpred_neg2 = pd.read_csv("./data/toxicity/nontoxic_swissprot_toxinnpred.txt", names=col_Names)
print ("--> toxinpred_neg2 number = ", len(toxinpred_neg2))
toxinpred_neg2["source2"] = ["toxinpred_neg_sp"] * len(toxinpred_neg2)
toxinpred_neg2 = toxinpred_neg2[["seq", "source2"]]
toxinpred_neg = pd.concat([toxinpred_neg1, toxinpred_neg2]).drop_duplicates('seq')
print('--> toxinpred_neg number = ', len(toxinpred_neg))

# valid_df_satpdb
toxic = valid_df_satpdb[valid_df_satpdb['Major.Functions'].str.contains("toxic")]
toxic = valid_df_satpdb[valid_df_satpdb['Major.Functions'].str.contains("toxic") | valid_df_satpdb['Sub.functions'].str.contains("toxic")]
print ('--> Valid toxicity sequences from Satpdb = ', len(toxic))

# for toxicity:
# dbassp
# all of them > 250 -> dbaap_neg
# all of them < 200-> dbaap_pos
#
# combined_toxic_pos = satpdb_pos + dbaap_pos
#
# combined_toxic_neg = 10*(dbaap_neg) + UNiprot0

# df from dbaasp, toxic from satpdb
print('2. Collecting Toxicity positive samples')
df_overlap_tox = pd.merge(df, toxic, on='seq', how='inner')[["seq", "toxic_list", "converted_toxic"]]
combined_toxic_pos = all_toxic_less_than(df_overlap_tox, 200)
dbaasp_toxic_pos = all_toxic_less_than(df, 200)
dbaasp_toxic_pos["source2"] = ["dbaasp"] * len(dbaasp_toxic_pos)
dbaasp_toxic_pos = dbaasp_toxic_pos[["seq", "source2"]]
toxic["source2"] = ["satpdb"]*len(toxic)
toxic = toxic[["seq", "source2"]]
combined_toxic_pos = pd.concat([dbaasp_toxic_pos, toxic]).drop_duplicates('seq')
combined_toxic_pos['source'] = 'tox_pos'
#combined_toxic_pos = combined_toxic_pos[["seq", "source", "tox"]]
combined_toxic_pos = combined_toxic_pos[["seq", "source", "source2"]]
combined_toxic_pos = combined_toxic_pos[combined_toxic_pos['seq'].str.contains('^[A-Z]+')]
combined_toxic_pos = combined_toxic_pos[~combined_toxic_pos.seq.str.contains("B")]
combined_toxic_pos = combined_toxic_pos[~combined_toxic_pos.seq.str.contains("J")]
combined_toxic_pos = combined_toxic_pos[~combined_toxic_pos.seq.str.contains("O")]
combined_toxic_pos = combined_toxic_pos[~combined_toxic_pos.seq.str.contains("U")]
combined_toxic_pos = combined_toxic_pos[~combined_toxic_pos.seq.str.contains("X")]
combined_toxic_pos = combined_toxic_pos[~combined_toxic_pos.seq.str.contains("Z")]
combined_toxic_pos = combined_toxic_pos[~combined_toxic_pos.seq.str.contains('[a-z]')]
combined_toxic_pos = combined_toxic_pos[~combined_toxic_pos.seq.str.contains("-")]
#combined_toxic_pos=combined_toxic_pos[~combined_toxic_pos.seq.str.contains("*")]
combined_toxic_pos = combined_toxic_pos[~combined_toxic_pos.seq.str.contains(r'[0-9]')]
combined_toxic_pos.to_csv("toxic_pos.csv", index=False, header=False)
print ('--> combined_toxic_pos number = ', len(combined_toxic_pos))

dbaasp_neg = all_toxic_more_than(df, 250)
dbaasp_neg["source2"] = ["dbaasp"] * len(dbaasp_neg)
dbaasp_neg['source'] = 'tox_negc'
dbaasp_neg = dbaasp_neg[["seq", "source", "source2"]]
dbaasp_neg.head()

toxinpred_neg['source'] = 'tox_negnc'
toxinpred_neg = toxinpred_neg[["seq",  "source", "source2"]]

combined_toxic_neg = pd.concat([dbaasp_neg, toxinpred_neg]).drop_duplicates('seq')
combined_toxic_neg = combined_toxic_neg[["seq", "source", "source2"]]
combined_toxic_neg.to_csv("toxic_neg_nofilter.csv", index=False, header=False)
print ('--> combined_toxic_neg number = ', len(combined_toxic_neg))

commont = combined_toxic_pos.merge(combined_toxic_neg, on=['seq'])
combined_negt1 = pd.concat([combined_toxic_neg, commont]).drop_duplicates('seq')
combined_negt1 = combined_negt1.drop(columns=['source_x', 'source_y', 'source2_x', 'source2_y'])
combined_negt1 = combined_negt1[combined_negt1['seq'].str.contains('^[A-Z]+')]
combined_negt1 = combined_negt1[~combined_negt1.seq.str.contains("B")]
combined_negt1 = combined_negt1[~combined_negt1.seq.str.contains("J")]
combined_negt1 = combined_negt1[~combined_negt1.seq.str.contains("O")]
combined_negt1 = combined_negt1[~combined_negt1.seq.str.contains("U")]
combined_negt1 = combined_negt1[~combined_negt1.seq.str.contains("X")]
combined_negt1 = combined_negt1[~combined_negt1.seq.str.contains("Z")]
combined_negt1 = combined_negt1[~combined_negt1.seq.str.contains('[a-z]')]
combined_negt1 = combined_negt1[~combined_negt1.seq.str.contains("-")]
combined_negt1 = combined_negt1[~combined_negt1.seq.str.contains(r'[0-9]')]
combined_negt1 = combined_negt1[['seq', 'source', 'source2']]
combined_negt1.to_csv("toxic_neg.csv", index=False, header=False)  # not multiplied the samples.


ampseq = pd.concat([amp_pos, combined_neg1]).drop_duplicates('seq')
ampseq.seq = ampseq.seq.apply(lambda x: " ".join(x))  # remove the space from the seq
ampseq = ampseq.sample(frac=1)
ampseq.columns = ['text', 'amp', 'source']
ampseq.to_csv("amp_lab_.csv", index=False, header=True)

toxseq = pd.concat([combined_toxic_pos, combined_negt1]).drop_duplicates('seq')
toxseq.seq = toxseq.seq.apply(lambda x: " ".join(x))  # remove the space from the seq
toxseq = toxseq.sample(frac=1)
toxseq.columns = ['text', 'tox', 'source']
toxseq.to_csv("tox_lab_.csv", index=False, header=True)

# Solubility dataset creation
print('***** Creating solubility dataset ***** ')
col_Names = ["seq", "source", "source2"]
sol_train = pd.read_csv("./data/solubility/sol_lab_train.csv", names=col_Names)
sol_val = pd.read_csv("./data/solubility/sol_lab_valid.csv", names=col_Names)
sol_test = pd.read_csv("./data/solubility/sol_lab_test.csv", names=col_Names)

solseq = pd.concat([sol_train, sol_val, sol_test])
solseq_short = solseq[solseq['seq'].apply(lambda x: len(x) <= 50)]
solseq = solseq_short
solseq.seq = solseq.seq.apply(lambda x: " ".join(x))  # remove the space from the seq
solseq = solseq.sample(frac=1)
solseq.columns = ['text', 'sol', 'source']
print('1. Writing solubility labeled data to sol_lab.csv')
solseq.to_csv("sol_lab_.csv", index=False, header=True)

uniprot_unk.columns = ['text', 'source', 'source2']
ampseq.columns = ['text', 'source', 'source2']
toxseq.columns = ['text', 'source', 'source2']
solseq.columns = ['text', 'source', 'source2']
ampseq.text = ampseq.text.apply(lambda x: "".join(x.split()))  # remove the space from the seq
toxseq.text = toxseq.text.apply(lambda x: "".join(x.split()))
solseq.text = solseq.text.apply(lambda x: "".join(x.split()))
allseq = pd.concat([uniprot_unk, ampseq, toxseq, solseq]).drop_duplicates('text')
#print (len(allseq))
allseq = allseq[allseq['text'].str.contains('^[A-Z]+')]
allseq = allseq[~allseq.text.str.contains("B")]
allseq = allseq[~allseq.text.str.contains("J")]
allseq = allseq[~allseq.text.str.contains("O")]
allseq = allseq[~allseq.text.str.contains("U")]
allseq = allseq[~allseq.text.str.contains("X")]
allseq = allseq[~allseq.text.str.contains("Z")]
allseq = allseq[~allseq.text.str.contains("-")]
allseq = allseq[~allseq.text.str.contains('[a-z]')]
allseq = allseq[~allseq.text.str.contains(r'[0-9]')]
allseq.text = allseq.text.apply(lambda x: " ".join(x))
allseq = allseq.sample(frac=1)
allseq = allseq[['text', 'source', 'source2']]
allseq.columns = ['text', 'lab_dummy', 'source']
allseq.to_csv("unlab_.csv", index=False, header=True)
