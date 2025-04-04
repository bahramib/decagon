from __future__ import division
from __future__ import print_function
from operator import itemgetter
from itertools import combinations
import time
import os
import random

import pandas as pd
import tensorflow as tf
import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn import metrics

from decagon.deep.optimizer import DecagonOptimizer
from decagon.deep.model import DecagonModel
from decagon.deep.minibatch import EdgeMinibatchIterator
from decagon.utility import rank_metrics, preprocessing

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Train on GPU
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

np.random.seed(0)

###########################################################
#
# Functions
#
###########################################################

def load_decagon_data(use_dummy=False, ppi_path='polypharmacy/bio-decagon-ppi.csv', 
                     drug_target_path='polypharmacy/bio-decagon-targets-all.csv',
                     combo_path='polypharmacy/bio-decagon-combo.csv'):
    """
    Load and preprocess data for Decagon model from the bio-decagon datasets.
    
    Args:
        use_dummy: If True, use synthetic dummy data. If False, load real data.
        ppi_path: Path to protein-protein interaction CSV file
        drug_target_path: Path to drug-target interaction CSV file
        combo_path: Path to drug-drug interaction CSV file
    
    Returns:
        Processed data in the format expected by Decagon
    """
    
    if use_dummy:
        # Use the synthetic dummy data
        val_test_size = 0.05
        n_genes = 19080
        n_drugs = 645
        n_drugdrug_rel_types = 1317
        gene_net = nx.planted_partition_graph(360, 53, 0.1, 0.01, seed=42)
        gene_adj = nx.adjacency_matrix(gene_net)
        gene_degrees = np.array(gene_adj.sum(axis=0)).squeeze()
        gene_drug_adj = sp.csr_matrix((10 * np.random.randn(n_genes, n_drugs) > 15).astype(int))
        drug_gene_adj = gene_drug_adj.transpose(copy=True)
        drug_drug_adj_list = []
        tmp = np.dot(drug_gene_adj, gene_drug_adj)
        
        for i in range(n_drugdrug_rel_types):
            if i % 15 == 0:
                print("Round %d with side effect %s" % (i, i))
            mat = np.zeros((n_drugs, n_drugs))
            for d1, d2 in combinations(list(range(n_drugs)), 2):
                if random.randint(0, 1000) < 3:
                    mat[d1, d2] = mat[d2, d1] = 1.
            drug_drug_adj_list.append(sp.csr_matrix(mat))
        drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list]
        
    else:
        # Load real data from the CSV files
        from polypharmacy.utility import load_ppi, load_targets, load_combo_se
        
        print("Loading PPI (Protein-Protein Interactions)...")
        ppi_network, gene2idx = load_ppi(ppi_path)
        n_genes = len(gene2idx)
        gene_adj = nx.adjacency_matrix(ppi_network)
        gene_degrees = np.array(gene_adj.sum(axis=0)).squeeze()
        
        print("Loading Drug-Target Interactions...")
        drug2proteins = load_targets(drug_target_path)
        
        print("Loading Drug-Drug Interactions...")
        drug2drug, drug2se, se2name = load_combo_se(combo_path)
        
        # Extract all unique drugs
        all_drugs = set()
        for combo, [drug1, drug2] in drug2drug.items():
            all_drugs.add(drug1)
            all_drugs.add(drug2)
        
        # Create mapping for drugs (STITCH IDs) to indices
        drug2idx = {drug: i for i, drug in enumerate(all_drugs)}
        n_drugs = len(drug2idx)
        
        print("Number of unique genes/proteins: %d" % n_genes)
        print("Number of unique drugs: %d" % n_drugs)
        
        # Group side effects and create a mapping
        unique_se = set()
        for se_set in drug2se.values():
            unique_se.update(se_set)
        se2idx = {se: i for i, se in enumerate(unique_se)}
        n_se_types = len(se2idx)
        
        print("Number of unique side effect types: %d" % n_se_types)
        
        # Create drug-gene adjacency matrix (n_drugs x n_genes)
        gene_drug_adj = sp.lil_matrix((n_genes, n_drugs))
        for drug, proteins in drug2proteins.items():
            if drug in drug2idx:
                drug_idx = drug2idx[drug]
                for protein in proteins:
                    if protein in gene2idx:
                        gene_idx = gene2idx[protein]
                        gene_drug_adj[gene_idx, drug_idx] = 1
        
        gene_drug_adj = gene_drug_adj.tocsr()
        drug_gene_adj = gene_drug_adj.transpose(copy=True)
        
        # Create drug-drug adjacency matrices for each side effect type
        drug_drug_adj_list = []
        count = 0
        for se in unique_se:
            count += 1
            if count % 15 == 0:
                print("Round %d with side effect %s" % (count, se))
            mat = sp.lil_matrix((n_drugs, n_drugs))
            for combo, se_set in drug2se.items():
                if se in se_set:
                    drug1, drug2 = drug2drug[combo]
                    if drug1 in drug2idx and drug2 in drug2idx:
                        d1_idx = drug2idx[drug1]
                        d2_idx = drug2idx[drug2]
                        mat[d1_idx, d2_idx] = 1
                        mat[d2_idx, d1_idx] = 1  # symmetric
            drug_drug_adj_list.append(mat.tocsr())
        
        drug_degrees_list = [np.array(adj.sum(axis=0)).squeeze() for adj in drug_drug_adj_list]
    
    # Prepare data representation (same for both dummy and real data)
    val_test_size = 0.05
    
    adj_mats_orig = {
        (0, 0): [gene_adj, gene_adj.transpose(copy=True)],
        (0, 1): [gene_drug_adj],
        (1, 0): [drug_gene_adj],
        (1, 1): drug_drug_adj_list + [x.transpose(copy=True) for x in drug_drug_adj_list],
    }
    degrees = {
        0: [gene_degrees, gene_degrees],
        1: drug_degrees_list + drug_degrees_list,
    }
    
    # Generate features (identity matrices)
    gene_feat = sp.identity(n_genes)
    gene_nonzero_feat, gene_num_feat = gene_feat.shape
    gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())
    
    drug_feat = sp.identity(n_drugs)
    drug_nonzero_feat, drug_num_feat = drug_feat.shape
    drug_feat = preprocessing.sparse_to_tuple(drug_feat.tocoo())
    
    num_feat = {
        0: gene_num_feat,
        1: drug_num_feat,
    }
    nonzero_feat = {
        0: gene_nonzero_feat,
        1: drug_nonzero_feat,
    }
    feat = {
        0: gene_feat,
        1: drug_feat,
    }
    
    edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.items()}
    edge_type2decoder = {
        (0, 0): 'bilinear',
        (0, 1): 'bilinear',
        (1, 0): 'bilinear',
        (1, 1): 'dedicom',
    }
    
    edge_types = {k: len(v) for k, v in adj_mats_orig.items()}
    num_edge_types = sum(edge_types.values())
    print("Edge types:", "%d" % num_edge_types)
    
    return val_test_size, adj_mats_orig, degrees, num_feat, nonzero_feat, feat, edge_type2dim, edge_type2decoder, edge_types, num_edge_types

def get_accuracy_scores(edges_pos, edges_neg, edge_type):
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
    feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
    feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
    rec = sess.run(opt.predictions, feed_dict=feed_dict)

    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    actual = []
    predicted = []
    edge_ind = 0
    for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 1, 'Problem 1'

        actual.append(edge_ind)
        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_neg = []
    for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds_neg.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 0, 'Problem 0'

        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_all = np.hstack([preds, preds_neg])
    preds_all = np.nan_to_num(preds_all)
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]

    roc_sc = metrics.roc_auc_score(labels_all, preds_all)
    aupr_sc = metrics.average_precision_score(labels_all, preds_all)
    apk_sc = rank_metrics.apk(actual, predicted, k=50)

    return roc_sc, aupr_sc, apk_sc

def construct_placeholders(edge_types):
    placeholders = {
        'batch': tf.placeholder(tf.int32, name='batch'),
        'batch_edge_type_idx': tf.placeholder(tf.int32, shape=(), name='batch_edge_type_idx'),
        'batch_row_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_row_edge_type'),
        'batch_col_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_col_edge_type'),
        'degrees': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
    }
    placeholders.update({
        'adj_mats_%d,%d,%d' % (i, j, k): tf.sparse_placeholder(tf.float32)
        for i, j in edge_types for k in range(edge_types[i,j])})
    placeholders.update({
        'feat_%d' % i: tf.sparse_placeholder(tf.float32)
        for i, _ in edge_types})
    return placeholders

###########################################################
#
# Load and preprocess data (This is a dummy toy example!)
#
###########################################################

####
# The following code uses artificially generated and very small networks.
# Expect less than excellent performance as these random networks do not have any interesting structure.
# The purpose of main.py is to show how to use the code!
#
# All preprocessed datasets used in the drug combination study are at: http://snap.stanford.edu/decagon:
# (1) Download datasets from http://snap.stanford.edu/decagon to your local machine.
# (2) Replace dummy toy datasets used here with the actual datasets you just downloaded.
# (3) Train & test the model.
####

from polypharmacy.utility import *

# Load data (True for dummy, False for real data)
val_test_size, adj_mats_orig, degrees, num_feat, nonzero_feat, feat, edge_type2dim, edge_type2decoder, edge_types, num_edge_types = load_decagon_data(use_dummy=True)

###########################################################
#
# Settings and placeholders
#
###########################################################

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('neg_sample_size', 1, 'Negative sample size.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('max_margin', 0.1, 'Max margin parameter in hinge loss')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_boolean('bias', True, 'Bias term.')
# Important -- Do not evaluate/print validation performance every iteration as it can take
# substantial amount of time
PRINT_PROGRESS_EVERY = 150

print("Defining placeholders")
placeholders = construct_placeholders(edge_types)

###########################################################
#
# Create minibatch iterator, model and optimizer
#
###########################################################

print("Create minibatch iterator")
minibatch = EdgeMinibatchIterator(
    adj_mats=adj_mats_orig,
    feat=feat,
    edge_types=edge_types,
    batch_size=FLAGS.batch_size,
    val_test_size=val_test_size
)

print("Create model")
model = DecagonModel(
    placeholders=placeholders,
    num_feat=num_feat,
    nonzero_feat=nonzero_feat,
    edge_types=edge_types,
    decoders=edge_type2decoder,
)

print("Create optimizer")
with tf.name_scope('optimizer'):
    opt = DecagonOptimizer(
        embeddings=model.embeddings,
        latent_inters=model.latent_inters,
        latent_varies=model.latent_varies,
        degrees=degrees,
        edge_types=edge_types,
        edge_type2dim=edge_type2dim,
        placeholders=placeholders,
        batch_size=FLAGS.batch_size,
        margin=FLAGS.max_margin
    )

print("Initialize session")
sess = tf.Session()
sess.run(tf.global_variables_initializer())
feed_dict = {}

###########################################################
#
# Train model
#
###########################################################

print("Train model")
for epoch in range(FLAGS.epochs):

    minibatch.shuffle()
    itr = 0
    while not minibatch.end():
        # Construct feed dictionary
        feed_dict = minibatch.next_minibatch_feed_dict(placeholders=placeholders)
        feed_dict = minibatch.update_feed_dict(
            feed_dict=feed_dict,
            dropout=FLAGS.dropout,
            placeholders=placeholders)

        t = time.time()

        # Training step: run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.batch_edge_type_idx], feed_dict=feed_dict)
        train_cost = outs[1]
        batch_edge_type = outs[2]

        if itr % PRINT_PROGRESS_EVERY == 0:
            val_auc, val_auprc, val_apk = get_accuracy_scores(
                minibatch.val_edges, minibatch.val_edges_false,
                minibatch.idx2edge_type[minibatch.current_edge_type_idx])

            print("Epoch:", "%04d" % (epoch + 1), "Iter:", "%04d" % (itr + 1), "Edge:", "%04d" % batch_edge_type,
                  "train_loss=", "{:.5f}".format(train_cost),
                  "val_roc=", "{:.5f}".format(val_auc), "val_auprc=", "{:.5f}".format(val_auprc),
                  "val_apk=", "{:.5f}".format(val_apk), "time=", "{:.5f}".format(time.time() - t))

        itr += 1

print("Optimization finished!")

for et in range(num_edge_types):
    roc_score, auprc_score, apk_score = get_accuracy_scores(
        minibatch.test_edges, minibatch.test_edges_false, minibatch.idx2edge_type[et])
    print("Edge type=", "[%02d, %02d, %02d]" % minibatch.idx2edge_type[et])
    print("Edge type:", "%04d" % et, "Test AUROC score", "{:.5f}".format(roc_score))
    print("Edge type:", "%04d" % et, "Test AUPRC score", "{:.5f}".format(auprc_score))
    print("Edge type:", "%04d" % et, "Test AP@k score", "{:.5f}".format(apk_score))
    print()
