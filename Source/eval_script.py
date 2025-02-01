#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:46:20 2020

@author: eschweiler
"""

import os
import glob

from argparse import ArgumentParser

from utils.utils import print_timestamp


def main(hparams):
    
    # Get the evaluator
    if hparams.type.lower() == 'instance':
        from evaluation.eval_instance_segmentation import eval_instance_segmentation as evaluator
    elif hparams.type.lower() == 'synthetic':
        from evaluation.eval_synthetic_data import eval_synthetic_data as evaluator
    else:
        raise NotImplementedError('The evaluation type "{0}" is not implemented yet!'.format(hparams.type.lower()))
        
    # Get the groundtruth and prediction files
    if 'Willis' in hparams.data_root:
        from evaluation.get_filelists import get_Willis_plant as filelist_generator
        print_timestamp('Loading Willis data set...')
        
    elif 'Refahi'in hparams.data_root:
        from evaluation.get_filelists import get_Refahi_all as filelist_generator
        print_timestamp('Loading Refahi data set...')
        
    elif 'Moreno'in hparams.data_root:
        print_timestamp('Loading Moreno data set...')
        from evaluation.get_filelists import get_Moreno_selection as filelist_generator
        
    else:
        print_timestamp('Loading unspecific data set...')
        from evaluation.get_filelists import get_generic_filelist as filelist_generator
        
    
    gt_files, pred_files = filelist_generator(data_root=hparams.data_root, gt_identifiers=hparams.gt_identifiers,\
                                              result_path=hparams.result_path, result_identifier=hparams.result_identifier,\
                                              subfolder=hparams.subfolder)
        
    # Start the evaluation
    assert len(gt_files)==len(pred_files), 'Number of files do not match ({0} vs {1})...'.format(len(gt_files), len(pred_files))
    print_timestamp('Evaluating {0} files...', [len(gt_files)])
    evaluator(gt_files, pred_files, safety_margin=hparams.safety_margin, threshs=hparams.threshs, max_size=hparams.max_size,\
              save_path=hparams.result_path, bg_label=hparams.bg_label, accumulator_name=hparams.accumulator_name,\
              adjust_range=hparams.adjust_range, identifier=hparams.result_identifier+'_')
        
        

if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    parent_parser = ArgumentParser(add_help=False)
    
    parent_parser.add_argument(
        '--type',
        type=str,
        default='instance',
        help='Which type of evaluation (Instance | Synthetic)'
    )

    parent_parser.add_argument(
        '--result_path',
        type=str,
        default=r'/work/scratch/username/results/fancy_experiment',
        help='Directory to the predictions'
    )
    
    parent_parser.add_argument(
        '--data_root',
        type=str,
        default=r'/netshares/BiomedicalImageAnalysis/Resources/fancy_dataset',
        help='Directory to the ground truth data'
    )
    
    parent_parser.add_argument(
        '--result_identifier',
        type=str,
        default='',
        help='Identifier for result data'
    )
        
    parent_parser.add_argument(
        '--gt_identifiers',
        type=str,
        default=('',),
        help='Identifiers for ground truth data',
        nargs='+'
    )
        
    parent_parser.add_argument(
        '--subfolder',
        type=str,
        default='',
        help='Subfolder pattern'
    )
    
    parent_parser.add_argument(
        '--bg_label',
        type=int,
        default=0,
        help='Background label'
    ) 
    
    parent_parser.add_argument(
        '--safety_margin',
        type=int,
        default=(5,5,5),
        help='Safety margin at data boudaries',
        nargs='+'
    )
    
    parent_parser.add_argument(
        '--threshs',
        type=int,
        default=None,
        help='Thresholds',
        nargs='+'
    )
    
    parent_parser.add_argument(
        '--max_size',
        type=int,
        default=1e10,
        help='Background label'
    )
    
    parent_parser.add_argument(
        '--adjust_range',
        dest='adjust_range',
        action='store_true',
        default=False,
        help='Adjust dynamic intensity range for synthetic image data'
    )
    
    parent_parser.add_argument(
        '--accumulator_name',
        type=str,
        default='avg_precision',
        help='Name of the score accumulator class'
    )
    
        
    hyperparams = parent_parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
