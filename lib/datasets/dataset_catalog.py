# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Collection of available datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from core.config import cfg

# Path to data dir
_DATA_DIR = cfg.DATA_DIR

# Required dataset entry keys
IM_DIR = 'image_directory'
ANN_FN = 'annotation_file'

# Optional dataset entry keys
IM_PREFIX = 'image_prefix'
DEVKIT_DIR = 'devkit_directory'
RAW_DIR = 'raw_dir'

# Available datasets
DATASETS = {
    # CS6 noisy ground truth
    'cs6_noise020' : {
        IM_DIR: _DATA_DIR + '/CS6_annot/',
        ANN_FN: _DATA_DIR + '/cs6_jsons/cs6-train-gt_face_train_annot_coco_style_noisy-020.json'
     },
    'cs6_noise030' : {
        IM_DIR: _DATA_DIR + '/CS6_annot/',
        ANN_FN: _DATA_DIR + '/cs6_jsons/cs6-train-gt_face_train_annot_coco_style_noisy-030.json'
    },
    'cs6_noise040' : {
        IM_DIR: _DATA_DIR + '/CS6_annot/',
        ANN_FN: _DATA_DIR + '/cs6_jsons/cs6-train-gt_face_train_annot_coco_style_noisy-040.json'
    },
    'cs6_noise050' : {
        IM_DIR: _DATA_DIR + '/CS6_annot/',
        ANN_FN: _DATA_DIR + '/cs6_jsons/cs6-train-gt_face_train_annot_coco_style_noisy-050.json'
    },
    'cs6_noise060' : {
        IM_DIR: _DATA_DIR + '/CS6_annot/',
        ANN_FN: _DATA_DIR + '/cs6_jsons/cs6-train-gt_face_train_annot_coco_style_noisy-060.json'
    },
    'cs6_noise070' : {
        IM_DIR: _DATA_DIR + '/CS6_annot/',
        ANN_FN: _DATA_DIR + '/cs6_jsons/cs6-train-gt_face_train_annot_coco_style_noisy-070.json'
    },
    'cs6_noise080' : {
        IM_DIR: _DATA_DIR + '/CS6_annot/',
        ANN_FN: _DATA_DIR + '/cs6_jsons/cs6-train-gt_face_train_annot_coco_style_noisy-080.json'
    },
    'cs6_noise085' : {
        IM_DIR: _DATA_DIR + '/CS6_annot/',
        ANN_FN: _DATA_DIR + '/cs6_jsons/cs6-train-gt_face_train_annot_coco_style_noisy-085.json'
    },
    'cs6_noise090' : {
        IM_DIR: _DATA_DIR + '/CS6_annot/',
        ANN_FN: _DATA_DIR + '/cs6_jsons/cs6-train-gt_face_train_annot_coco_style_noisy-090.json'
    },
    'cs6_noise095' : {
        IM_DIR: _DATA_DIR + '/CS6_annot/',
        ANN_FN: _DATA_DIR + '/cs6_jsons/cs6-train-gt_face_train_annot_coco_style_noisy-095.json'
    },
    'cs6_noise100' : {
        IM_DIR: _DATA_DIR + '/CS6_annot/',
        ANN_FN: _DATA_DIR + '/cs6_jsons/cs6-train-gt_face_train_annot_coco_style_noisy-100.json'
    },

     # CS6 ground truth
    'cs6_train_gt' : {
        IM_DIR: _DATA_DIR + '/CS6_annot/',
        ANN_FN: _DATA_DIR + '/cs6_jsons/cs6-train-gt_face_train_annot_coco_style.json'
    },
     # CS6 evaluation datasets
     'cs6_annot_eval_val-easy' : {
        IM_DIR: _DATA_DIR + '/CS6_annot/',
        ANN_FN: _DATA_DIR + '/cs6_jsons/cs6_gt_annot_val-easy.json'
     },
    'cs6_TEST_gt' : {
        IM_DIR: _DATA_DIR + '/CS6_annot/',
        ANN_FN: _DATA_DIR + '/cs6_jsons/cs6-test-gt.json'
    },
    
    # Detection dataset created by removing the traker output from the HP json
    'cs6_train_det_from_hp' : {
        IM_DIR: _DATA_DIR + '/CS6_annot/',
        ANN_FN: _DATA_DIR + '/cs6_jsons/cs6_train_det_derived_from_hp.json'
    },
    # Dataset with only HP: removed the detections from the HP json
    'cs6_train_hp_tracker_only':{
        IM_DIR: _DATA_DIR + '/CS6_annot/',
        ANN_FN: _DATA_DIR + '/cs6_jsons/cs6_train_hp_tracker_only.json'
    },
    # HP dataset with noisy labels: used to prevent the DA models using any HP info
    'cs6_train_hp_noisy_100' : {
        IM_DIR: _DATA_DIR + '/CS6_annot/',
        ANN_FN: _DATA_DIR + '/cs6_jsons/cs6_train_hp_noisy_100.json' 
    },

     # Pedestrian datasets
     'bdd_peds_full_train' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN: _DATA_DIR + '/bdd_jsons/bdd_peds_full_train.json'
     },
    'bdd_peds_full_val' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN: _DATA_DIR + '/bdd_jsons/bdd_peds_full_val.json'
     },
     'bdd_peds_train' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN: _DATA_DIR + '/bdd_jsons/bdd_peds_train.json'
     },
     'bdd_peds_val' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN: _DATA_DIR + '/bdd_jsons/bdd_peds_val.json'
     },
     'cityscapes_peds_train' : {
         IM_DIR:
            _DATA_DIR + '/cityscapes',
         ANN_FN: '/mnt/nfs/work1/elm/pchakrabarty/cityscapes_jsons/cityscapes_peds_train.json'
     },
     'cityscapes_peds_val' : {
        IM_DIR:
            _DATA_DIR + '/cityscapes',
        ANN_FN: '/mnt/nfs/work1/elm/pchakrabarty/cityscapes_jsons/cityscapes_peds_val.json'
     },
    
    #7-class datasets
    'bdd_clear_any_daytime_train' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN: '/srv/data1/pchakrabarty/bdd_jsons/bdd_weather-clear_scene-Any_timeofday-daytime_train.json'
    },
    'bdd_clear_any_daytime_val' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN: '/srv/data1/pchakrabarty/bdd_jsons/bdd_weather-clear_scene-Any_timeofday-daytime_val.json'
    },

    'bdd_any_any_daytime_train' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN: '/srv/data1/pchakrabarty/bdd_jsons/bdd_weather-Any_scene-Any_timeofday-daytime_train.json'
    },
    'bdd_any_any_daytime_val' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN: '/srv/data1/pchakrabarty/bdd_jsons/bdd_weather-Any_scene-Any_timeofday-daytime_val.json'
    },   

    'bdd_any_any_any_train' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN: '/srv/data1/pchakrabarty/bdd_jsons/bdd_weather-Any_scene-Any_timeofday-Any_train.json'
    },
    'bdd_any_any_any_val' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN: '/srv/data1/pchakrabarty/bdd_jsons/bdd_weather-Any_scene-Any_timeofday-Any_val.json'
    },
    
    # bdd peds dets on 20k target domain videos
    'bdd_peds_dets_20k_target_domain' : {
        IM_DIR:
            _DATA_DIR + '/bdd_detections_20k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_peds_clear_any_daytime_det_conf080.json'
    },

    # bdd peds HP on partial (13k) target domain videos
    'bdd_peds_HP_target_domain' : {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_peds_clear_any_daytime_HP.json'
    },

    # bdd peds HP on 18k target domain videos
    'bdd_peds_HP18k_target_domain' : {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_HP18k.json'
    },
    'bdd_HP18k_track_only' : {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_HP18k_track_only.json'    
    },
    # bdd peds HP on 18k target videos -- different conf thresholds theta
    'bdd_HP18k_thresh-050' : {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_HP18k_thresh-050.json'
    },
    'bdd_HP18k_thresh-060' : {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_HP18k_thresh-060.json'
    },
    'bdd_HP18k_thresh-070' : {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_HP18k_thresh-070.json'
    },
    'bdd_HP18k_thresh-090' : {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_HP18k_thresh-090.json'
    },
    
    # Subsets extracted from  bdd peds HP on 18k target domain 
    # night images only
    'bdd_HP18k_any_any_night' : {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_HP18k_any_any_night.json'  
    },
    'bdd_HP18k_any_any_night_val': {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_HP18k_any_any_night_val.json'
    },
    # rainy, day
    'bdd_HP18k_rainy_any_daytime' : {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_HP18k_rainy_any_daytime.json'
    },
    'bdd_peds_rainy_any_daytime_val' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_peds_rainy_any_daytime_val.json'
    },
    # rainy, night
    'bdd_HP18k_rainy_any_night' : {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_HP18k_rainy_any_night.json'
    },
    'bdd_peds_rainy_any_night_val' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k', 
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_peds_rainy_any_night_val.json'
    },
    # overcast, day
    'bdd_HP18k_overcast_any_daytime' : {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_HP18k_overcast_any_daytime.json'
    },
    'bdd_peds_overcast_any_daytime_val' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_peds_overcast_any_daytime_val.json'
    },
    # overcast, night
    'bdd_HP18k_overcast_any_night' : {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_HP18k_overcast_any_night.json'
    }, 
    'bdd_peds_overcast_any_night_val' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_peds_overcast_any_night_val.json'
    },
    # snowy, day
    'bdd_HP18k_snowy_any_daytime' : {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_HP18k_snowy_any_daytime.json'
    },
    'bdd_peds_snowy_any_daytime_val' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_peds_snowy_any_daytime_val.json'
    },
    # snowy, night
    'bdd_HP18k_snowy_any_night' : {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_HP18k_snowy_any_night.json'
    },
    'bdd_peds_snowy_any_night_val' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_peds_snowy_any_night_val.json'
    },
    # overcast, rainy, day
    'bdd_HP18k_overcast,rainy_any_daytime' : {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_HP18k_overcast,rainy_any_daytime.json'  
    },
    'bdd_peds_overcast,rainy_any_daytime_val' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k', 
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_peds_overcast,rainy_any_daytime_val.json'
    },
    # overcast, rainy, night
    'bdd_HP18k_overcast,rainy_any_night' : {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_HP18k_overcast,rainy_any_night.json'
    },
    'bdd_peds_overcast,rainy_any_night_val' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_peds_overcast,rainy_any_night_val.json'
    },
    # overcast, rainy, snowy day
    'bdd_HP18k_overcast,rainy,snowy_any_daytime' : {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_HP18k_overcast,rainy,snowy_any_daytime.json'
    },
    'bdd_peds_overcast,rainy,snowy_any_daytime_val' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_peds_overcast,rainy,snowy_any_daytime_val.json'
    },
    # overcast, rainy, snowy night
    'bdd_HP18k_overcast,rainy,snowy_any_night' : {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_peds_overcast,rainy,snowy_any_night.json'
    },
    'bdd_peds_overcast,rainy,snowy_any_night_val' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_peds_overcast,rainy,snowy_any_night_val.json'
    },
    #### end of bdd_HP18k subsets ####

    # BDD HP18k after histogram remapping
    'bdd_peds_HP18k_target_domain_remap_hist': {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_HP18k_remap_hist.json'
    },
    # BDD HP18k after random remapping
    'bdd_peds_HP18k_target_domain_remap_random': {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_HP18k_remap_random.json'
    },
    # BDD HP18k with remapping accroding to the Cityscapes baseline model
    'bdd_peds_HP18k_target_domain_remap_cityscape_hist': {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_HP18k_remap_cityscapes_hist.json'
    },
    # noisy bdd HP on 18k target domain videos -- used to prevent domain adv roi using HP info.
    'bdd_HP18k_noisy_100k' : {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_HP18k_noisy_100.json'    
    },
    'bdd_HP18k_noisy_060' : {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_HP18k_noisy_060.json' # noise level 0.6
    },
    'bdd_HP18k_noisy_070' : {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_HP18k_noisy_070.json' # noise level 0.7
    },
    'bdd_HP18k_noisy_080' : {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_HP18k_noisy_080.json' # noise level 0.8
    },

    # bdd peds dets on the same images as HP18k samples
    'bdd_peds_dets18k_target_domain' : {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_dets18k.json'
    },

    # bdd peds -- complement of clear_any_daytime
    'bdd_peds_not_clear_any_daytime_train' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN: _DATA_DIR + '/bdd_jsons/bdd_peds_not_clear_any_daytime_train.json'
    },
    'bdd_peds_not_clear_any_daytime_val' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN: _DATA_DIR + '/bdd_jsons/bdd_peds_not_clear_any_daytime_val.json'
    },

    ##### Varying amounts of BDD target domain labeled data #####
    'bdd_peds_not_clear_any_daytime_train_100' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN: _DATA_DIR + '/bdd_jsons/vary_pure_target/bdd_peds_not_clear_any_daytime_train_subsampled_100.json'
    },
    'bdd_peds_not_clear_any_daytime_train_075' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN: _DATA_DIR + '/bdd_jsons/vary_pure_target/bdd_peds_not_clear_any_daytime_train_subsampled_075.json'
    },
    'bdd_peds_not_clear_any_daytime_train_050' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN: _DATA_DIR + '/bdd_jsons/vary_pure_target/bdd_peds_not_clear_any_daytime_train_subsampled_050.json'
    },
    'bdd_peds_not_clear_any_daytime_train_025' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN: _DATA_DIR + '/bdd_jsons/vary_pure_target/bdd_peds_not_clear_any_daytime_train_subsampled_025.json'
    },
    'bdd_peds_not_clear_any_daytime_train_010' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN: _DATA_DIR + '/bdd_jsons/vary_pure_target/bdd_peds_not_clear_any_daytime_train_subsampled_010.json'
    },
    'bdd_peds_not_clear_any_daytime_train_005' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN: _DATA_DIR + '/bdd_jsons/vary_pure_target/bdd_peds_not_clear_any_daytime_train_subsampled_005.json'
    },
    'bdd_peds_not_clear_any_daytime_train_001' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN: _DATA_DIR + '/bdd_jsons/vary_pure_target/bdd_peds_not_clear_any_daytime_train_subsampled_001.json'
    },
    #########################################################
   
    ##### Jsons for the rebuttal: data distillation on BDD #####
    'bdd_data_dist_small' : {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/data_dist/bdd_data_dist_small.json'
    },
    'bdd_data_dist_mid' : {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/data_dist/bdd_data_dist_mid.json'
    },
    'bdd_data_dist' : {
        IM_DIR:
            _DATA_DIR + '/bdd_peds_HP18k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/data_dist/bdd_data_dist.json'
    },
    
    #########################################################
   
    
    # Ashish's 20k sampled video
    'bdd_peds_not_clear_any_daytime_20k_train' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN: _DATA_DIR + '/bdd_jsons/bdd_peds_not_clear_any_daytime_20k_train.json'
    },
    
    # BDD peds test set
    'bdd_peds_TEST' : {
        IM_DIR:
            _DATA_DIR + '/bdd100k',
        ANN_FN:
            _DATA_DIR + '/bdd_jsons/bdd_peds_TEST.json'
    },
    # Cityscapes JSONS: peds and cars
    'cityscapes_peds_train': {
        IM_DIR:
            _DATA_DIR + '/cityscapes',
        ANN_FN:
            _DATA_DIR + '/cityscapes_jsons/cityscapes_peds_train.json'
    },
    'cityscapes_peds_val': {
        IM_DIR:
            _DATA_DIR + '/cityscapes',
        ANN_FN:
            _DATA_DIR + '/cityscapes_jsons/cityscapes_peds_val.json'
    },
    'cityscapes_car_train': {
        IM_DIR:
            _DATA_DIR + '/cityscapes',
        ANN_FN:
            _DATA_DIR + '/cityscapes_jsons/cityscapes_car_train.json'
        },
    'cityscapes_car_val': {
        IM_DIR:
            _DATA_DIR + '/cityscapes',
        ANN_FN:
            _DATA_DIR + '/cityscapes_jsons/cityscapes_car_val.json'
    },
    'cityscapes_cars_HPlen5': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/cityscapes_cars_HP_frames',
        ANN_FN:
            _DATA_DIR + '/cityscapes_jsons/cityscapes_cars_HPlen5.json'
    },
    'cityscapes_cars_HPlen3': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/cityscapes_cars_HP_frames',
        ANN_FN:
            _DATA_DIR + '/cityscapes_jsons/cityscapes_cars_HPlen3.json'
    },
    ####################################

    # KITTI JSONS: Cars
    'kitti_car_train': {
        IM_DIR:
            _DATA_DIR + '/KITTI',
        ANN_FN:
            #_DATA_DIR + '/KITTI_jsons/kitti_car_train_annot_coco_style.json'
            _DATA_DIR + '/KITTI_jsons/kitti_car_train.json'
    },
    ###################################

    'cityscapes_fine_instanceonly_seg_train': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_train.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_val': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        # use filtered validation as there is an issue converting contours
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_test': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_test.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'coco_2014_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2014.json'
    },
    'coco_2014_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2014.json'
    },
    'coco_2014_minival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_minival2014.json'
    },
    'coco_2014_valminusminival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_valminusminival2014.json'
    },
    'coco_2015_test': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    },
    'coco_2015_test-dev': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },
    'coco_2017_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2017.json',
    },
    'coco_2017_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2017.json',
    },
    'coco_2017_test': {  # 2017 test uses 2015 test images
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json',
        IM_PREFIX:
            'COCO_test2015_'
    },
    'coco_2017_test-dev': {  # 2017 test-dev uses 2015 test images
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2017.json',
        IM_PREFIX:
            'COCO_test2015_'
    },
    'coco_stuff_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/stuff_train.json'
    },
    'coco_stuff_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/stuff_val.json'
    },
    'keypoints_coco_2014_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_train2014.json'
    },
    'keypoints_coco_2014_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_val2014.json'
    },
    'keypoints_coco_2014_minival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_minival2014.json'
    },
    'keypoints_coco_2014_valminusminival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_valminusminival2014.json'
    },
    'keypoints_coco_2015_test': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    },
    'keypoints_coco_2015_test-dev': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },
    'keypoints_coco_2017_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_train2017.json'
    },
    'keypoints_coco_2017_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_val2017.json'
    },
    'keypoints_coco_2017_test': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json'
    },
    'voc_2007_trainval': {
        IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_trainval.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },
    'voc_2007_test': {
        IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_test.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },
    'voc_2012_trainval': {
        IM_DIR:
            _DATA_DIR + '/VOC2012/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOC2012/annotations/voc_2012_trainval.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOC2012/VOCdevkit2012'
    },

    # WIDER DATASET
    'wider_train': {
        IM_DIR:
            _DATA_DIR + '/WIDER',
        ANN_FN:
            _DATA_DIR + '/WIDER/wider_face_train_annot_coco_style.json',
    },
    'wider_val' : {
        IM_DIR:
            _DATA_DIR + '/WIDER',
        ANN_FN:
            _DATA_DIR + '/WIDER/wider_face_val_annot_coco_style.json'
    },


    # CS6 DATASET SUBSET
    'cs6-subset': {
        IM_DIR:
            _DATA_DIR + '/CS6_annot',
        ANN_FN:
            _DATA_DIR + '/CS6_annot/cs6-subset_face_train_annot_coco_style.json',
    },
    
    'cs6-subset-score': {
        IM_DIR:
            _DATA_DIR + '/CS6_annot',
        ANN_FN:
            _DATA_DIR + '/CS6_annot/cs6-subset_face_train_score-annot_coco_style.json',
    },

    # CS6 DATASET SUBSET - using GROUND-TRUTH
    'cs6-subset-gt': {
        IM_DIR:
            _DATA_DIR + '/CS6_annot',
        ANN_FN:
            _DATA_DIR + '/CS6_annot/cs6-subset_face_train_annot_coco_style.json',
    },

    # CS6 DATASET - single video overfitting (DEBUG)
    'cs6-3013-gt': {
        IM_DIR:
            _DATA_DIR + '/CS6_annot',
        ANN_FN:
            _DATA_DIR + '/CS6_annot/cs6-subset-gt_face_train_annot_coco_style_3013.json',
    },

    # CS6 DATASET FULL TRAIN
    'cs6-train-gt': {
        IM_DIR:
            _DATA_DIR + '/CS6_annot',
        ANN_FN:
            _DATA_DIR + '/CS6_annot/cs6-train-gt.json',
    },

    # CS6 DATASET NOISY TRAIN
    'cs6-train-gt-noisy-0.3': {
        IM_DIR:
            _DATA_DIR + '/CS6_annot',
        ANN_FN:
            _DATA_DIR + '/CS6_annot/cs6-train-gt_noisy-0.3.json',
    },
    'cs6-train-gt-noisy-0.5': {
        IM_DIR:
            _DATA_DIR + '/CS6_annot',
        ANN_FN:
            _DATA_DIR + '/CS6_annot/cs6-train-gt_noisy-0.5.json',
    },

    ##### CS6 GT and HP JSONS with the same images -- for supp mat noise reduction section #####
    'cs6_train_gt_same_imgs': {
        IM_DIR:
            _DATA_DIR + '/CS6_annot',
        ANN_FN:
            _DATA_DIR + '/cs6_jsons/noise_reduction_with_soft_label/cs6-train-gt_same_imgs.json',
    },
    'cs6_train_hp_same_imgs': {
        IM_DIR:
            _DATA_DIR + '/CS6_annot',
        ANN_FN:
            _DATA_DIR + '/cs6_jsons/noise_reduction_with_soft_label/cs6-train-hp_same_imgs.json',
    },
    ########################################


    # CS6 DATASET DETECTIONS DISTILLATION TRAIN
    'cs6-train-det-score': {
        IM_DIR:
            _DATA_DIR + '/CS6_annot',
        ANN_FN:
            _DATA_DIR + '/CS6_annot/cs6-train-det-score_face_train_annot_coco_style.json',
    },
    'cs6-train-det-score-0.5': {
        IM_DIR:
            _DATA_DIR + '/CS6_annot',
        ANN_FN:
            _DATA_DIR + '/CS6_annot/cs6-train-det-score-0.5_face_train_annot_coco_style.json',
    },

    'cs6-train-det-0.5': {
        IM_DIR:
            _DATA_DIR + '/CS6_annot',
        ANN_FN:
            _DATA_DIR + '/CS6_annot/cs6-train-det-0.5_face_train_annot_coco_style.json',
    },

    'cs6-train-easy-gt': {
        IM_DIR:
            _DATA_DIR + '/CS6_annot',
        ANN_FN:
            _DATA_DIR + '/CS6_annot/cs6-train-easy-gt.json',
    },
    'cs6-train-easy-gt-sub': {
        IM_DIR:
            _DATA_DIR + '/CS6_annot',
        ANN_FN:
            _DATA_DIR + '/CS6_annot/cs6-train-easy-gt-sub.json',
    },
    'cs6-train-easy-hp': {
        IM_DIR:
            _DATA_DIR + '/CS6_annot',
        ANN_FN:
            _DATA_DIR + '/CS6_annot/cs6-train-easy-hp.json',
    },
    'cs6-train-easy-det': {
        IM_DIR:
            _DATA_DIR + '/CS6_annot',
        ANN_FN:
            _DATA_DIR + '/CS6_annot/cs6-train-easy-hp.json',
    },


    # -------------------------------------------------------------------------#
    # CS6 DATASET DISTILLATION TRAIN
    # -------------------------------------------------------------------------#
    'cs6-train-hp':  {
        IM_DIR:
            _DATA_DIR + '/CS6_annot',
        ANN_FN:
            _DATA_DIR + '/CS6_annot/cs6-train-hp.json',
    },

    'cs6-train-det': {
        IM_DIR:
            _DATA_DIR + '/CS6_annot',
        ANN_FN:
            _DATA_DIR + '/CS6_annot/cs6-train-det.json',
    },

    'cs6-train-dummy':  {
        IM_DIR:
            _DATA_DIR + '/CS6_annot',
        ANN_FN:
            _DATA_DIR + '/CS6_annot/cs6-train-hp_noisy-1.00_dataset-cs6-train-dummy.json',
    },

}
