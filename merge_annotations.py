"""Script to merge individual annotation files generated from the synthetic data tool

This script simplifies the process of generating the annotations.json file in COCO format
from individual annotation files. It reads the individual annotation files, merges them into
a final annotation file in COCO format that can be used for training.

Command line args:
    --input-folder {str} -- folder that contains all datasets that will be part of the training pipeline

The results will be stored in each subfolder that represents a dataset. They will be stored as annotations.json
"""

import argparse
import datetime
import os, glob
import ujson
json = ujson


def parse_configs():
    """
    Read the config from the command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Folder that contains the partial annotations",
    )
    args = parser.parse_args()
    return args

def get_info():
    return {
        "description": "Synthetic data for visual product recognition",
        "url": "",
        "version": "1",
        "year": 2019,
        "contributor": "Aifi",
        "date_created": str(datetime.datetime.now())
    }

def get_licenses():
    return [
        {
          "id": 1,
          "name": "Attribution-NonCommercial-ShareAlike License",
          "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ]

def get_image_object(_id):
    return {
      "id": _id,
      "file_name": "{}".format(_id).zfill(6)+".jpeg",
      "width": 1280,
      "height": 720,
      "date_captured": str(datetime.datetime.now()),
      "license": 1,
      "coco_url": "",
      "flickr_url": ""
    }

def get_category(name, _id):
    return {
      "id": _id,
      "name": name,
      "supercategory": "skus"
    }

def get_segementation(segmentation_list):
  filtered_segmentations = []
  for segmentation in segmentation_list:
    if len(segmentation) > 5:
      filtered_segmentations.append(segmentation)
  return filtered_segmentations

args = parse_configs()
input_folder = args.input_folder
prev_categories = [
    get_category("Premium napkins", 1),
    get_category("Cuttlery", 2),
    get_category("Heavy Duty Aluminium Foil", 3),
    get_category("Reynolds Wrap", 4),
    get_category("Cling Wrap", 5),
    get_category("Sandwiches bag", 6),
    get_category("Quart Freezer", 7),
    get_category("Gallon Freezer", 8),
    get_category("Large Drawstring Trash Bags", 9),
    get_category("Flap Tie Lawn & Leaf Bags", 10),
    get_category("Tall Kitchen Bags White", 11),
    get_category("Bounty Paper Towels", 12),
    get_category("Ultra Premium Paper Towels", 13),
    get_category("Premium paper towels", 14),
    get_category("Sanitex Vinyl Gloves", 15),
    get_category("Windex Original", 16),
    get_category("Scotch Brite Spounge", 17),
    get_category("Palmolive Ultra Strenght dish liquid", 18),
    get_category("Dawn dish liquid", 19),
    get_category("7Eleven Dish Liquid", 20),
    get_category("Fabric Febreeze", 21),
    get_category("Glade Clean linen", 22),
    get_category("Glade scented Candel", 23),
    get_category("Lysol spray", 24),
    get_category("409 Multi-Surface", 25),
    get_category("Toilet bowl cleaner", 26),
    get_category("Lysol Toilet bleach", 27),
    get_category("7E Glass Cleaner", 28),
    get_category("Liquid Plumr", 29),
    get_category("Lysol all purpose cleaner Lemon", 30),
    get_category("Fabuloso Cleaner", 31),
    get_category("Raid (ANt & Roach remover)", 32),
    get_category("Pine-sol Multi-surfaces", 33)
]
categories = {c['name']: c for c in prev_categories}
category_counter = len(prev_categories)+1
#all_subdirs = [d for d in os.listdir(input_folder)]
all_subdirs = sorted(next(os.walk(input_folder))[1])

for folder in all_subdirs:
  print("Generating annotations for", folder)
  annotation_counter = 1
  results = {}
  results['info'] = get_info()
  results['licenses'] = get_licenses()
  results['categories'] = prev_categories
  results['images'] = []
  results['annotations'] = []

  files = glob.glob(os.path.join(input_folder, folder, 'annotations', '*.json'))
  files.sort()

  # Process individual annotations
  for filename in files:
    with open(filename, 'r') as f:
      print(filename)
      text = f.read()
      data = json.loads(text)
      _id = int(filename.split(".json")[0].split("/")[-1])
      results['images'].append(get_image_object(_id))
      annotations = data['annotations']
      for index, annotation in enumerate(annotations):
          annotation['id'] = annotation_counter
          category = annotation['category']
          if category not in categories:
              new_category = get_category(category, category_counter)
              categories[category] = new_category
              results['categories'].append(new_category)
              category_counter += 1
          
          annotation['segmentation'] = get_segementation(annotation['segmentation'])

          annotation['category_id'] = categories[category]['id']
          annotation_counter += 1
          del annotation['category']
          
          if len(annotation['segmentation']) > 0:
            results['annotations'].append(annotation)

  prev_categories = results['categories']
  
  # Store final annotations in folder
  with open(os.path.join(input_folder, folder, 'annotations.json'), 'w') as f:
    f.write(json.dumps(results))


