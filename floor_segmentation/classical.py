import cv2
import numpy as np
import argparse
from rag import RAG
from merge_algos import agglomerative_merge, agg_merge_dominant_area
from visual_tools import build_final_labels, color_regions
import img_preprocess as pp

from collections import defaultdict

def main(img_path):
  # 1. Create RAG
  rag = RAG(img_path)

  # 2. Get area info for all superpixels and set it in RAG
  superpixel_img = pp.get_superpixel_img(img_path)
  area_stats = pp.get_area_info(superpixel_img)
  rag.area_stats = area_stats # required for agg_merge_dominant_area

  # 2b. Set Active Regions and Components in RAG
  initial_regions = np.unique(superpixel_img)
  for r in initial_regions:
    rag.add_region(r)
    rag.components[r] = {r}

  # 3. perform merge
  agg_merge_dominant_area(rag, area_stats=area_stats, dom_proportion=0.35)

  # 4. visualize result
  final_labels = build_final_labels(rag, superpixel_img)
  color_viz = color_regions(final_labels)

  cv2.imshow("final segmentation", color_viz)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  # 5. Get Contours
  largest_label = rag.get_largest_area_label()
  mask = (final_labels == largest_label).astype(np.uint8) * 255

  # 5b. Do some basic erosion and dilation to clean up the mask
  kernel = np.ones((5, 5), np.uint8)
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contour_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

  cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
  cv2.imshow("contours", contour_img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Agglomerative segmentation for floor segmentation")
  parser.add_argument("--img_path", type=str, required=True, help="Path to input image")
  args = parser.parse_args()
  img_path = args.img_path
  main(img_path)
  