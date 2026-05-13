from collections import defaultdict
import cv2
import numpy as np
from img_preprocess import get_sobel_edge_map, get_superpixel_img


class RAG:
  """
  Region Adjacency Graph (RAG) for agglomerative segmentation.

  The graph stores:
    - neighbours: region -> set of adjacent regions
    - edge_stats: (a, b) -> {"sum", "count"} boundary statistics
    - active:     set of currently active region ids
    - components: dict mapping merged components to their original constituting superpixels

  Pixel-level data is NOT stored here; all operations are performed
  using aggregated boundary statistics.

  The User is responsible for setting active regions and constructing an initial component tree.
  """
  @staticmethod
  def get_superpixels_and_edge_map(img_path):
    superpixel_img = get_superpixel_img(img_path)
    edge_map = get_sobel_edge_map(img_path)
    return superpixel_img, edge_map

  @staticmethod
  def create_boundary_stats(superpixel_img, edge_map):
    """
    Compute boundary statistics between adjacent superpixels.

    This function scans the superpixel label image and detects boundaries
    where neighboring pixels belong to different superpixels. For each
    adjacent superpixel pair (i, j), it accumulates:

        - sum:   total edge strength along their shared boundary
        - count: number of boundary pixels (boundary length)

    These statistics allow later computation of a *perimeter-weighted*
    mean edge strength:
        mean = sum / count
    
    Use Case
    ---------
    Use this function to generate edge_stats required for the RAG class

    Parameters
    ----------
    superpixel_img : (H, W) ndarray of int
        Superpixel label image (e.g. output of SLIC).
    edge_map : (H, W) ndarray
        Edge strength image (e.g. Canny, Sobel magnitude).

    Returns
    -------
    stats : dict
        Mapping (a, b) -> {"sum": float, "count": int},
        where a < b are superpixel ids.
    """
    assert superpixel_img.shape == edge_map.shape

    stats = defaultdict(lambda: {"sum": 0.0, "count": 0})

    H, W = superpixel_img.shape
    for y in range(H - 1):
        for x in range(W - 1):
            s = superpixel_img[y, x]

            # right neighbor
            s_r = superpixel_img[y, x + 1]
            if s != s_r:
                a, b = sorted((s, s_r))
                stats[(a, b)]["sum"] += edge_map[y, x]
                stats[(a, b)]["count"] += 1

            # bottom neighbor
            s_d = superpixel_img[y + 1, x]
            if s != s_d:
                a, b = sorted((s, s_d))
                stats[(a, b)]["sum"] += edge_map[y, x]
                stats[(a, b)]["count"] += 1
    
    return stats

  def __init__(self, img_path):
    """
    Initialize RAG from precomputed boundary statistics.

    rag.active and rag.components MUST be set by user outside of init

    Parameters
    ----------
    img_path: str
        Path to input image, used to compute superpixels and edge map for boundary stats.
      
    """
    self.neighbours = defaultdict(set)
    self.edge_stats = dict()
    self.active = set()
    self.components = {} # region_id -> set of original superpixels
    self.cachedLargestAreas = [] # can have arbitrary number of top k areas
    self.area_stats = None

    superpixel_img, edge_map = RAG.get_superpixels_and_edge_map(img_path)
    self.edge_stats = RAG.create_boundary_stats(superpixel_img, edge_map)

    assert self.edge_stats is not None and len(self.edge_stats) > 0, "Edge stats should be computed and non-empty."
    for (a, b), stats in self.edge_stats.items():
      self.add_edge(a, b, stats["sum"], stats["count"])

  def add_region(self, r):
    self.active.add(r)

  def add_edge(self, s1, s2, sum_, count_):
    a, b = sorted((s1, s2))
    self.neighbours[a].add(b)
    self.neighbours[b].add(a)
    self.edge_stats[(a, b)] = {"sum": sum_, "count": count_}

  def compute_largest_area(self, k=0):
    """
    Compute the largest areas in the current state of the RAG and store it in cache.

    Parameter
    ----------
    k : int
      calculate the top k areas.
    """
    assert self.area_stats is not None
    self.cachedLargestAreas = [] # must be reset every computation
    for cmp in self.components:
      # sum up each component by matching with area_stats
      area_of_cmp = 0
      for sp in self.components[cmp]:
        # imagine if you refactored components into a sp class
        area_of_cmp += self.area_stats[sp]

      self.cachedLargestAreas.append(area_of_cmp)  
    
    self.cachedLargestAreas.sort(reverse=True)
    return self.cachedLargestAreas[0]
  
  def get_largest_area_label(self):
    """
    Get the label of the largest area in the current state of the RAG.

    Returns
    -------
    largest_area_label : int
        The region id corresponding to the largest area.
    """
    assert self.cachedLargestAreas, "Largest areas not computed. Call compute_largest_area() first."
    largest_area = self.cachedLargestAreas[0]
    
    for cmp in self.components:
      area_of_cmp = 0
      for sp in self.components[cmp]:
        area_of_cmp += self.area_stats[sp]

      if area_of_cmp == largest_area:
        return cmp
    
    raise ValueError("Largest area label not found.")


  def merge(self, s1, s2) -> int:
    """
    Merge two active regions into a new region.

    Boundary statistics are updated as follows:
      - For shared neighbors k:
          new_sum   = sum(s1,k) + sum(s2,k)
          new_count = count(s1,k) + count(s2,k)
      - For non-shared neighbors:
          inherit boundary stats unchanged.

    Parameters
    ----------
    s1, s2 : int
        Region ids to merge (must be active).

    Returns
    -------
    s_new : int
        Newly created region id.
    """
    # Invariants
    assert s1 in self.active and s2 in self.active
    assert s1 in self.components and s2 in self.components
    assert s1 != s2
    assert s2 in self.neighbours[s1]
    assert s1 in self.neighbours[s2]

    # since s1 and s2 are neighbours, there should be edge_stats
    key = tuple(sorted((s1, s2)))
    assert key in self.edge_stats

    # all components in s1 should really be disjoint from s2 before a merge
    assert self.components[s1].isdisjoint(self.components[s2])


    s_new = max(self.active) + 1
    self.active.add(s_new)

    neighbors = (self.neighbours[s1] | self.neighbours[s2]) - {s1, s2}

    for k in neighbors:
        key1 = tuple(sorted((s1, k)))
        key2 = tuple(sorted((s2, k)))

        if key1 in self.edge_stats and key2 in self.edge_stats:
            # shared neighbor → add stats
            sum_ = self.edge_stats[key1]["sum"] + self.edge_stats[key2]["sum"]
            count_ = self.edge_stats[key1]["count"] + self.edge_stats[key2]["count"]
        else:
            # single neighbor → inherit
            key = key1 if key1 in self.edge_stats else key2
            sum_ = self.edge_stats[key]["sum"]
            count_ = self.edge_stats[key]["count"]

        self.add_edge(s_new, k, sum_, count_)
    
    # new region should track what original superpixels construct it
    self.components[s_new] = self.components[s1] | self.components[s2]

    # cleanup
    foo = self.components.pop(s1)
    bar = self.components.pop(s2)

    self.active.remove(s1)
    self.active.remove(s2)
    return s_new
