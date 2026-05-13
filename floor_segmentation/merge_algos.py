## Merging algorithm
import heapq
import numpy as np

def build_pq(rag):
    """
    Build a priority queue of all active RAG edges.

    Each queue entry is:
        (mean_edge_strength, region_a, region_b)

    The queue is ordered so that the weakest boundaries
    are popped first.

    Parameters
    ----------
    rag : RAG

    Returns
    -------
    pq : list
        Heap-based priority queue.
    """
    pq = []
    seen = set()

    assert len(rag.active) > 0
    for r in rag.active:
      assert len(rag.neighbours[r]) > 0
      for n in rag.neighbours[r]:
          a, b = sorted((r, n))
          if (a, b) in seen:
            continue
          seen.add((a, b))

          stats = rag.edge_stats[(a, b)]
          mean_strength = stats["sum"] / stats["count"]
          heapq.heappush(pq, (mean_strength, a, b))

    return pq

def agglomerative_merge(rag, threshold):
  """
  Perform agglomerative hierarchical merging on a RAG.

  The algorithm repeatedly:
    1. Selects the adjacent region pair with the weakest boundary
    2. Merges them if the boundary strength is below `threshold`
    3. Updates the RAG and priority queue

  The RAG is mutated in-place.

  Parameters
  ----------
  rag : RAG
      Region Adjacency Graph.
  threshold : float (0, 255)
      Maximum allowed mean boundary strength for merging.
      Stronger edges are treated as true region boundaries.

  Returns
  -------
  None
  """
  pq = build_pq(rag)
  print(len(pq))

  # while !pq.empty()
  while pq: 
    w, s1, s2 = heapq.heappop(pq)

    # only consider active entries
    if s1 not in rag.active or s2 not in rag.active:
      continue

    if w > threshold:
      print("threshold reached")
      break # we shouldn't merge hard edges and should stop

    # merge in rag class
    s_new = rag.merge(s1, s2)

    # do update for heap here 
    for neighbour in rag.neighbours[s_new]:
      a, b = sorted((neighbour, s_new))
      stats = rag.edge_stats[(a, b)]
      w = stats["sum"] / stats["count"]
      heapq.heappush(pq, (w, a, b))

def agg_merge_dominant_area(rag, area_stats, dom_proportion=0.35, check_freq=5):
  # alternative merging until there is one dominant area only check per X steps
  pq = build_pq(rag)
  step = 0

  total_area = np.sum(np.array(list(area_stats.values())))

  # while !pq.empty()
  while pq: 
    step = (step + 1) % check_freq
    w, s1, s2 = heapq.heappop(pq)

    # only consider active entries
    if s1 not in rag.active or s2 not in rag.active:
      continue
    
    a = 0
    if (step % check_freq == 0): 
      a = rag.compute_largest_area() / total_area
    elif len(rag.cachedLargestAreas) > 0:
      a = rag.cachedLargestAreas[0] / total_area
      
    if a > dom_proportion and step == 0:
      print("dom prop reached")
      break # we shouldn't merge hard edges and should stop

    # merge in rag class
    s_new = rag.merge(s1, s2)

    # do update for heap here 
    for neighbour in rag.neighbours[s_new]:
      a, b = sorted((neighbour, s_new))
      stats = rag.edge_stats[(a, b)]
      w = stats["sum"] / stats["count"]
      heapq.heappush(pq, (w, a, b))