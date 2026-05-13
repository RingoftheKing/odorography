import numpy as np

# Getting all active regions
def build_final_labels(rag, superpixel_labels):
  """
  Map original superpixel labels to final merged region labels.
  """
  H, W = superpixel_labels.shape
  final_labels = np.zeros((H, W), dtype=np.int32)
  

  # For each pixel in the image, we map it to the correct region
  # substep 1, we only know which superpixel each pixel belongs to, thus we must do the
  # superpixel -> region mapping
  sp_to_region = {}
  for region, superpixels in rag.components.items():
    for sp in superpixels:
      sp_to_region[sp] = region # treat map like function now

  # substep 2, now we can map each pixel to region
  for y in range(H):
    for x in range(W):
      # get the superpixel a pixel belongs to
      sp = superpixel_labels[y, x]
      final_labels[y, x] = sp_to_region[sp]

  return final_labels


def color_regions(final_labels):
    H, W = final_labels.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)

    colors = {}
    for r in np.unique(final_labels):
        colors[r] = np.random.randint(0, 255, size=3)

    for y in range(H):
        for x in range(W):
            out[y, x] = colors[final_labels[y, x]]

    return out