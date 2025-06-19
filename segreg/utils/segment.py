import numpy as np
from skimage import measure, segmentation
from sklearn import metrics


def calculate_confusion_matrix_metrics(seg1, seg2):
    """
    Calculates the True Positive Rate (TPR) and Jaccard Index
    from the confusion matrix.

    Args:
        seg1: The first segmentation mask.
        seg2: The second segmentation mask.

    Returns:
        A tuple containing the TPR and Jaccard Index.
    """

    seg1_flat = seg1.flatten()
    seg2_flat = seg2.flatten()
    cm = metrics.confusion_matrix(seg1_flat == 255, seg2_flat == 255)
    cm_flat = cm.flatten()

    tpr = cm_flat[3] / np.sum(seg1_flat == 255)
    jaccard = cm_flat[3] / (cm_flat[1] + cm_flat[2] + cm_flat[3])

    return tpr, jaccard


def calculate_object_overlap_ratio(seg1, seg2):
    """
    Calculates the Object Overlap Ratio (OOR) between two segmentations.

    Args:
        seg1: The first segmentation mask.
        seg2: The second segmentation mask.

    Returns:
        The Object Overlap Ratio.
    """

    seg1_label = measure.label(seg1)
    seg2_label = measure.label(seg2)
    n_match_obj = 0
    seg1_label_ids = list(np.nonzero(np.unique(seg1_label))[0])
    for seg1_id in seg1_label_ids:
        seg2_corrids = [
            num for num in np.unique(seg2_label[np.where(seg1_label == seg1_id)]) if num
        ]
        if len(seg2_corrids) > 0:
            n_match_obj += 1

    oor = n_match_obj / np.max(seg1_label)

    return oor


def calculate_metrics(seg1, seg2):
    """
    Calculates multiple metrics between two segmentations.

    Args:
        seg1: The first segmentation mask.
        seg2: The second segmentation mask.

    Returns:
        A tuple containing the TPR, Jaccard Index, and OOR.
    """

    tpr, jaccard = calculate_confusion_matrix_metrics(seg1, seg2)
    oor = calculate_object_overlap_ratio(seg1, seg2)

    return tpr, jaccard, oor


def cross_entropy(label1, label2, img1, img2, weight_bg_to_obj):
    """Calculates the weighted cross-entropy loss between two segmentation masks.

    Args:
        label1: The first segmentation mask.
        label2: The second segmentation mask.
        img1: The first image.
        img2: The second image.
        weight_bg_to_obj: The weight factor for the background class.

    Returns:
        The weighted cross-entropy loss.
    """

    seg1 = (label1 > 1).astype(np.uint8) * 255
    seg2 = (label2 > 1).astype(np.uint8) * 255
    img1 = img1 / np.max(img1)
    img2 = img2 / np.max(img2)

    # Flatten the arrays.
    seg1_flat = (seg1 / 255).flatten()
    seg2_flat = (seg2 / 255).flatten()
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()

    # Calculate the number of nonzero pixels.
    img_flat = img1_flat * img2_flat
    num_nonzero_pix = len(np.nonzero(img_flat)[0])

    # Clip the values to prevent numerical instability.
    seg1_flat = np.clip(seg1_flat, 0.001, 0.999)
    seg2_flat = np.clip(seg2_flat, 0.001, 0.999)
    img1_flat = np.clip(img1_flat, 0.001, 0.999)
    img2_flat = np.clip(img2_flat, 0.001, 0.999)

    # Calculate the weighted cross entropy.
    # weighted_xent = -(seg1_flat * np.log(seg2_flat) + weight_bg_to_obj * (1 - seg1_flat) * np.log(1 - seg2_flat)).mean()

    seg1_flat_clipped = np.clip(seg1_flat, 1e-8, 1 - 1e-8)
    seg2_flat_clipped = np.clip(seg2_flat, 1e-8, 1 - 1e-8)
    loss_positive = seg1_flat_clipped * np.log(seg1_flat_clipped)
    loss_negative = (1 - seg1_flat_clipped) * np.log(1 - seg2_flat_clipped)
    weighted_loss = loss_positive + weight_bg_to_obj * loss_negative
    weighted_xent = -weighted_loss.mean()

    return weighted_xent


def norm_2d(img):
    """
    Normalizes a 2D image to the range [0, 255].

    Args:
        img (np.ndarray): The input image.

    Returns:
        np.ndarray: The normalized image.
    """

    epsilon = np.finfo(float).eps

    # Calculate the maximum and minimum values of the image.
    maxval = np.max(img)
    minval = np.min(img)

    # Normalize the image to the range [0, 1].
    oimg = (img - minval) / (maxval - minval + epsilon)

    # Scale the image to the range [0, 255].
    oimg = oimg * 255

    # Cast the image to uint8.
    oimg = oimg.astype(np.uint8)

    return oimg


def overlap_images(img1, img2):
    """
    Overlays two grayscale images, combining them into a single RGB image.

    Args:
        img1 (np.ndarray): The first grayscale image.
        img2 (np.ndarray): The second grayscale image.

    Returns:
        np.ndarray: The combined RGB image.
    """
    img12 = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)

    img12[..., 0] = (img2 / np.max(img2) * 255).astype(np.uint8)
    img12[..., 1] = (img1 / np.max(img1) * 255).astype(np.uint8)

    return img12


def relabel(label, offset=2):
    """
    Relabels the connected components in a segmentation mask.

    Args:
        label: The input segmentation mask.
        offset: The starting label value.

    Returns:
        The relabeled segmentation mask.
    """
    # Flatten the label image.
    label_flattened = label.flatten()

    # Relabel the label image.
    relab, fw, inv = segmentation.relabel_sequential(label_flattened, offset)

    # Reshape the relabeled label image.
    relab = relab.reshape(label.shape)

    # Cast the relabeled label image to uint8.
    relab = relab.astype(np.uint8)

    return relab


def seg2label(seg):
    """
    Converts a segmentation mask to a label image.

    Args:
        seg: The input segmentation mask.

    Returns:
        The relabeled segmentation mask.
    """
    # Change instance segmentation to semantic segmentation
    seg = ((seg > 0) * 255).astype(np.uint8)

    # Change semantic segmentation to label starting [0, 2, 3, ...]
    objects = measure.label(seg).astype(np.uint8)
    objects = relabel(objects, offset=2)

    return objects
