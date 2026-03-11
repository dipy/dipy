import numpy as np


def calculate_entropy(image, bins=32):
    """Calculates 1D Shannon Entropy."""
    pixels = image.flatten()
    counts, bin_edges = np.histogram(pixels, bins=bins, range=(0, 255), density=True)
    bin_widths = np.diff(bin_edges)
    p = counts * bin_widths

    safe_p = p[p > 0]
    return -np.sum(safe_p * np.log2(safe_p))

def calculate_joint_entropy(img_a, img_b, bins=32):
    """
    Calculates 2D Joint Entropy H(A, B).
    This measures the uncertainty in the combined system.
    """
    ha = img_a.flatten()
    hb = img_b.flatten()

    joint_counts, _, _ = np.histogram2d(ha, hb, bins=bins, range=[[0, 255], [0, 255]])

    p_joint = joint_counts / np.sum(joint_counts)

    # using shanon entropy for 2d
    safe_p = p_joint[p_joint > 0]
    return -np.sum(safe_p * np.log2(safe_p))

def calculate_mutual_information(img_a, img_b, bins=32):
    """
    MI(A, B) = H(A) + H(B) - H(A, B)
    Higher MI = Better alignment.
    """
    h_a = calculate_entropy(img_a, bins=bins)
    h_b = calculate_entropy(img_b, bins=bins)
    h_ab = calculate_joint_entropy(img_a, img_b, bins=bins)

    return h_a + h_b - h_ab

if __name__ == "__main__":
    brain_a = np.zeros((64, 64, 64))
    brain_a[:32, :, :] = 255

    brain_b = np.zeros((64, 64, 64))
    brain_b[5:37, :, :] = 255

    print("Project 2 MI analysis")

    ent_a = calculate_entropy(brain_a)
    joint_ent_self = calculate_joint_entropy(brain_a, brain_a)
    mi_self = calculate_mutual_information(brain_a, brain_a)

    print(f"Entropy H(A):        {ent_a:.4f}")
    print(f"Joint H(A, A):       {joint_ent_self:.4f} (Should match H(A)!)")
    print(f"MI (Self):           {mi_self:.4f} (Max information)")

    mi_shifted = calculate_mutual_information(brain_a, brain_b)
    print(f"MI (Shifted):        {mi_shifted:.4f} (Information lost due to shift)")
