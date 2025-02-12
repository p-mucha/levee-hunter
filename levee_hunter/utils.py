def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_splits(Z, s=256, max_overlap_frac=0.1):
    """
    Finds ways to tile dimension Z with patches of size s and overlap <= max_overlap_frac * s.

    Returns a sorted list of tuples:
        (n, overlap, total_covered)
    where:
        - n         = number of patches along one dimension
        - overlap   = integer overlap in pixels
        - total_covered = n*s - (n-1)*overlap
    """
    # Ensure s is a multiple of 32
    if s % 32 != 0:
        raise ValueError("Patch size s must be divisible by 32.")

    max_overlap_px = int(s * max_overlap_frac)
    solutions = []

    # The equation can be written as Z = n(s-o) + o
    # Which can be rearranged into n = (Z - o) // (s - o)
    # Since all are positive integers, we can get the bound by setting
    # o = 0 in the numerator and o = max_overlap_px in the denominator:
    # n <= Z // (s - max_overlap_px), then we can even add +1 to the bound
    max_n = (Z // (s - max_overlap_px)) + 1

    for n in range(1, max_n + 1):
        for overlap in range(max_overlap_px + 1):

            # Z = n*s - (n-1)*o
            total_covered = n * s - (n - 1) * overlap
            if total_covered <= Z:
                solutions.append((n, overlap, total_covered))

    # Sort primarily by how much of Z is covered (descending),
    # secondarily by overlap (ascending) so smaller overlaps appear first if coverage ties.
    solutions.sort(key=lambda x: (x[2], -x[1]), reverse=True)

    return solutions
