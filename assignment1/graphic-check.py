import sys


def print_fail(deg_seq_file, msg, is_original):
    """
    Prints the given failure message.
    :param deg_seq_file: degree sequence file being examined
    :param msg: failure message to print
    :param is_original: flag indicating if it is a sequence or a sub-sequence
    :return: void
    """
    if is_original:
        print "{} is NOT a graphic degree sequence. It fails test: {}.".format(deg_seq_file, msg)
    else:
        print "{} is NOT a graphic degree sequence. It fails test: Havel-Hakimi: {} for a sequence.".format(deg_seq_file, msg)


def havil_hakimi(deg_seq_file, deg_seq, is_first_iter=True):
    """
    Prints if the provided degree sequence is graphical or not. If not, also prints the reason for failure.
    :param deg_seq_file: degree sequence file being examined.
    :param deg_seg: deg sequence loaded from the deg_seq_file.
    :param is_original: Flag indicating first iteration of the algorithm.
    :returns void
    """
    for _ in range(len(deg_seq)):
        if deg_seq[0] >= len(deg_seq):
            print_fail(deg_seq_file, "Max degree is greater than the number of nodes", is_first_iter)
            return
        elif sum(deg_seq) % 2 != 0:
            print_fail(deg_seq_file, "Sum of degree sequence is not even", is_first_iter)
            return
        elif len(deg_seq) == len(set(deg_seq)):
            print_fail(deg_seq_file, "At least 2 nodes must have the same degree", is_first_iter)
            return
        elif len(deg_seq) <= 2:
            for d in deg_seq:
                if d < 0:
                    print_fail(deg_seq_file, "Node degree is negative", is_first_iter)
                    return
        else:
            max_deg = deg_seq[0]
            deg_seq = deg_seq[1:]

            for i in range(max_deg):
                deg_seq[i] -= 1

            deg_seq = sorted(deg_seq, reverse=True)
            is_first_iter = False

    print "{} is a graphic degree sequence.".format(deg_seq_file)


def test_cases():
    test_sequences =[
        [4, 4, 3, 3],               # max degree
        [3, 2, 1, 1],               # sum is odd
        [7, 5, 5, 4, 4, 4, 4, 3],   # degree sequence
        [3, 3, 3, 1],               # -ve degree
        [-2, -2]                    # -ve degree
    ]

    for index, t in enumerate(test_sequences):
        havil_hakimi("test #{}".format(index), t)


def main(argv):
    if len(argv) != 1:
        print "usage: python graphic-check.py <path/to/degree_sequence_file>"
        sys.exit(0)

    deg_seq_path = argv[0]
    deg_seq_file = deg_seq_path.split('/')[-1]

    deg_seq = []

    # Read the Degree Sequence File into a list
    try:
        with open(deg_seq_path, 'r') as ds:
            for d in ds:
                deg_seq.append(int(d.strip()))
    except IOError:
        print "No such file: {}".format(deg_seq_path)
        sys.exit(0)

    # Uncomment to run basic test cases
    # test_cases()

    #  Perform the Havel-Hakimi test to check if a degree sequence is graphical
    havil_hakimi(deg_seq_file, deg_seq)


if __name__ == "__main__":
    main(sys.argv[1:])
