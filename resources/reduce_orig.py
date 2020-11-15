def reduce_cards(state_subset, fp_orig, fp_out):
    remaining_lines = []
    with open(fp_orig) as f:
        txt = f.read()
        for line in txt.splitlines():
            if any(line.startswith(state) for state in state_subset):
                remaining_lines.append(line)
    with open(fp_out, "w+") as f:
        f.write("\n".join(remaining_lines))


def reduce_board(state_subset, fp_orig, fp_out):
    remaining_lines = []
    with open(fp_orig) as f:
        txt = f.read()
        for line in txt.splitlines():
            if line.startswith("--"):
                remaining_lines.append(line)
            if any(line.startswith(state) for state in state_subset) and any(line.endswith(state) for state in state_subset):
                remaining_lines.append(line)
            if ":" in line:
                region_name, states = line.split(":")
                new_states = [st for st in states.strip().split() if st in state_subset]
                if new_states:
                    remaining_lines.append(region_name + ": " + " ".join(new_states))
    with open(fp_out, "w+") as f:
        f.write("\n".join(remaining_lines))


if __name__ == '__main__':
    state_subset = ["OR", "ID", "WY",
                    "NV", "UT", "CO"]
    reduce_cards(state_subset, "cards_orig.txt", "cards_very_small.txt")
    reduce_board(state_subset, "board_orig.txt", "board_very_small.txt")