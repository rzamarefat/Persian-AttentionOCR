from tqdm import tqdm

path_to_gt_file = "/home/rmarefat/Desktop/git/OCR-data-generator/gt/gt.txt"

with open(path_to_gt_file) as h:
    content = [l.split(" --> ")[-1].replace("\n", "") for l in h.readlines()]


chars = []

for c in tqdm(content):
    chars.extend(list(c))

print("".join(set(chars)))