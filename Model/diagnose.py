from collections import Counter

path = r'E:\FAKE REVIEW DECTECTOR\data\reviews.txt'

with open(path, encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

print(f"Total lines in file: {len(lines)}")

# Count how many tabs each line has (header should tell us the expected count)
tab_counts = Counter(line.count('\t') for line in lines)
print("\nTab-count distribution (field_count -> how many lines):")
for count, freq in sorted(tab_counts.items()):
    print(f"  {count} tabs: {freq} lines")

print("\n--- First 3 raw lines (repr, so we see exact characters) ---")
for line in lines[:3]:
    print(repr(line[:300]))

print("\n--- Example __label1__ raw lines ---")
count = 0
for line in lines[1:]:
    if '__label1__' in line and count < 2:
        print(repr(line[:300]))
        count += 1

print("\n--- Example __label2__ raw lines ---")
count = 0
for line in lines[1:]:
    if '__label2__' in line and count < 2:
        print(repr(line[:300]))
        count += 1