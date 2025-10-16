# format_to_list.py
import sys, re

# read from file or stdin ("-" means stdin)
src = sys.argv[1] if len(sys.argv) > 1 else "-"
text = sys.stdin.read() if src == "-" else open(src, "r").read()

# grab first two numbers on each line (supports scientific notation)
pairs = []
for ln in text.splitlines():
    nums = re.findall(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', ln)
    if len(nums) >= 2:
        pairs.append((float(nums[0]), float(nums[1])))

# pick widths based on data so columns align
def w_for(col):
    return max(len(f"{v:.6f}") for v in col) + 1  # +1 for a leading space
w1 = w_for([x for x, _ in pairs])
w2 = w_for([y for _, y in pairs])

prec = 6  # decimals; change to 8/10 if you want more

lines = ["data = ["]
for x, y in pairs:
    lines.append(f"    ({x: {w1}.{prec}f}, {y: {w2}.{prec}f}),")
lines.append("]")

print("\n".join(lines))
