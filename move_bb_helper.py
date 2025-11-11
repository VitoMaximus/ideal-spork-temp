import sys, re
p = "Trading.py"
s = open(p).read()

start_marker = "# --- BB Squeeze + Band Position helper ---"
end_marker   = "# --- END helper ---"

i = s.find(start_marker)
if i == -1:
    print("helper markers not found"); sys.exit(1)
j = s.find(end_marker, i)
if j == -1:
    print("end marker not found"); sys.exit(1)
j += len(end_marker)

block = s[i:j].rstrip("\n")
s2 = s[:i] + s[j:]  # remove helper block

k = s2.find("\nif __name__ == \"__main__\":")
if k == -1:
    # fallback: put at end (still fine)
    k = len(s2)

# insert helper BEFORE the __main__ guard
s3 = s2[:k] + "\n" + block + "\n" + s2[k:]
open(p, "w").write(s3)
print("moved helper before __main__")
