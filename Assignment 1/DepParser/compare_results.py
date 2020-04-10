from subprocess import Popen, PIPE


# Get output from script
print("Compute Results")
print("---------------------------------")
print('Running "compute_correct_move.sh"')
print("...Well actually running: ")
cmd = "python dep_parser.py -m en-ud-dev.conllu".split(" ")
print()
print(" ".join(cmd))
print()
print("(Make sure the python environment is corret)")
print("---------------------------------")

p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
output, err = p.communicate(b"input data that is passed to subprocess' stdin")
rc = p.returncode

# Decode byte output to list of ints

rows = output.decode().split("\n")
new_rows = []
for i, row in enumerate(rows):
    new_row = []
    for char in row:
        if char.isdigit():
            new_row.append(int(char))
    new_rows.append(new_row)


# Load correct file
correct_file = "correct_moves_en-ud-dev.conllu"
print(f"Load correct file: {correct_file}")
print("(Make sure you are in the correct location/dir)")

# f = open(correct_file)
crows = []
with open(correct_file) as f:
    for row in f.readlines():
        new_row = []
        for char in row:
            if char.isdigit():
                new_row.append(int(char))
        crows.append(new_row)

# Calculate Accuracy
correct = 0
total = 0
for guess, corr in zip(new_rows, crows):
    total += 1
    if guess == corr:
        correct += 1

acc = correct / total
print("----------------------------------------------")
print(f"Acc:  {acc}, ({correct}/{total})")
print("----------------------------------------------")
