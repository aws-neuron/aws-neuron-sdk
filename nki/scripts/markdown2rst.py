import re
import sys
from m2r import convert

args = sys.argv
assert len(args) == 3

# Useful regex engine for testing: https://www.regexpal.com/

# Gets user input
# filename = input("What's the file you want to insert line breaks?")
# new_filename = input("What's the file you want to write results to?")
filename = args[1]
new_filename = args[2]

f = open(filename, "r")
text = f.read()
f.close()

# Step 1: run m2r tool to convert markdown to sphinx
text = convert(text)

# Step 2:
# Replace image with sphinx figure directive
# There can be two formats for images coming out of Quip markdown
pattern1 = r"\[Image: image\.png\]\n(.*)\n"
pattern2 = r"\[Image: image\.png\](.*)\n"

replacement=r'''
.. _<FIXME>:

.. figure:: img/<FIXME>.png
   :align: center
   :width: 60%

   \1
'''

text = re.sub(pattern1, replacement, text)
text = re.sub(pattern2, replacement, text)

# Replace code browser URL
pattern_url = r"https:\/\/prod\.artifactbrowser\.brazil\.aws\.dev(\S*)_build\/html\/"
replacement= ""

text = re.sub(pattern_url, replacement, text)

# Step 3.
# Insert line breaks
text = text.split("\n")
max_char = 120

split_lines = []
for line in text:
    words = line.replace("\n", "").split(" ")
    i_words = 0

    while (i_words < len(words)):
        buffer = ""
        while(len(buffer) < 120):
            buffer += words[i_words] + " "
            i_words += 1

            if i_words >= len(words):
                break

        split_lines.append(buffer)



with open(new_filename, "w+") as f:
    f.write("\n".join(split_lines))
