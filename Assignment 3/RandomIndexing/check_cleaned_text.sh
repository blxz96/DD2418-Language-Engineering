#! /bin/sh
python random_indexing.py -c -co cleaned_example.txt
diff correct_cleaned_example.txt cleaned_example.txt