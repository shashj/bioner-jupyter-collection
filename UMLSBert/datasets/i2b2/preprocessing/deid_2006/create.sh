python3 to_conll.py deid_surrogate_train_all_version2.xml > train_dev.conll
python3 split_train_dev.py
python3 to_conll.py deid_surrogate_test_all_groundtruth_version2.xml > test.txt
python3 get_labels.py
mv train.conll train.txt
mv dev.conll dev.txt