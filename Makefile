.PHONY: get_data clean_all clean_data clean_pkl

get_data: clean_data
	wget -P data https://pjreddie.com/media/files/mnist_train.csv
	wget -P data https://pjreddie.com/media/files/mnist_test.csv

clean_all: clean_data clean_pkl

clean_data:
	rm -rf data/*

clean_pkl:
	rm -rf pkl/*
