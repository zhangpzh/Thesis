#! /bin/bash

# all
#n_hidden_list="1024 2048"
#begin_learning_rate_list="0.01 0.001"
#coef_rnn_list="1 2"
#distribution_solution="1 2 3 4"

# used for computing confusion matrice
n_hidden_list="1024"
begin_learning_rate_list="0.001"
coef_rnn_list="1 2"
distribution_solution="1 2 3 4"

for l in $distribution_solution;
do
	for i in $n_hidden_list;
	do
		for j in $begin_learning_rate_list;
		do
			for k in $coef_rnn_list;
			do
				python single_rnn_tf.py $i $j $k $l
			done
		done
	done
done
