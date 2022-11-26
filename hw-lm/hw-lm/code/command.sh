
for l2 in 1 0.5 0.1 0.05 0.01 
do 
	./train_lm.py vocab-en_sp.txt log_linear_improved --lexicon ../lexicons/chars-10.txt --l2_regularization $l2 ../data/english_spanish/train/en.10K --output log_linear_en_$l2.model
	./train_lm.py vocab-en_sp.txt log_linear_improved --lexicon ../lexicons/chars-10.txt --l2_regularization $l2 ../data/english_spanish/train/sp.10K --output log_linear_sp_$l2.model

	#./fileprob.py log_linear_en.model ../data/english_spanish/dev/english/*/*
	#./fileprob.py log_linear_sp.model ../data/english_spanish/dev/spanish/*/*

	./textcat.py log_linear_en_$l2.model log_linear_sp_$l2.model 0.7 ../data/english_spanish/dev/english/*/*
	./textcat.py log_linear_en_$l2.model log_linear_sp_$l2.model 0.7 ../data/english_spanish/dev/spanish/*/*
done
