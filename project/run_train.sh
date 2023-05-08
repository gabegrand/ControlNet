export CUDA_VISIBLE_DEVICES=6,7,15

for BATCH in 2 4
do
	for LR in 5e-6 1e-5 5e-5
	do
		python train.py --batch_size ${BATCH} --lr ${LR}
	done
done
