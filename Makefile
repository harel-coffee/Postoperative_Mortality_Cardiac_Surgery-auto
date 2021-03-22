clean-results:
	@echo "Cleaning model saves and plots..."
	@rm -rvf training_data/*
	@rm -rvf plots/*

setup-conda-environment:
	@conda env create -f carola.yml
	@source ~/anaconda3/etc/profile.d/conda.sh
	@conda activate CAROLA-gpu

train_model_total:
	@echo "Train total model with random500 dataset"
	@python predictor.py --dataset_type random500 --model_type total
