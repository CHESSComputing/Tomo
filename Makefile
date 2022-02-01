.PHONY: test env clean deploy

conda_base = /nfs/chess/sw/miniconda3_msnc
conda_user = chess_msnc

test:
	# Create fresh test environment
	ksu $(conda_user) -e /usr/bin/cp environment.yml $(conda_base)/tomo_test.yml
	ksu $(conda_user) -e $(conda_base)/bin/conda env remove -n tomo_test
	ksu $(conda_user) -e $(conda_base)/bin/conda env create -n tomo_test -f $(conda_base)/tomo_test.yml
	# Run test
	/usr/bin/cp tests/config.txt config.txt
	/usr/bin/rm -rf tests/output/actual
	/usr/bin/mkdir tests/output/actual
	$(conda_base)/envs/tomo_test/bin/python -m unittest tests/test_tomo.py
	# Remove the test environment
	ksu $(conda_user) -e $(conda_base)/bin/conda env remove -n tomo_test
	ksu $(conda_user) -e /usr/bin/rm $(conda_base)/tomo_test.yml
	/usr/bin/rm config.txt

env:
	ksu $(conda_user) -e /usr/bin/cp environment.yml $(conda_base)/tomopy.yml
	ksu $(conda_user) -e $(conda_base)/bin/conda env remove -n tomopy
	ksu $(conda_user) -e $(conda_base)/bin/conda env create -f $(conda_base)/tomopy.yml

clean:
	find . -name "*~" -delete
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "tests/output/actual" -delete

deploy: clean env
