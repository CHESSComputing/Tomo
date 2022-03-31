.PHONY: test env clean deploy

conda_base = /nfs/chess/sw/miniconda3_msnc
conda_user = chess_msnc

test:
	# Create fresh test environment
	ksu $(conda_user) -e /usr/bin/cp environment.yaml $(conda_base)/tomo_test.yaml
	ksu $(conda_user) -e $(conda_base)/bin/conda env remove -n tomo_test
	ksu $(conda_user) -e $(conda_base)/bin/conda env create -n tomo_test -f $(conda_base)/tomo_test.yaml
	# Run tests
	/usr/bin/rm -rf tests/test1/output/actual
	/usr/bin/mkdir tests/test1/output/actual
	/usr/bin/rm -rf tests/test2/output/actual
	/usr/bin/mkdir tests/test2/output/actual
	$(conda_base)/envs/tomo_test/bin/python -m unittest tests/test_tomo.py
	# Remove the test environment
	ksu $(conda_user) -e $(conda_base)/bin/conda env remove -n tomo_test
	ksu $(conda_user) -e /usr/bin/rm $(conda_base)/tomo_test.yaml

env:
	ksu $(conda_user) -e /usr/bin/cp environment.yaml $(conda_base)/tomopy.yaml
	ksu $(conda_user) -e $(conda_base)/bin/conda env remove -n tomopy
	ksu $(conda_user) -e $(conda_base)/bin/conda env create -f $(conda_base)/tomopy.yaml
	ksu $(conda_user) -e /usr/bin/cp environment-galaxy.yaml $(conda_base)/tomopy-galaxy.yaml
	ksu $(conda_user) -e $(conda_base)/bin/conda env remove -n tomopy-galaxy
	ksu $(conda_user) -e $(conda_base)/bin/conda env create -f $(conda_base)/tomopy-galaxy.yaml

clean:
	find . -name "*~" -delete
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find tests -type d -name "actual" -delete

deploy: clean env
