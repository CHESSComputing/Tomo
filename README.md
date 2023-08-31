## Code is obsolete. It is now incorporated in ChessAnalysisPipeline

## Creating conda environment

 * Use the file environment.yml to create the environment  (note: you can change the name of the environment from "tomopy" to another name, but then you will need to use that name in the following commands when activating environment)

        conda env create -f environment.yml

 * The new environment, tomopy should be present

        conda info --envs

 * Activate the new environment

        conda activate tomopy
        
        depending on your conda installation, it may be necessary to use the alternative:
        source activate tomopy

 * then use commandline and run 

         python tomo.py

 * when finished, deactivate the environment

         conda deactivate
 
* should something go wrong, the environment can be removed using

        conda remove --name tomopy --all
