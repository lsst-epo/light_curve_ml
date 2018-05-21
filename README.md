# Machine learning for light curves. 
Currently focusing on broad classification of astronomical objects. Project in 
hibernation as of May 2018. Like a bear.

# Requirements
- Python 3
- SQLite command line tool (optional)

## Local install
- Set `LCML` environment variable to repo checkout's path 
(e.g., `export LCML=/Users/*/code/light_curve_ml`)
- `cd $LCML && pip install -e . --user`

## AWS Ubuntu install
See instructions in `conf/dev/ubuntu_install.txt`

# Running ML pipeline
Supervised and unsupervised machine learning pipelines are run via the 
`run_pipeline.py` entry point. It expects the path to a job (config) file and 
file name for logger output. For example:

`python3 lcml/pipeline/run_pipeline.py --path conf/local/supervised/macho.json
--logFileName super_macho.log`

## Job File
The pipeline expects a job file (`macho.json` in above example) specifying the 
configuration of the pipeline and detailed declaration of experiment parameters.

The specified job file supercedes and overrides the default job file 
(`conf/common/pipeline.json`) on a per field basis recursively. So any, or none,
of the default fields may be overridden. The default settings are located at 
`conf/common/pipeline.json`. 
 
### Sections
Job files have the following structure:
- `globalParams` - Parameters used across multiple pipeline stages
- `database` - All db config and table names
- `loadData` - Stage coverting raw data into coherent light curves 
- `preprocessData` - Stage cleaning and preprocessing light curves
- `extractFeatures` - Stage extracting features from cleaned light curves
- `postprocessFeatures` - Stage further processing extracted features
- `modelSearch` - Stage testing several ML models with differing hyperparameters
    - `function` - search function name
    - `model` - ML model spec including non-searched parameters
    - `params` - parameters controlling the model search
- `serialization` - Stage persisting ML model and metadata to disk

Pipeline 'stages' are customizable processors. Each stage definition has the 
following components:
- `skip` - Boolean determining whether stage should execute
- `params` - stage-specific parameters
- `writeTable` - name of db table to which output is written 

### Example Jobs
Some representative job files provided in this repo include:
- `local/supervised/fast_macho.json` - Runs tiny portion of MACHO dataset 
through all supervised stages. Useful for pipeline debugging and for integration
 testing.
- `local/supervised/macho.json` - Full supervised learning pipeline for MACHO 
dataset. Uses `feets` library for feature extraction and random forests for 
classification.
- `local/supervised/ogle3.json` - Ditto for OGLE3
- `local/unsupervised/macho.json` - Unsupervised learning pipeline for MACHO 
focused on Mini-batch KMeans and Agglomerative clustering

# Other Scripts
- `lcml.data.acquisistion` - Scripts used to acquire and/or process various 
datasets including MACHO, OGLE3, Catalina, and Gaia
- `lcml.poc` - One-off proof-of-concept scripts for various libaries

# Logging Config
The `LoggingManager` class allows for convenient customization of Python Logger 
objects. The default Logging config is specified `conf/common/logging.json`. 
This config should contain the following main keys:
- `basicConfig` - values passed to `logging.basicConfig`
- `handlers` - handler definitions with a `type` attribute, which may be 
 either `stream` or `file`
- `modules` - list of module specific logger level settings

Main modules should initialize the manager by invoking `LoggingManager.initLogging` 
at the start of execution before logger objects have been created.
