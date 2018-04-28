# Machine learning for light curves. 
Currently focusing on broad classification of astronomical objects.

Requirements:
- Python 3

Development install
- Set `LCML` environment variable to repo checkout's path 
(e.g., `export LCML=/Users/*/code/light_curve_ml`)
- `cd $LCML && python setup.py develop`

# Running ML pipeline
`python lcml/pipeline/run_pipeline.py --path conf/local/supervised/macho.json
--logFileName super_macho.log`

## Job File
The pipeline expects a job file (`macho.json` in above example) specifying the 
configuration of the pipeline and detailed declaration of experiment parameters.

The specified job file supercedes the default job file on a per field basis recursively. 
So any, or none, of the default fields may be overridden. The default settings are located
 at `conf/common/pipeline.json`. 

# Other Scripts
- `lcml.data.acquisistion` - Scripts used to acquire and/or process various datasets including MACHO, OGLE3, Catalina, and Gaia
- `lcml.poc` - One-off proof-of-concept scripts of how to use various libaries
